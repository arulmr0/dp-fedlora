"""Flower-compatible FL client with optional Opacus differential privacy.

Opacus integration notes:
- `ModuleValidator.fix` is called once in __init__ to replace unsupported layers
  (e.g., any non-LayerNorm norm layers) with DP-compatible equivalents.
- A fresh PrivacyEngine is created per round. `dp_model._module is self.model`
  is always True (Opacus wraps by reference), so self.model holds updated weights
  after training. `dp_model.remove_hooks()` is called explicitly to clean up
  backward hooks before the wrapper goes out of scope.
- poisson_sampling=True is required for the RDP accountant's epsilon bound to be
  valid. This uses uniform-with-replacement sampling.
"""

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.model_module.backbone import MedicalViT
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _train_local(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    local_epochs: int,
    device: torch.device,
    multilabel: bool,
    max_batches: int = 0,
    use_amp: bool = False,
) -> tuple[float, int]:
    """Train for local_epochs; returns (mean_loss, total_examples).

    Args:
        max_batches: If >0, stop after this many batches per epoch (fast_dev_run).
        use_amp: Enable torch AMP mixed precision (2-3x speedup on Tensor Core GPUs).
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    for _ in range(local_epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(device)
            if multilabel:
                labels = labels.float().to(device)
            else:
                labels = labels.squeeze(-1).long().to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)

    return total_loss / max(total_examples, 1), total_examples


class MedMNISTClient:
    """NumPy-serialisable FL client compatible with the local sequential simulator
    and Flower's start_simulation (same interface as fl.client.NumPyClient).
    """

    def __init__(
        self,
        client_id: int,
        model: MedicalViT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        self.multilabel = cfg.dataset.get("multilabel", False)

        if cfg.privacy.enabled:
            from opacus.validators import ModuleValidator
            if model._lora_active():
                # Backbone was already fixed in build_model() before LoRA injection.
                # Opacus hooks only modules whose leaf params have requires_grad=True;
                # frozen backbone layers are skipped automatically.
                pass
            else:
                # Full fine-tuning: fix incompatible norm layers and validate.
                model = ModuleValidator.fix(model)
                errors = ModuleValidator.validate(model, strict=False)
                if errors:
                    raise RuntimeError(f"Opacus-incompatible layers remain: {errors}")

        self.model = model
        self.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self) -> list[np.ndarray]:
        return self.model.get_parameters()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.model.set_parameters(parameters)

    def fit(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[list[np.ndarray], int, dict]:
        self.set_parameters(parameters)
        metrics: dict[str, float] = {}

        # Fresh optimizer per round — standard FedAvg assumes stateless clients
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        max_batches = int(self.cfg.training.get("fast_dev_run", 0))
        use_amp = bool(self.cfg.training.get("use_amp", True))

        if self.cfg.privacy.enabled:
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine(accountant=self.cfg.privacy.accountant)
            # poisson_sampling=True: required for RDP accountant's epsilon to be valid.
            # Uses UniformWithReplacementSampler; expected batch size = len(dataset)*sample_rate.
            dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.cfg.privacy.noise_multiplier,
                max_grad_norm=self.cfg.privacy.max_grad_norm,
                poisson_sampling=True,
            )
            loss, num_examples = _train_local(
                dp_model,
                dp_optimizer,
                dp_loader,
                self.cfg.training.local_epochs,
                self.device,
                self.multilabel,
                max_batches=max_batches,
                use_amp=use_amp,
            )
            epsilon = privacy_engine.get_epsilon(self.cfg.privacy.target_delta)
            metrics["epsilon"] = float(epsilon)

            # Explicitly remove backward hooks before dp_model goes out of scope
            dp_model.remove_hooks()

            logger.debug(
                "Client %d: loss=%.4f, eps=%.3f (delta=%.0e)",
                self.client_id, loss, epsilon, self.cfg.privacy.target_delta,
            )
        else:
            loss, num_examples = _train_local(
                self.model,
                optimizer,
                self.train_loader,
                self.cfg.training.local_epochs,
                self.device,
                self.multilabel,
                max_batches=max_batches,
                use_amp=use_amp,
            )
            logger.debug("Client %d: loss=%.4f", self.client_id, loss)

        metrics["train_loss"] = float(loss)
        return self.get_parameters(), num_examples, metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict
    ) -> tuple[float, int, dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss() if self.multilabel else nn.CrossEntropyLoss()

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                if self.multilabel:
                    labels_device = labels.float().to(self.device)
                else:
                    labels_device = labels.squeeze(-1).long().to(self.device)
                logits = self.model(images)
                total_loss += criterion(logits, labels_device).item() * images.size(0)
                total_examples += images.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        eval_metrics = compute_metrics(logits_cat, labels_cat, multilabel=self.multilabel)
        return total_loss / max(total_examples, 1), total_examples, eval_metrics


def build_client_fn(
    models: list[MedicalViT],
    fed_dataset,
    cfg: DictConfig,
    device: torch.device,
) -> Callable[[str], MedMNISTClient]:
    """Returns client_fn(cid: str) -> MedMNISTClient for the FL simulator."""

    def client_fn(cid: str) -> MedMNISTClient:
        client_id = int(cid)
        return MedMNISTClient(
            client_id=client_id,
            model=models[client_id],
            train_loader=fed_dataset.get_train_loader(client_id),
            val_loader=fed_dataset.get_val_loader(client_id),
            cfg=cfg,
            device=device,
        )

    return client_fn
