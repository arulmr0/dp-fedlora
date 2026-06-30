"""MedMNIST federated dataset with Dirichlet non-IID partitioning.

Each simulated hospital (client) receives a per-class label distribution drawn
from Dirichlet(alpha). Smaller alpha = more heterogeneous (non-IID):
  alpha=0.1  → very non-IID (1–2 dominant classes per client)
  alpha=0.5  → moderate non-IID (proposal default)
  alpha=100  → near-IID

Partition correctness: every sample is assigned to exactly one client, no samples
are lost. Remainder from floor-division is distributed to clients with largest
fractional parts, following the standard Dirichlet-FL convention.
"""

import logging
from functools import lru_cache
from typing import Callable

import medmnist
import numpy as np
import torch
from medmnist import INFO
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _dataset_registry() -> dict[str, type]:
    """Build name→class map from medmnist.INFO (cached after first call)."""
    registry: dict[str, type] = {}
    for name in INFO:
        cls_name = INFO[name]["python_class"]
        if hasattr(medmnist, cls_name):
            registry[name] = getattr(medmnist, cls_name)
    return registry


def build_transforms(img_size: int, is_train: bool) -> Callable:
    """ImageNet-normalised transforms with optional light augmentation for training."""
    aug = (
        [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        if is_train
        else []
    )
    return transforms.Compose(
        aug
        + [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[int]]:
    """Partition dataset indices per client using Dirichlet(alpha).

    Every sample is assigned exactly once. Remainder samples from floor-division
    are distributed to the clients with the largest fractional parts, so the
    total count is always len(labels).

    Returns:
        List of length num_clients; each element is a list of integer indices.
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(labels)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in unique_classes:
        cls_idx = np.where(labels == cls)[0].copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        proportions = rng.dirichlet(np.full(num_clients, alpha))
        # Floor allocation
        counts = np.floor(proportions * n).astype(int)
        # Distribute remainder to clients with largest fractional parts
        remainder = n - counts.sum()
        fractional_parts = proportions * n - counts
        top_k = np.argsort(-fractional_parts)[:remainder]
        counts[top_k] += 1

        assert counts.sum() == n, "Partition sanity check failed"

        cumulative = np.cumsum(counts)
        cuts = np.concatenate([[0], cumulative])
        for k in range(num_clients):
            client_indices[k].extend(cls_idx[cuts[k] : cuts[k + 1]].tolist())

    sizes = [len(idx) for idx in client_indices]
    if min(sizes) == 0:
        raise ValueError(
            f"Client(s) {[k for k, s in enumerate(sizes) if s == 0]} received 0 samples. "
            f"Increase alpha or reduce num_clients."
        )
    logger.info(
        "Dirichlet(alpha=%.2f): %d clients, sizes min=%d max=%d mean=%.0f",
        alpha, num_clients, min(sizes), max(sizes), np.mean(sizes),
    )
    return client_indices


class MedMNISTFederated:
    """Manages MedMNIST train/val/test splits and per-client data loaders.

    Usage:
        fed = MedMNISTFederated(cfg)
        train_loader = fed.get_train_loader(client_id=3)
        test_loader  = fed.get_test_loader()
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        dataset_name = cfg.dataset.name
        registry = _dataset_registry()
        if dataset_name not in registry:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {sorted(registry.keys())}"
            )

        cls = registry[dataset_name]
        # Always download the native 28px files; the transform handles resizing to img_size.
        # This avoids downloading multi-GB 224px archives for a ~40MB dataset.
        size_arg = None
        train_tf = build_transforms(cfg.dataset.img_size, is_train=True)
        eval_tf = build_transforms(cfg.dataset.img_size, is_train=False)

        # Resolve data_dir relative to the original cwd (before Hydra changes it).
        import os
        from pathlib import Path as _Path
        data_dir = cfg.dataset.data_dir
        if not _Path(data_dir).is_absolute():
            try:
                from hydra.utils import get_original_cwd
                data_dir = str(_Path(get_original_cwd()) / data_dir)
            except Exception:
                data_dir = str(_Path(os.getcwd()) / data_dir)
        _Path(data_dir).mkdir(parents=True, exist_ok=True)

        common_kwargs = dict(download=cfg.dataset.download, root=data_dir)
        if size_arg is not None:
            common_kwargs["size"] = size_arg

        self._train_ds = cls(split="train", transform=train_tf, **common_kwargs)
        self._val_ds = cls(split="val", transform=eval_tf, **common_kwargs)
        self._test_ds = cls(split="test", transform=eval_tf, **common_kwargs)

        train_labels = np.array(self._train_ds.labels).flatten()
        self._client_indices = dirichlet_partition(
            labels=train_labels,
            num_clients=cfg.dataset.num_clients,
            alpha=cfg.dataset.alpha,
            seed=cfg.experiment.seed,
        )

    def get_train_loader(self, client_id: int) -> DataLoader:
        client_data = self._client_indices[client_id]
        if len(client_data) < self.cfg.training.batch_size:
            logger.warning(
                "Client %d has only %d samples (< batch_size=%d); "
                "consider increasing alpha or reducing num_clients.",
                client_id, len(client_data), self.cfg.training.batch_size,
            )
        subset = Subset(self._train_ds, client_data)
        return DataLoader(
            subset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,  # keep all samples; Opacus Poisson sampler handles batching
        )

    def get_val_loader(self, client_id: int) -> DataLoader:
        rng = np.random.default_rng(self.cfg.experiment.seed + client_id)
        n = len(self._val_ds)
        val_idx = rng.choice(n, size=max(1, int(n * 0.1)), replace=False).tolist()
        return DataLoader(
            Subset(self._val_ds, val_idx),
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
        )

    @property
    def num_clients(self) -> int:
        return self.cfg.dataset.num_clients

    def client_data_size(self, client_id: int) -> int:
        return len(self._client_indices[client_id])
