"""ViT backbone with optional LoRA injection (Month 3+).

Month 1:  cfg.model.use_lora=false  → full fine-tuning of all parameters
Month 3+: cfg.model.use_lora=true   → freeze backbone, train LoRA adapters only

Using HuggingFace ViTForImageClassification so PEFT LoRA can be added with a
two-line change (see build_model). BiomedCLIP or other healthcare-pretrained VLMs
can be swapped in by changing cfg.model.hf_model_id.
"""

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import ViTConfig, ViTForImageClassification

logger = logging.getLogger(__name__)


class MedicalViT(nn.Module):
    """Thin wrapper around HuggingFace ViT that exposes forward(images) → logits."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        num_labels = cfg.dataset.num_classes
        logger.info(
            "Loading ViT backbone: %s (pretrained=%s, num_labels=%d)",
            cfg.model.hf_model_id,
            cfg.model.pretrained,
            num_labels,
        )
        if cfg.model.pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                cfg.model.hf_model_id,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            # Offline / fast-test mode: random weights, no download needed
            self.vit = ViTForImageClassification(_vit_b16_config(num_labels))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=images).logits

    def _lora_active(self) -> bool:
        try:
            from peft import PeftModel
            return isinstance(self.vit, PeftModel)
        except ImportError:
            return False

    def get_parameters(self) -> list[np.ndarray]:
        """Return model state as flat list of numpy arrays for FL parameter exchange.

        When LoRA is active, returns only trainable adapter parameters (~0.3M).
        Full fine-tuning returns the entire state_dict (85M params + buffers).
        """
        if self._lora_active():
            return [p.cpu().detach().numpy() for p in self.parameters() if p.requires_grad]
        return [val.cpu().numpy() for val in self.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load aggregated parameters from the FL server.

        When LoRA is active, updates only trainable adapter parameters in-place,
        leaving frozen backbone weights untouched.

        Full fine-tuning preserves each tensor's original dtype and device to avoid:
        - int64 buffers (position_ids, num_batches_tracked) being cast to float32
        - GPU-placed parameters being silently moved to CPU after aggregation
        """
        if self._lora_active():
            trainable = [p for p in self.parameters() if p.requires_grad]
            with torch.no_grad():
                for param, new_arr in zip(trainable, parameters):
                    param.copy_(torch.as_tensor(new_arr, dtype=param.dtype, device=param.device))
            return
        current_state = self.state_dict()
        new_state = OrderedDict()
        for (key, current_tensor), new_array in zip(current_state.items(), parameters):
            new_state[key] = torch.as_tensor(
                new_array,
                dtype=current_tensor.dtype,
                device=current_tensor.device,
            )
        self.load_state_dict(new_state, strict=True)


def build_model(cfg: DictConfig) -> "MedicalViT":
    """Factory: builds backbone and optionally injects LoRA adapters."""
    model = MedicalViT(cfg)

    if cfg.model.use_lora:
        # Month 3+: inject LoRA adapters into attention query/value projections
        from peft import LoraConfig, get_peft_model

        # No task_type — creates a bare PeftModel with plain suffix matching,
        # which is correct for ViT (not natively supported by PEFT task wrappers).
        lora_cfg = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=list(cfg.model.lora_target_modules),
            bias="none",
        )
        model.vit = get_peft_model(model.vit, lora_cfg)
        trainable, total = _count_parameters(model)
        logger.info(
            "LoRA injected: trainable %d / %d params (%.2f%%)",
            trainable,
            total,
            100 * trainable / total,
        )
    else:
        trainable, total = _count_parameters(model)
        logger.info("Full fine-tuning: %d trainable parameters", trainable)

    return model


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _vit_b16_config(num_labels: int) -> ViTConfig:
    """Minimal ViT-B/16 config for offline / no-download testing."""
    return ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=num_labels,
    )
