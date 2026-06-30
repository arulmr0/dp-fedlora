"""Utilities for LoRA adapter analysis and DP-FedLoRA sensitivity reporting.

Used for paper Table 1 (communication cost) and §3.1 (sensitivity analysis).
"""

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def adapter_parameter_stats(model: nn.Module) -> dict[str, int | float]:
    """Return trainable / frozen / total parameter counts and communication reduction."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
        "reduction_factor": total / max(trainable, 1),
    }


def log_adapter_stats(model: nn.Module) -> None:
    """Log parameter breakdown for paper reporting (call after build_model)."""
    s = adapter_parameter_stats(model)
    logger.info(
        "Adapter stats | trainable=%d (%.4f%%) | frozen=%d | total=%d | "
        "comm_reduction=%.0fx | comm_bytes=%.1f MB",
        s["trainable"],
        100 * s["trainable"] / max(s["total"], 1),
        s["frozen"],
        s["total"],
        s["reduction_factor"],
        s["trainable"] * 4 / 1e6,  # float32 bytes → MB
    )


def communication_cost_mb(model: nn.Module) -> float:
    """MB transmitted per FL round (float32 for trainable parameters only)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable * 4 / 1e6


def adapter_l2_sensitivity(max_grad_norm: float) -> float:
    """Per-sample L2 sensitivity for adapter-only DP noise.

    In DP-FedLoRA, DP noise is applied only to the adapter (LoRA) gradient
    updates. The per-sample sensitivity equals max_grad_norm because each
    sample's gradient is clipped to this norm before noise addition.

    NOTE (Month 3 → Month 4): This gives the per-step sensitivity.
    The formal (ε, δ)-DP guarantee across T steps and R rounds requires
    composing via the RDP accountant:
        ε(R·T steps) = RDP_compose(σ, q, R·T) converted to (ε, δ)-DP
    where σ = noise_multiplier, q = batch_size / dataset_size (Poisson sampling rate).
    Opacus handles this automatically; this function documents the sensitivity bound.
    See proposal-card-a.md §3.1 for the derivation.
    """
    return float(max_grad_norm)
