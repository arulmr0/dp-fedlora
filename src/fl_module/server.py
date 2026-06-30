"""FedAvg aggregation and global evaluation.

Epsilon accounting: for DP-FL, the per-round privacy cost is the MAX over
participating clients (worst-case guarantee for any individual whose data
appears in at most one client). Averaging epsilon across clients is incorrect.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.model_module.backbone import MedicalViT
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


def fedavg_aggregate(
    results: list[tuple[list[np.ndarray], int, dict]],
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Weighted FedAvg aggregate: weighted mean of parameters, max for epsilon.

    Args:
        results: [(parameters, num_examples, metrics), ...] from each client.

    Returns:
        (aggregated_parameters, aggregated_metrics)
    """
    total_examples = sum(n for _, n, _ in results)
    if total_examples == 0:
        raise ValueError("All clients returned 0 examples — check data loaders.")

    # Template tensors preserve original dtype per layer (fixes int64 buffers)
    template = results[0][0]
    aggregated = [np.zeros_like(p, dtype=np.float64) for p in template]

    for params, num_examples, _ in results:
        weight = num_examples / total_examples
        for i, p in enumerate(params):
            aggregated[i] += weight * p.astype(np.float64)

    # Cast each layer back to its original dtype
    aggregated_params = [
        agg.astype(template[i].dtype) for i, agg in enumerate(aggregated)
    ]

    # Aggregate scalar metrics
    agg_metrics: dict[str, float] = {}
    metric_keys = set().union(*[m.keys() for _, _, m in results])
    for key in metric_keys:
        vals_weights = [(m[key], n) for _, n, m in results if key in m]
        if not vals_weights:
            continue
        if key == "epsilon":
            # DP guarantee: worst-case (max) over clients, not average
            agg_metrics["epsilon"] = float(max(v for v, _ in vals_weights))
        else:
            total_w = sum(w for _, w in vals_weights)
            agg_metrics[key] = float(
                sum(v * w for v, w in vals_weights) / total_w
            ) if total_w > 0 else 0.0

    return aggregated_params, agg_metrics


def global_evaluate(
    model: MedicalViT,
    test_loader: DataLoader,
    device: torch.device,
    multilabel: bool = False,
    max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate the global model on the held-out test / val set."""
    model.eval()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            images = images.to(device)
            if multilabel:
                labels_device = labels.float().to(device)
            else:
                labels_device = labels.squeeze(-1).long().to(device)
            logits = model(images)
            total_loss += criterion(logits, labels_device).item() * images.size(0)
            total_examples += images.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)
    metrics = compute_metrics(logits_cat, labels_cat, multilabel=multilabel)
    metrics["test_loss"] = total_loss / max(total_examples, 1)
    return metrics
