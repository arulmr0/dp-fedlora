import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Accumulates per-round epsilon values for FL privacy accounting.

    In FL+DP, each round contributes an epsilon cost. Advanced composition
    (RDP) gives tighter bounds than naive summation; this class tracks both
    so the paper can report whichever the venue expects.
    """

    target_delta: float = 1e-5
    per_round_epsilons: list[float] = field(default_factory=list)

    def update(self, epsilon: float) -> None:
        self.per_round_epsilons.append(epsilon)

    @property
    def naive_total(self) -> float:
        """Upper bound: linear composition ε_total ≤ Σ ε_r.

        TODO (Month 3): Replace with RDP composition across FL rounds.
        The per-round accountant inside Opacus tracks composition within a round;
        composition across rounds requires passing the RDP orders and moments
        accumulated over all T rounds into the Rényi accountant, then converting
        to (ε, δ)-DP via rdp_to_dp. Reporting the naive sum is conservative and
        must be flagged as such in the paper.
        """
        return sum(self.per_round_epsilons)

    @property
    def num_rounds(self) -> int:
        return len(self.per_round_epsilons)

    def summary(self) -> dict[str, float]:
        if not self.per_round_epsilons:
            return {}
        return {
            "epsilon_naive": self.naive_total,
            "epsilon_last_round": self.per_round_epsilons[-1],
            "delta": self.target_delta,
            "num_rounds": float(self.num_rounds),
        }


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Top-1 accuracy for single-label classification."""
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def compute_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
    multilabel: bool = False,
) -> float:
    """ROC-AUC for single-label (macro-OvR) or multi-label tasks.

    Returns 0.0 when a class has no positive samples in the batch,
    which can happen under non-IID partitioning with small clients.
    """
    probs = torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)
    y_score = probs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    try:
        if multilabel:
            return float(roc_auc_score(y_true, y_score, average="macro"))
        else:
            # Pass explicit labels so roc_auc_score doesn't fail when only a subset of
            # classes appears in y_true (common under non-IID partitioning with small alpha)
            all_labels = np.arange(y_score.shape[1])
            return float(
                roc_auc_score(
                    y_true, y_score, multi_class="ovr", average="macro", labels=all_labels
                )
            )
    except ValueError:
        logger.warning("AUC computation failed (likely missing class in batch); returning 0.0")
        return 0.0


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    multilabel: bool = False,
) -> dict[str, float]:
    """Compute all evaluation metrics for one evaluation pass."""
    metrics: dict[str, float] = {}
    if not multilabel:
        metrics["accuracy"] = compute_accuracy(logits, labels)
    metrics["auc"] = compute_auc(logits, labels, multilabel=multilabel)
    return metrics
