"""Sequential FL simulator — no Ray required, Windows-compatible.

Runs the FedAvg round loop directly (no Flower dependency for Month 1).
For large-scale distributed experiments (>100 clients), replace with:

    import flwr as fl
    fl.simulation.start_simulation(
        client_fn=..., num_clients=N,
        config=fl.server.ServerConfig(num_rounds=R),
        strategy=fl.server.strategy.FedAvg(...),
    )

Memory note: a single global_model is shared across evaluation; one model per
client is created for training so gradients don't interfere. For >20 clients on
ViT-B (~344 MB each), reduce num_clients or use gradient checkpointing.
"""

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
from omegaconf import DictConfig
from rich.console import Console

from src.fl_module.client import MedMNISTClient, build_client_fn
from src.fl_module.server import fedavg_aggregate, global_evaluate
from src.model_module.backbone import MedicalViT, build_model
from src.utils.metrics import PrivacyBudget

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class RoundResult:
    round_num: int
    train_loss: float
    val_accuracy: float
    val_auc: float
    epsilon: Optional[float]
    round_seconds: float

    @property
    def epsilon_str(self) -> str:
        return f"{self.epsilon:.3f}" if self.epsilon is not None else "N/A"


@dataclass
class SimulationHistory:
    rounds: list[RoundResult] = field(default_factory=list)
    privacy_budget: PrivacyBudget = field(default_factory=PrivacyBudget)

    def add_round(self, result: RoundResult) -> None:
        self.rounds.append(result)
        if result.epsilon is not None:
            self.privacy_budget.update(result.epsilon)

    def best_accuracy(self) -> float:
        if not self.rounds:
            return 0.0
        return max(r.val_accuracy for r in self.rounds)

    def to_dict(self) -> dict:
        return {
            "rounds": [
                {
                    "round": r.round_num,
                    "train_loss": r.train_loss,
                    "val_accuracy": r.val_accuracy,
                    "val_auc": r.val_auc,
                    "epsilon": r.epsilon,
                    "seconds": r.round_seconds,
                }
                for r in self.rounds
            ],
            "privacy_summary": self.privacy_budget.summary(),
        }


def _find_resume_point(output_dir: Path) -> int:
    """Return the highest completed round number found in output_dir, or 0."""
    ckpts = sorted(
        output_dir.glob("model_checkpoint_round*.pt"),
        key=lambda p: int(p.stem.split("round")[-1]),
    )
    if not ckpts:
        return 0
    return int(ckpts[-1].stem.split("round")[-1])


def run_fl_simulation(
    cfg: DictConfig,
    fed_dataset,
    global_model: MedicalViT,
    device: torch.device,
) -> SimulationHistory:
    """Run FedAvg rounds and return the full history.

    Each round:
      1. Broadcast global parameters to all clients.
      2. Each client trains locally (with optional DP).
      3. FedAvg aggregates updates into new global parameters.
      4. Global model is evaluated on val_loader[0] for per-round logging.

    Checkpointing: after each round, history_checkpoint.json and
    model_checkpoint_round{N}.pt are written. On restart the simulation
    resumes from the last completed round automatically.
    """
    num_clients = cfg.dataset.num_clients
    num_rounds = cfg.training.num_rounds
    multilabel = cfg.dataset.get("multilabel", False)

    # ── Output dir (resolved before client setup so resume detection runs first) ─
    try:
        from hydra.utils import get_original_cwd
        _cwd = get_original_cwd()
    except Exception:
        import os
        _cwd = os.getcwd()
    _output_dir = Path(_cwd) / "outputs" / cfg.experiment.name
    _output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume detection ──────────────────────────────────────────────────────
    resume_from = _find_resume_point(_output_dir)
    if resume_from > 0:
        ckpt = torch.load(
            _output_dir / f"model_checkpoint_round{resume_from}.pt",
            map_location=device,
        )
        global_model.set_parameters(ckpt["params"])
        console.print(
            f"  [bold yellow]Checkpoint found — resuming after round "
            f"{resume_from}/{num_rounds}[/bold yellow]"
        )

    # ── Client setup (after resume so deepcopy picks up correct weights) ──────
    # Memory: ViT-B ≈ 344 MB × num_clients. Reduce num_clients for low-RAM machines.
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    client_fn: Callable[[str], MedMNISTClient] = build_client_fn(
        models=client_models,
        fed_dataset=fed_dataset,
        cfg=cfg,
        device=device,
    )

    # ── History (restored from JSON if resuming) ──────────────────────────────
    history = SimulationHistory(
        privacy_budget=PrivacyBudget(
            target_delta=cfg.privacy.target_delta if cfg.privacy.enabled else 1e-5
        )
    )
    if resume_from > 0:
        saved = json.loads((_output_dir / "history_checkpoint.json").read_text())
        for r in saved["history"]["rounds"]:
            if r["round"] <= resume_from:
                history.add_round(RoundResult(
                    round_num=r["round"],
                    train_loss=r["train_loss"],
                    val_accuracy=r["val_accuracy"],
                    val_auc=r["val_auc"],
                    epsilon=r["epsilon"],
                    round_seconds=r["seconds"],
                ))

    current_params = global_model.get_parameters()

    resume_tag = (
        f"  [yellow](resuming from round {resume_from})[/yellow]"
        if resume_from > 0 else ""
    )
    console.rule(f"[bold cyan]FL Simulation — {cfg.experiment.name}")
    console.print(
        f"  Clients={num_clients}  Rounds={num_rounds}  "
        f"DP={cfg.privacy.enabled}  alpha={cfg.dataset.alpha}{resume_tag}"
    )

    val_loader = fed_dataset.get_val_loader(0)

    for round_num in range(resume_from + 1, num_rounds + 1):
        t0 = time.perf_counter()
        client_results = []

        for cid in range(num_clients):
            client = client_fn(str(cid))
            params, n_examples, metrics = client.fit(current_params, config={})
            client_results.append((params, n_examples, metrics))

        current_params, agg_metrics = fedavg_aggregate(client_results)

        # Update global model with aggregated parameters for evaluation
        global_model.set_parameters(current_params)
        max_batches = int(cfg.training.get("fast_dev_run", 0))
        val_metrics = global_evaluate(global_model, val_loader, device, multilabel, max_batches=max_batches)

        elapsed = time.perf_counter() - t0
        result = RoundResult(
            round_num=round_num,
            train_loss=agg_metrics.get("train_loss", float("nan")),
            val_accuracy=val_metrics.get("accuracy", 0.0),
            val_auc=val_metrics.get("auc", 0.0),
            epsilon=agg_metrics.get("epsilon"),
            round_seconds=elapsed,
        )
        history.add_round(result)

        console.print(
            f"  Round {round_num:3d}/{num_rounds}  "
            f"loss={result.train_loss:.4f}  "
            f"acc={result.val_accuracy:.4f}  "
            f"auc={result.val_auc:.4f}  "
            f"eps={result.epsilon_str}  "
            f"({elapsed:.1f}s)"
        )
        logger.info(
            "Round %d/%d | loss=%.4f | acc=%.4f | auc=%.4f | eps=%s | %.1fs",
            round_num, num_rounds, result.train_loss, result.val_accuracy,
            result.val_auc, result.epsilon_str, elapsed,
        )

        # Write history then model checkpoint — both done before next round starts
        _ckpt_json = {"rounds_completed": round_num, "history": history.to_dict()}
        (_output_dir / "history_checkpoint.json").write_text(json.dumps(_ckpt_json, indent=2))
        torch.save(
            {"params": current_params, "round": round_num},
            _output_dir / f"model_checkpoint_round{round_num}.pt",
        )

    return history
