"""Month 1 entry point: FedAvg baseline with optional DP-SGD.

Run FedAvg (no DP):
    uv run python pipeline/train.py

Run FedAvg + DP-SGD:
    uv run python pipeline/train.py privacy=dp_sgd

Sweep non-IID intensity:
    uv run python pipeline/train.py dataset.alpha=0.1
    uv run python pipeline/train.py dataset.alpha=0.5
    uv run python pipeline/train.py dataset.alpha=1.0

Multi-dataset sweep (requires hydra-launcher-joblib):
    uv run python pipeline/train.py --multirun dataset=pathmnist,chestmnist,dermamnist privacy=none,dp_sgd

All outputs land in outputs/<experiment.name>/<timestamp>/ (configured by Hydra).
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

# Add project root to path so `src` is importable without pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_module.dataset import MedMNISTFederated
from src.fl_module.server import global_evaluate
from src.fl_module.simulation import run_fl_simulation
from src.model_module.backbone import build_model
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)
console = Console()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.experiment.seed)
    device = _get_device()

    console.rule("[bold green]DP-FedLoRA Experiment — Month 1 Baseline")
    console.print(OmegaConf.to_yaml(cfg))
    logger.info("Device: %s", device)
    logger.info("Privacy enabled: %s", cfg.privacy.enabled)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading %s (alpha=%.2f)", cfg.dataset.name, cfg.dataset.alpha)
    fed_dataset = MedMNISTFederated(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building model: %s (lora=%s)", cfg.model.name, cfg.model.use_lora)
    global_model = build_model(cfg)
    global_model.to(device)

    # ── Simulate FL ───────────────────────────────────────────────────────────
    history = run_fl_simulation(
        cfg=cfg,
        fed_dataset=fed_dataset,
        global_model=global_model,
        device=device,
    )

    # ── Final test evaluation ─────────────────────────────────────────────────
    test_loader = fed_dataset.get_test_loader()
    multilabel = cfg.dataset.get("multilabel", False)
    max_batches = int(cfg.training.get("fast_dev_run", 0))
    test_metrics = global_evaluate(global_model, test_loader, device, multilabel, max_batches=max_batches)

    console.rule("[bold green]Final Results")
    console.print(f"  Test accuracy : {test_metrics.get('accuracy', 'N/A'):.4f}")
    console.print(f"  Test AUC      : {test_metrics.get('auc', 0.0):.4f}")
    console.print(f"  Test loss     : {test_metrics.get('test_loss', 0.0):.4f}")
    if cfg.privacy.enabled:
        ps = history.privacy_budget.summary()
        console.print(f"  eps (naive sum): {ps.get('epsilon_naive', 'N/A'):.3f}")
        console.print(f"  delta         : {ps.get('delta', cfg.privacy.target_delta):.0e}")
        console.print(f"  Rounds        : {ps.get('num_rounds', 0):.0f}")

    # ── Save results ──────────────────────────────────────────────────────────
    from hydra.utils import get_original_cwd
    output_dir = Path(get_original_cwd()) / "outputs" / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "test_metrics": test_metrics,
        "history": history.to_dict(),
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path.resolve())

    # ── Save model checkpoint ─────────────────────────────────────────────────
    ckpt_path = output_dir / "global_model.pt"
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved to %s", ckpt_path.resolve())
    logger.info("Best val accuracy across rounds: %.4f", history.best_accuracy())


if __name__ == "__main__":
    main()
