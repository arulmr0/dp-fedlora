# DP-FedLoRA

Differentially Private Federated Fine-Tuning of Vision-Language Models for Multi-Site Medical Imaging.

## Setup

```powershell
uv sync
```

## Run

```powershell
# FedAvg baseline (no DP)
uv run python pipeline/train.py

# FedAvg + DP-SGD
uv run python pipeline/train.py privacy=dp_sgd

# Non-IID sweep
uv run python pipeline/train.py dataset.alpha=0.1
```

## Project layout

```
conf/        Hydra configs (dataset, model, fl strategy, privacy)
src/         Python package (data_module, model_module, fl_module, utils)
pipeline/    Experiment entry points
outputs/     Auto-created by Hydra — run results + checkpoints
```
