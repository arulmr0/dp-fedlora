# Task Plan: Federated Learning in Healthcare — Research Initiation

## Goal
Develop a literature-grounded research programme in federated learning for healthcare, identify the 2–3 most tractable open problems given the PI's AI/CV/NLP background, and produce a concrete execution plan toward a venue-ready contribution.

## Phases
- [x] Phase 1: Literature sweep — surveys, open challenges, privacy, robustness (May 2026)
- [x] Phase 2: Gap analysis + research question cards
- [x] Phase 3: Select one primary direction, write 2-page research proposal (Card A → proposal-card-a.md)
- [ ] Phase 4: Baseline reproduction + preliminary experiment (scaffold at C:\Users\user\dp-fedlora\)
  - [x] Month 1 code scaffold: fl_module, data_module, model_module, pipeline/train.py, Hydra configs
  - [x] Smoke test 1: random weights, 2 clients, 2 rounds — pipeline confirmed working (exit 0)
  - [x] Smoke test 2: pretrained ViT-B/16, no DP, 2 clients, 3 rounds — AUC 0.982 test, loss 1.64→0.95, strong convergence
  - [x] Smoke test 3: pretrained ViT-B/16, DP-SGD (σ=1.1, C=1.0, RDP), α=0.5, 3 rounds — ε=0.484/round, ε_naive=1.453 (δ=1e-5), test AUC 0.501
  - [x] Smoke test 4: pretrained ViT-B/16, DP-SGD, α=0.1, 3 rounds — ε=0.483/round, ε_naive=1.450, test AUC 0.502
  - [ ] Full baseline (GPU required): FedAvg (no DP), 20 rounds, 10 clients, PathMNIST
  - [ ] Full baseline (GPU required): FedAvg + DP-SGD, privacy=dp_sgd, same config
  - [ ] Non-IID sweep: alpha ∈ {0.1, 0.5, 1.0} for both above
  - [ ] Month 3: LoRA injection (use_lora=true), DP-FedLoRA implementation
- [ ] Phase 5: Manuscript draft

## Key Questions
1. Which sub-problem is underexplored enough to produce a novel contribution?
2. What is the minimum viable experiment to validate the core hypothesis?
3. Which venue is the right target (NeurIPS / ICLR / KDD / AAAI / Nature Digital Medicine)?

## Decisions Made
- Background (AI/CV/NLP) maps most naturally to: (a) federated vision-language models for medical imaging, (b) federated NLP for clinical notes/EHR.
- Strongest gap as of May 2026: formal privacy guarantees for federated fine-tuning of foundation models on healthcare data, especially under non-IID clinical distributions.
- Second strongest gap: ontology/LLM-based semantic harmonization inside a privacy-preserving FL pipeline.

## Status
**Currently in Phase 4, Month 1** — All 4 CPU smoke tests passed (2026-06-30). Full baselines blocked on GPU access.

## Smoke Test Summary (2026-06-30, CPU, fast_dev_run=2, 2 clients, 3 rounds)

| Run | DP | α | Test AUC | ε naive (3 rounds) | Key finding |
|---|---|---|---|---|---|
| Random weights | No | 0.5 | 0.791 | — | Pipeline works end-to-end |
| Pretrained ViT-B/16 | No | 0.5 | 0.982 | — | Pre-training gives massive AUC lift |
| Pretrained + DP-SGD | Yes | 0.5 | 0.501 | 1.453 | Full-model DP collapses AUC by 0.48 |
| Pretrained + DP-SGD | Yes | 0.1 | 0.502 | 1.450 | α has no effect at 2 clients (expected) |

**Critical finding:** Non-DP test AUC 0.982 vs DP test AUC 0.501 — full-model DP-SGD on 85M parameters destroys the gradient signal at noise_multiplier=1.1. This is the empirical core of the DP-FedLoRA motivation.

**ε accounting confirmed correct:** per-round ε=0.484 constant across rounds (same noise config each round); naive 3-round sum = 1.453. At full training scale (843 steps/client vs 6 in smoke test), per-round ε will be ~5.7, giving ε_naive ≈ 114 over 20 rounds — well above the target ε≤8. Adapter-only DP is needed.

**Known warning (non-blocking):** Opacus `Full backward hook` warning on `client.py:62` — caused by ViT patch embedding Conv2d receiving non-differentiable image inputs. Harmless; expected to disappear when backbone is frozen for LoRA.

## GPU Baseline Commands (copy-paste ready)
```bash
cd dp-fedlora

# FedAvg, no DP (full 20 rounds, 10 clients, PathMNIST)
uv run python pipeline/train.py experiment.name=fedavg_baseline

# FedAvg + DP-SGD
uv run python pipeline/train.py privacy=dp_sgd experiment.name=fedavg_dp_baseline

# Non-IID sweep (FedAvg, no DP)
uv run python pipeline/train.py dataset.alpha=0.1 experiment.name=fedavg_alpha01
uv run python pipeline/train.py dataset.alpha=0.5 experiment.name=fedavg_alpha05
uv run python pipeline/train.py dataset.alpha=1.0 experiment.name=fedavg_alpha10

# Non-IID sweep (FedAvg + DP-SGD)
uv run python pipeline/train.py privacy=dp_sgd dataset.alpha=0.1 experiment.name=fedavg_dp_alpha01
uv run python pipeline/train.py privacy=dp_sgd dataset.alpha=0.5 experiment.name=fedavg_dp_alpha05
uv run python pipeline/train.py privacy=dp_sgd dataset.alpha=1.0 experiment.name=fedavg_dp_alpha10
```
Results land in `dp-fedlora/outputs/<experiment.name>/results.json`.
