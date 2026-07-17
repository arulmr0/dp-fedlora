# Task Plan: Experiment Campaign 2 — Ablation + Second Dataset + Controlled Comparison

## Goal
Produce 5 new GPU experiments that close the three reviewer-facing gaps:
1. **Controlled comparison** (P0) — DP-FedLoRA vs full-model DP at identical batch=8 → same ε, fair utility comparison
2. **Rank ablation** (P1a/b/c) — r ∈ {4, 8, 32} at σ=1.1, PathMNIST (r=16 already in Exp 3)
3. **Second dataset** (P2/P3) — ChestMNIST no-DP baseline + DP-FedLoRA

## Privacy Budget Preview

| Run | σ | batch | T | naive ε (10R) | GPU |
|-----|---|-------|---|---------------|-----|
| P0: DP-FedLoRA batch=8 | 1.1 | 8 | 10 | ~4.77 | Kaggle T4 |
| Exp2 full-model (done) | 1.1 | 8 | 10 | 5.17 | local RTX 4060 |
| P1a: r=4 | 1.1 | 32 | 10 | ~6.65 | Kaggle T4 |
| P1b: r=8 | 1.1 | 32 | 10 | ~6.65 | Kaggle T4 |
| P1c: r=32 | 1.1 | 32 | 10 | ~6.65 | Kaggle T4 |
| P2: ChestMNIST no-DP | --- | 32 | 10 | --- | local RTX 4060 |
| P3: ChestMNIST DP-FedLoRA | 1.1 | 32 | 10 | ~7.0 | Kaggle T4 |

## Phases

- [x] Phase 0: Analysis — ε values computed, code assessed
- [ ] Phase 1: Code fix — Dirichlet partition for multi-label (dataset.py, ~5 lines)
- [ ] Phase 2: Create Hydra experiment configs for all runs
- [ ] Phase 3a: Kaggle — P0 Controlled (batch=8) — ~5h
- [ ] Phase 3b: Kaggle — P1a r=4 + P1b r=8 — ~10h (2 sessions)
- [ ] Phase 3c: Kaggle — P1c r=32 + P3 ChestMNIST DP — ~10h (2 sessions)
- [ ] Phase 3d: Local — P2 ChestMNIST no-DP — ~4h (background)
- [ ] Phase 4: Paper — add controlled comparison table, rank ablation table, ChestMNIST table

## Decisions
- **ChestMNIST > DermaMNIST**: DermaMNIST has 7k train → q=0.025 → ε=18 (above threshold). ChestMNIST has 78k → q≈0.002 → ε≈7 ✓
- **Multi-label Dirichlet**: ChestMNIST labels are (N,14) binary. Fix: argmax of label vector as primary class for partition; samples with no disease label → class 14 ("normal")
- **Controlled comparison via batch=8**: DP-FedLoRA at batch=8 gives same q as Exp2 → Theorem 1 demonstrated empirically
- **Sigma ablation deferred**: at batch=32, σ=0.67 → ε=24 (above threshold); not clinically relevant at current 5-client scale

## Errors
(none yet)

## Status
**Phase 1** — implementing Dirichlet multi-label fix
