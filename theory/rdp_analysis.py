"""
RDP Privacy Analysis for DP-FedLoRA vs Full-Model DP-FedAvg.

Run:   uv run python theory/rdp_analysis.py

Outputs the numerical evidence for Theorem 1 (Privacy Equivalence) and
Proposition 1 (Noise Energy Advantage) from theory/bounds.md.
"""

from __future__ import annotations

import sys
import numpy as np

try:
    from opacus.accountants.analysis import rdp as rdp_analysis
    from opacus.accountants import RDPAccountant
except ImportError:
    sys.exit("Opacus not found. Run: uv add opacus")


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "smoke_test (fast_dev_run=2, 2 clients)": {
        "sigma": 1.1,
        "C": 1.0,
        "delta": 1e-5,
        "n_per_client": 45000,  # PathMNIST 90k / 2 clients
        "batch_size": 32,
        "local_epochs": 3,
        "fast_dev_run": 2,      # batches per epoch
        "n_rounds": 3,
    },
    "cpu_baseline (fast_dev_run=2, 3 clients)": {
        "sigma": 1.1,
        "C": 1.0,
        "delta": 1e-5,
        "n_per_client": 30000,  # PathMNIST 90k / 3 clients
        "batch_size": 32,
        "local_epochs": 3,
        "fast_dev_run": 2,
        "n_rounds": 8,
    },
    "full_scale (10 clients, 20 rounds)": {
        "sigma": 1.1,
        "C": 1.0,
        "delta": 1e-5,
        "n_per_client": 9000,   # PathMNIST 90k / 10 clients
        "batch_size": 32,
        "local_epochs": 3,
        "fast_dev_run": 0,      # full data
        "n_rounds": 20,
    },
}

# LoRA-B/16 adapter vs full model parameter counts
LORA_PARAMS = 589_824      # trainable: q_proj + v_proj, r=16, 12 layers
FULL_PARAMS = 85_805_577   # all ViT-B/16 parameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def steps_per_round(cfg: dict) -> int:
    """Number of Opacus DP-SGD steps per local round per client."""
    n = cfg["n_per_client"]
    B = cfg["batch_size"]
    epochs = cfg["local_epochs"]
    fdr = cfg["fast_dev_run"]
    batches_per_epoch = fdr if fdr > 0 else (n // B)
    return batches_per_epoch * epochs


def compute_epsilon(sigma: float, q: float, total_steps: int, delta: float) -> tuple[float, float]:
    """Return (epsilon, best_alpha) using the tightest RDP bound."""
    orders = list(np.arange(2.0, 200.0, 0.5)) + list(np.arange(200.0, 512.0, 2.0))
    rdp_vals = rdp_analysis.compute_rdp(
        q=q, noise_multiplier=sigma, steps=total_steps, orders=orders
    )
    best_eps, best_alpha = float("inf"), None
    for alpha, rdp_eps in zip(orders, rdp_vals):
        if alpha <= 1:
            continue
        # Convert RDP to (eps, delta)-DP via the standard formula
        eps_dp = rdp_eps + (np.log((alpha - 1) / alpha) - np.log(delta) / (alpha - 1)) / 1
        # Simpler: eps = rdp_eps + log(1/delta) / (alpha - 1)
        eps_dp_simple = rdp_eps + np.log(1.0 / delta) / (alpha - 1)
        eps_use = min(eps_dp, eps_dp_simple)
        if eps_use < best_eps:
            best_eps = eps_use
            best_alpha = alpha
    return best_eps, best_alpha


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(cfg: dict, label: str) -> dict:
    sigma = cfg["sigma"]
    delta = cfg["delta"]
    n = cfg["n_per_client"]
    B = cfg["batch_size"]
    T = cfg["n_rounds"]
    q = B / n

    spr = steps_per_round(cfg)
    total_steps = spr * T

    eps, alpha = compute_epsilon(sigma, q, total_steps, delta)

    noise_energy_full = sigma**2 * cfg["C"]**2 * FULL_PARAMS
    noise_energy_lora = sigma**2 * cfg["C"]**2 * LORA_PARAMS

    return {
        "label": label,
        "sigma": sigma,
        "C": cfg["C"],
        "delta": delta,
        "n_per_client": n,
        "q": q,
        "steps_per_round": spr,
        "T": T,
        "total_steps": total_steps,
        "epsilon": eps,
        "best_alpha": alpha,
        "noise_energy_full": noise_energy_full,
        "noise_energy_lora": noise_energy_lora,
        "noise_reduction_factor": noise_energy_full / noise_energy_lora,
        "comm_full_MB": FULL_PARAMS * 4 / 1e6,
        "comm_lora_MB": LORA_PARAMS * 4 / 1e6,
        "comm_reduction": FULL_PARAMS / LORA_PARAMS,
    }


def print_report(r: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Config: {r['label']}")
    print(f"{'='*70}")
    print(f"  sigma={r['sigma']}, C={r['C']}, delta={r['delta']}")
    print(f"  n_per_client={r['n_per_client']:,}, q={r['q']:.5f}")
    print(f"  steps/round={r['steps_per_round']}, T={r['T']}, total_steps={r['total_steps']:,}")
    print()
    print(f"  Privacy (IDENTICAL for full-model and DP-FedLoRA):")
    print(f"    epsilon = {r['epsilon']:.4f}  (alpha* = {r['best_alpha']:.1f})")
    print()
    print(f"  Noise energy E[||noise||^2] = sigma^2 * C^2 * dim:")
    print(f"    Full model (d={FULL_PARAMS:,}): {r['noise_energy_full']:.3e}")
    print(f"    DP-FedLoRA (k={LORA_PARAMS:,}): {r['noise_energy_lora']:.3e}")
    print(f"    Reduction factor: {r['noise_reduction_factor']:.0f}x")
    print()
    print(f"  Communication per round:")
    print(f"    Full model: {r['comm_full_MB']:.1f} MB")
    print(f"    DP-FedLoRA: {r['comm_lora_MB']:.2f} MB")
    print(f"    Reduction: {r['comm_reduction']:.0f}x")


def sigma_for_epsilon(target_eps: float, q: float, total_steps: int, delta: float) -> float:
    """Binary search for the sigma that achieves target_eps."""
    lo, hi = 0.01, 100.0
    for _ in range(64):
        mid = (lo + hi) / 2
        eps, _ = compute_epsilon(mid, q, total_steps, delta)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    print("DP-FedLoRA: RDP Privacy Analysis")
    print("Theorem 1 (Privacy Equivalence) + Proposition 1 (Noise Energy Advantage)")

    results = []
    for label, cfg in CONFIGS.items():
        r = analyze(cfg, label)
        print_report(r)
        results.append(r)

    # Sigma budget: what sigma is needed to hit eps=8 at full scale?
    print(f"\n{'='*70}")
    print("Sigma-epsilon tradeoff (full scale, T=20, 10 clients, 9k/client)")
    print(f"{'='*70}")
    cfg_full = CONFIGS["full_scale (10 clients, 20 rounds)"]
    n = cfg_full["n_per_client"]
    B = cfg_full["batch_size"]
    q = B / n
    spr = steps_per_round(cfg_full)
    T = cfg_full["n_rounds"]
    total = spr * T

    for target_eps in [1.0, 2.0, 4.0, 8.0, 10.0]:
        sig = sigma_for_epsilon(target_eps, q, total, cfg_full["delta"])
        print(f"  eps={target_eps:.1f}: sigma_needed={sig:.4f}  (current sigma={cfg_full['sigma']} -> eps={results[2]['epsilon']:.4f})")

    print()
    print("Key result: sigma=1.1 satisfies eps<8 at full scale with margin.")
    print("DP-FedLoRA injects 145x less noise energy at identical privacy budget.")
