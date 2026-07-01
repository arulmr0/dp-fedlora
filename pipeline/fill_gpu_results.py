"""
fill_gpu_results.py
-------------------
Reads GPU experiment outputs from outputs/<name>/results.json and prints
the LaTeX blocks that replace the \\needsevidence{} markers in paper/dp_fedlora.tex.

Usage (after GPU runs complete):
    uv run python pipeline/fill_gpu_results.py

Expected experiment names (from task_plan.md GPU commands):
    Experiment 3 - Main comparison (α = 0.1, 0.5, 1.0):
        fedavg_baseline       -- FedAvg no-DP, α=0.5
        fedavg_dp_alpha05     -- FedAvg + DP-SGD, α=0.5
        fedavg_dp_alpha01     -- FedAvg + DP-SGD, α=0.1
        fedavg_dp_alpha10     -- FedAvg + DP-SGD, α=1.0
        fedlora_dp_alpha05    -- DP-FedLoRA, α=0.5
        fedlora_dp_alpha01    -- DP-FedLoRA, α=0.1
        fedlora_dp_alpha10    -- DP-FedLoRA, α=1.0

    Experiment 4 - Ablation (rank × σ grid):
        fedlora_nodp_alpha05  -- LoRA no-DP upper bound
        fedlora_r4_dp_alpha05
        fedlora_r8_dp_alpha05
        fedlora_r16_dp_alpha05  (= fedlora_dp_alpha05)
        fedlora_r32_dp_alpha05
        fedlora_sigma067_alpha05
        fedlora_sigma080_alpha05
        fedlora_sigma150_alpha05

Outputs:
    paper/gpu_table3.tex   -- drop-in LaTeX for Experiment 3 table
    paper/gpu_table4.tex   -- drop-in LaTeX for Experiment 4 ablation table
    Stdout summary of which \\needsevidence{} lines to replace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PAPER = ROOT / "paper"


def load(name: str) -> Optional[dict]:
    p = OUTPUTS / name / "results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def test_auc(r: Optional[dict]) -> str:
    if r is None:
        return r"{\color{red}MISSING}"
    return f"{r['test_metrics']['auc']:.3f}"


def final_epsilon(r: Optional[dict]) -> str:
    if r is None:
        return "---"
    rounds = r["history"]["rounds"]
    eps_values = [ro["epsilon"] for ro in rounds if ro.get("epsilon") is not None]
    if not eps_values:
        return "---"
    return f"{eps_values[-1]:.2f}"


def avg_time(r: Optional[dict]) -> str:
    if r is None:
        return "---"
    rounds = r["history"]["rounds"]
    secs = [ro["seconds"] for ro in rounds]
    return f"{sum(secs)/len(secs):.0f}s" if secs else "---"


def comm_mb(use_lora: bool) -> str:
    return r"\textbf{2.4 MB}" if use_lora else "343 MB"


# ---------------------------------------------------------------------------
# Experiment 3: main GPU comparison table
# ---------------------------------------------------------------------------
def build_table3() -> str:
    rows_spec = [
        # (label, exp_name, alpha_str, use_lora)
        ("FedAvg (no DP)",        "fedavg_baseline",    "0.5", False),
        ("FedAvg + DP-SGD",       "fedavg_dp_alpha01",  "0.1", False),
        ("FedAvg + DP-SGD",       "fedavg_dp_alpha05",  "0.5", False),
        ("FedAvg + DP-SGD",       "fedavg_dp_alpha10",  "1.0", False),
        (r"\method{} ($r{=}16$)", "fedlora_dp_alpha01", "0.1", True),
        (r"\method{} ($r{=}16$)", "fedlora_dp_alpha05", "0.5", True),
        (r"\method{} ($r{=}16$)", "fedlora_dp_alpha10", "1.0", True),
    ]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{GPU full-scale comparison (20 rounds, 10 clients, PathMNIST,"
        r" $\noise{=}1.1$, $\clip{=}1.0$, $\delta{=}10^{-5}$).}",
        r"\label{tab:gpu}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & $\alpha$ & Test AUC & $\eps$ & Comm/round \\",
        r"\midrule",
    ]
    for label, name, alpha, use_lora in rows_spec:
        r = load(name)
        auc = test_auc(r)
        eps = final_epsilon(r) if not use_lora or name != "fedavg_baseline" else "---"
        comm = comm_mb(use_lora)
        lines.append(f"{label} & {alpha} & {auc} & {eps} & {comm} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment 4: rank × sigma ablation table
# ---------------------------------------------------------------------------
def build_table4() -> str:
    # (sigma, eps_label, [r4, r8, r16, r32])
    ablation_rows = [
        ("1.5", "1.2", ["fedlora_sigma150_r4_alpha05",  "fedlora_sigma150_r8_alpha05",
                         "fedlora_sigma150_r16_alpha05", "fedlora_sigma150_r32_alpha05"]),
        ("1.1", "2.6", ["fedlora_r4_dp_alpha05",         "fedlora_r8_dp_alpha05",
                         "fedlora_dp_alpha05",             "fedlora_r32_dp_alpha05"]),
        ("0.8", "5.5", ["fedlora_sigma080_r4_alpha05",   "fedlora_sigma080_r8_alpha05",
                         "fedlora_sigma080_r16_alpha05",  "fedlora_sigma080_r32_alpha05"]),
        ("0.67", "8.0", ["fedlora_sigma067_r4_alpha05",  "fedlora_sigma067_r8_alpha05",
                          "fedlora_sigma067_alpha05",      "fedlora_sigma067_r32_alpha05"]),
    ]

    nodp = load("fedlora_nodp_alpha05")
    nodp_auc = test_auc(nodp)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation: test AUC vs.\ noise multiplier $\noise$ and adapter"
        r" rank $r$ ($\alpha{=}0.5$, 20 rounds, 10 clients)."
        r" LoRA no-DP upper bound: " + nodp_auc + r".}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{ccrrrr}",
        r"\toprule",
        r"$\noise$ & $\eps$ & $r{=}4$ & $r{=}8$ & $r{=}16$ & $r{=}32$ \\",
        r"\midrule",
    ]
    for sigma, eps_label, names in ablation_rows:
        cells = [test_auc(load(n)) for n in names]
        lines.append(f"{sigma} & {eps_label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------
def status_summary() -> None:
    required = [
        "fedavg_baseline",
        "fedavg_dp_alpha01", "fedavg_dp_alpha05", "fedavg_dp_alpha10",
        "fedlora_dp_alpha01", "fedlora_dp_alpha05", "fedlora_dp_alpha10",
        "fedlora_nodp_alpha05",
        "fedlora_r4_dp_alpha05", "fedlora_r8_dp_alpha05", "fedlora_r32_dp_alpha05",
        "fedlora_sigma067_alpha05", "fedlora_sigma080_alpha05", "fedlora_sigma150_alpha05",
    ]
    missing = [n for n in required if not (OUTPUTS / n / "results.json").exists()]
    done = [n for n in required if (OUTPUTS / n / "results.json").exists()]

    print("=" * 60)
    print(f"GPU experiment status: {len(done)}/{len(required)} done")
    print("=" * 60)
    if missing:
        print("\nMISSING (run these first):")
        for n in missing:
            print(f"  {n}")
    if done:
        print("\nCOMPLETE:")
        for n in done:
            r = load(n)
            print(f"  {n:40s}  AUC={test_auc(r)}  eps={final_epsilon(r)}")
    print()


# ---------------------------------------------------------------------------
def main() -> None:
    status_summary()

    t3 = build_table3()
    t4 = build_table4()

    out3 = PAPER / "gpu_table3.tex"
    out4 = PAPER / "gpu_table4.tex"
    out3.write_text(t3)
    out4.write_text(t4)

    print(f"Written: {out3}")
    print(f"Written: {out4}")
    print()
    print("Next step: in dp_fedlora.tex, find each \\needsevidence{} block and")
    print("replace it with \\input{gpu_table3.tex} or \\input{gpu_table4.tex}")
    print("as appropriate. Then recompile.")

    missing_count = sum(
        1 for n in ["fedavg_baseline", "fedavg_dp_alpha05", "fedlora_dp_alpha05"]
        if not (OUTPUTS / n / "results.json").exists()
    )
    if missing_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
