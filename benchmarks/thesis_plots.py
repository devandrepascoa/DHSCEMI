#!/usr/bin/env python3
"""
Thesis-quality plots for the cost-aware autoscaler.

Generates publication-ready figures from benchmark data and the
autoscaler's cost model. Outputs PDF + PNG to benchmarks/thesis_figures/.

Usage:
    uv run python benchmarks/thesis_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Matplotlib global style — clean, thesis-friendly
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (7, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

BENCH_DATA = Path(__file__).parent / "parsed_logs" / "night_logs_6_with_model_info.json"

# Cost model from main_cost_aware.py
HARDWARE_CONFIGS = {
    "cpu_4":   {"hourly_cost": 0.40, "throughput": 12.0,  "label": "CPU 4 cores",  "color": "#4e79a7"},
    "cpu_8":   {"hourly_cost": 0.80, "throughput": 18.0,  "label": "CPU 8 cores",  "color": "#59a14f"},
    "cpu_12":  {"hourly_cost": 1.20, "throughput": 22.0,  "label": "CPU 12 cores", "color": "#f28e2b"},
    "gpu_50":  {"hourly_cost": 1.00, "throughput": 50.0,  "label": "GPU 50%",      "color": "#e15759"},
    "gpu_100": {"hourly_cost": 2.00, "throughput": 100.0, "label": "GPU 100%",     "color": "#b07aa1"},
}


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.pdf / .png")


# ===================================================================
# Figure 1 — Measured throughput by hardware config (from benchmarks)
# ===================================================================
def fig1_throughput_by_hardware() -> None:
    """Bar chart: token generation throughput across CPU/GPU configs,
    grouped by model family, for the smallest quantized model (Q4_K_M)."""
    with open(BENCH_DATA) as f:
        raw = json.load(f)

    # Filter: Q4_K_M quant, concurrent=1, token_size=128 (single-request perf)
    df = pd.DataFrame(raw)
    df = df[(df["model_quant"] == "Q4_K_M") & (df["concurrent_requests"] == 1)]

    # Build a config label
    def hw_label(row):
        if row["variant"] == "cuda":
            return f"GPU {int(row['gpu_percentage'])}%"
        return f"CPU {int(row['cpu_cores'])} cores"

    df["hw"] = df.apply(hw_label, axis=1)

    # Order hardware configs logically
    hw_order = ["CPU 1 cores", "CPU 2 cores", "CPU 4 cores", "CPU 8 cores",
                "GPU 25%", "GPU 50%", "GPU 75%", "GPU 100%"]
    df["hw"] = pd.Categorical(df["hw"], categories=hw_order, ordered=True)
    df = df.sort_values("hw")

    # Pivot: one group per model, bars per hw config
    models = sorted(df["model_name"].unique())
    model_labels = []
    for m in models:
        row = df[df["model_name"] == m].iloc[0]
        size_b = row["model_size"] / 1000
        family = "DS" if "DeepSeek" in m else "Qwen"
        model_labels.append(f"{family}-{size_b:.1f}B")

    fig, ax = plt.subplots(figsize=(10, 5))

    hw_configs = [h for h in hw_order if h in df["hw"].values]
    x = np.arange(len(models))
    width = 0.8 / len(hw_configs)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(hw_configs)))

    for i, hw in enumerate(hw_configs):
        vals = []
        errs = []
        for m in models:
            subset = df[(df["model_name"] == m) & (df["hw"] == hw)]
            if len(subset) > 0:
                vals.append(subset["token_generation_throughput_mean"].values[0])
                errs.append(subset["token_generation_throughput_stddev"].values[0])
            else:
                vals.append(0)
                errs.append(0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               yerr=errs, label=hw, color=colors[i], capsize=2, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Token Generation Throughput (tok/s)")
    ax.set_title("Token Generation Throughput by Hardware Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right")
    ax.legend(title="Hardware", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig1_throughput_by_hardware")


# ===================================================================
# Figure 2 — Cost-per-token vs demand (the autoscaler's core insight)
# ===================================================================
def fig2_cost_per_token_vs_demand() -> None:
    """Line plot showing cost-per-token for each config as demand varies.
    Highlights the crossover points where switching configs saves money."""
    demand = np.linspace(0.1, 120, 500)

    fig, ax = plt.subplots(figsize=(8, 5))

    for cid, cfg in HARDWARE_CONFIGS.items():
        throughput = cfg["throughput"]
        hourly = cfg["hourly_cost"]
        # cost_per_token = hourly / (throughput * 3600) when throughput >= demand
        # When demand > throughput, the config can't keep up (infinite queue)
        cpt = np.where(demand <= throughput,
                       hourly / (throughput * 3600),
                       np.nan)
        ax.plot(demand, cpt * 1e6, label=cfg["label"], color=cfg["color"], linewidth=2)
        # Mark the throughput limit with a vertical tick
        ax.axvline(throughput, color=cfg["color"], linestyle=":", alpha=0.4, linewidth=1)

    ax.set_xlabel("Demand (tokens/s)")
    ax.set_ylabel("Cost per Token (×10⁻⁶ $)")
    ax.set_title("Cost per Token vs. Demand by Hardware Configuration")
    ax.legend()
    ax.set_xlim(0, 120)
    fig.tight_layout()
    _save(fig, "fig2_cost_per_token_vs_demand")


# ===================================================================
# Figure 3 — Optimal hardware selection regions
# ===================================================================
def fig3_optimal_selection_regions() -> None:
    """Stacked area / step chart showing which config the autoscaler
    selects at each demand level."""
    demand = np.linspace(0, 110, 1000)
    configs = list(HARDWARE_CONFIGS.items())

    selected_ids = []
    selected_costs = []
    for d in demand:
        viable = [(cid, c) for cid, c in configs if c["throughput"] >= d]
        if not viable:
            best_cid = max(configs, key=lambda x: x[1]["throughput"])[0]
        else:
            best_cid = min(viable, key=lambda x: x[1]["hourly_cost"] / (x[1]["throughput"] * 3600))[0]
        selected_ids.append(best_cid)
        cfg = HARDWARE_CONFIGS[best_cid]
        selected_costs.append(cfg["hourly_cost"])

    # Map config ids to numeric indices for coloring
    unique_configs = list(dict.fromkeys(selected_ids))  # preserve order of first appearance
    config_to_idx = {c: i for i, c in enumerate(unique_configs)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: colored regions showing selected config
    prev_cid = selected_ids[0]
    start_d = demand[0]
    for i in range(1, len(demand)):
        if selected_ids[i] != prev_cid or i == len(demand) - 1:
            end_d = demand[i]
            cfg = HARDWARE_CONFIGS[prev_cid]
            ax1.axvspan(start_d, end_d, alpha=0.35, color=cfg["color"], label=cfg["label"])
            # Label in the middle
            mid = (start_d + end_d) / 2
            ax1.text(mid, 0.5, cfg["label"], ha="center", va="center",
                     fontsize=9, fontweight="bold", transform=ax1.get_xaxis_transform())
            start_d = end_d
            prev_cid = selected_ids[i]

    # Remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax1.set_ylabel("Selected Configuration")
    ax1.set_yticks([])
    ax1.set_title("Autoscaler Optimal Hardware Selection by Demand Level")

    # Bottom: hourly cost of selected config
    ax2.fill_between(demand, selected_costs, alpha=0.3, color="#4e79a7")
    ax2.plot(demand, selected_costs, color="#4e79a7", linewidth=1.5)
    ax2.set_xlabel("Demand (tokens/s)")
    ax2.set_ylabel("Hourly Cost ($)")
    ax2.set_title("Hourly Cost of Selected Configuration")

    fig.tight_layout()
    _save(fig, "fig3_optimal_selection_regions")


# ===================================================================
# Figure 4 — Cost savings: dynamic vs static allocation
# ===================================================================
def fig4_cost_savings() -> None:
    """Compare total hourly cost of dynamic (autoscaler) vs static
    (always GPU 100%) allocation across a range of demand levels."""
    demand = np.linspace(0.5, 100, 200)
    configs = list(HARDWARE_CONFIGS.items())

    dynamic_cost = []
    static_gpu100_cost = []
    static_cpu8_cost = []

    for d in demand:
        # Dynamic: autoscaler picks optimal
        viable = [(cid, c) for cid, c in configs if c["throughput"] >= d]
        if not viable:
            best = max(configs, key=lambda x: x[1]["throughput"])[1]
        else:
            best = min(viable, key=lambda x: x[1]["hourly_cost"] / (x[1]["throughput"] * 3600))[1]
        dynamic_cost.append(best["hourly_cost"])

        # Static GPU 100%: always on
        static_gpu100_cost.append(HARDWARE_CONFIGS["gpu_100"]["hourly_cost"])

        # Static CPU 8: always on (can't handle >18 tok/s)
        static_cpu8_cost.append(HARDWARE_CONFIGS["cpu_8"]["hourly_cost"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: hourly cost comparison
    ax1.plot(demand, dynamic_cost, label="Dynamic (autoscaler)", color="#4e79a7", linewidth=2)
    ax1.plot(demand, static_gpu100_cost, label="Static GPU 100%", color="#e15759",
             linewidth=2, linestyle="--")
    ax1.plot(demand, static_cpu8_cost, label="Static CPU 8 cores", color="#f28e2b",
             linewidth=2, linestyle="--")
    ax1.set_xlabel("Demand (tokens/s)")
    ax1.set_ylabel("Hourly Cost ($)")
    ax1.set_title("Hourly Cost: Dynamic vs. Static Allocation")
    ax1.legend()

    # Right: savings percentage vs GPU 100%
    savings_pct = [(1 - dc / sg) * 100 for dc, sg in zip(dynamic_cost, static_gpu100_cost)]
    ax2.fill_between(demand, savings_pct, alpha=0.3, color="#59a14f")
    ax2.plot(demand, savings_pct, color="#59a14f", linewidth=2)
    ax2.set_xlabel("Demand (tokens/s)")
    ax2.set_ylabel("Cost Savings (%)")
    ax2.set_title("Cost Savings vs. Static GPU 100% Allocation")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax2.set_ylim(-5, 100)

    fig.tight_layout()
    _save(fig, "fig4_cost_savings_dynamic_vs_static")


# ===================================================================
# Figure 5 — Throughput scaling efficiency (from benchmark data)
# ===================================================================
def fig5_throughput_scaling_efficiency() -> None:
    """For the 1.5B Q4_K_M model, show how throughput scales with
    CPU cores and GPU %, plus the diminishing returns."""
    with open(BENCH_DATA) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    # Use the smallest model for clearest scaling picture
    target = df[(df["model_name"] == "01-DeepSeek-R1-Distill-Qwen") &
                (df["model_quant"] == "Q4_K_M") &
                (df["concurrent_requests"] == 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # CPU scaling
    cpu = target[target["variant"] == "cpu"].sort_values("cpu_cores")
    if len(cpu) > 0:
        cores = cpu["cpu_cores"].values
        tput = cpu["token_generation_throughput_mean"].values
        err = cpu["token_generation_throughput_stddev"].values
        ax1.errorbar(cores, tput, yerr=err, marker="o", capsize=4,
                     color="#4e79a7", linewidth=2, markersize=8)
        # Ideal linear scaling line from 1-core baseline
        if len(cores) > 0 and tput[0] > 0:
            ideal = tput[0] * cores / cores[0]
            ax1.plot(cores, ideal, "--", color="gray", alpha=0.5, label="Ideal linear scaling")
        ax1.set_xlabel("CPU Cores")
        ax1.set_ylabel("Token Generation Throughput (tok/s)")
        ax1.set_title("CPU Scaling: DeepSeek-R1 1.5B Q4_K_M")
        ax1.set_xticks(cores)
        ax1.legend()

    # GPU scaling
    gpu = target[target["variant"] == "cuda"].sort_values("gpu_percentage")
    if len(gpu) > 0:
        pcts = gpu["gpu_percentage"].values
        tput = gpu["token_generation_throughput_mean"].values
        err = gpu["token_generation_throughput_stddev"].values
        ax2.errorbar(pcts, tput, yerr=err, marker="s", capsize=4,
                     color="#e15759", linewidth=2, markersize=8)
        ax2.set_xlabel("GPU Allocation (%)")
        ax2.set_ylabel("Token Generation Throughput (tok/s)")
        ax2.set_title("GPU Scaling: DeepSeek-R1 1.5B Q4_K_M")
        ax2.set_xticks(pcts)

    fig.tight_layout()
    _save(fig, "fig5_throughput_scaling_efficiency")


# ===================================================================
# Figure 6 — Model size impact on throughput
# ===================================================================
def fig6_model_size_impact() -> None:
    """Show how model size affects throughput for a fixed hardware config."""
    with open(BENCH_DATA) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    # Q4_K_M quant, concurrent=1, CPU 4 cores and GPU 100%
    df = df[(df["model_quant"] == "Q4_K_M") & (df["concurrent_requests"] == 1)]

    fig, ax = plt.subplots(figsize=(8, 5))

    for variant, color, marker, label_prefix in [
        ("cpu", "#4e79a7", "o", "CPU 4 cores"),
        ("cuda", "#e15759", "s", "GPU 100%"),
    ]:
        if variant == "cpu":
            subset = df[(df["variant"] == "cpu") & (df["cpu_cores"] == 4)]
        else:
            subset = df[(df["variant"] == "cuda") & (df["gpu_percentage"] == 100)]

        if len(subset) == 0:
            continue

        subset = subset.sort_values("model_size")
        sizes = subset["model_size"].values / 1000  # to billions
        tput = subset["token_generation_throughput_mean"].values
        err = subset["token_generation_throughput_stddev"].values

        ax.errorbar(sizes, tput, yerr=err, marker=marker, capsize=4,
                    color=color, linewidth=2, markersize=8, label=label_prefix)

    ax.set_xlabel("Model Size (Billion Parameters)")
    ax.set_ylabel("Token Generation Throughput (tok/s)")
    ax.set_title("Impact of Model Size on Throughput (Q4_K_M Quantization)")
    ax.legend()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))

    fig.tight_layout()
    _save(fig, "fig6_model_size_impact")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print(f"Generating thesis figures in {OUT_DIR}/\n")

    print("Figure 1: Throughput by hardware configuration")
    fig1_throughput_by_hardware()

    print("Figure 2: Cost-per-token vs demand")
    fig2_cost_per_token_vs_demand()

    print("Figure 3: Optimal hardware selection regions")
    fig3_optimal_selection_regions()

    print("Figure 4: Cost savings — dynamic vs static")
    fig4_cost_savings()

    print("Figure 5: Throughput scaling efficiency")
    fig5_throughput_scaling_efficiency()

    print("Figure 6: Model size impact on throughput")
    fig6_model_size_impact()

    print(f"\nDone — {len(list(OUT_DIR.glob('*.pdf')))} PDFs, "
          f"{len(list(OUT_DIR.glob('*.png')))} PNGs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
