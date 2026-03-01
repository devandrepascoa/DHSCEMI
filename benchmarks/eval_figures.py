"""Generate evaluation figures for the thesis from benchmark data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Thesis" / "Images"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(ROOT / "throughput_benchmark_results.json") as f:
    bench = json.load(f)

with open(ROOT / "hardware_configs.json") as f:
    hw = json.load(f)

CONFIGS = {c["config_id"]: c for c in hw["configs"]}
MEASURED = hw["measured_throughput"]

# Tier-order sorted (explicit ordering from hardware_configs.json)
TIER_ORDER = sorted(CONFIGS.keys(), key=lambda c: CONFIGS[c]["tier_order"])

# Single blue gradient for all configs (tier-order: lightest = lowest tier)
_cmap = plt.cm.Blues  # type: ignore[attr-defined]
_gradient_vals = [0.3 + 0.55 * i / 4 for i in range(5)]
COLORS = {c: _cmap(v) for c, v in zip(TIER_ORDER, _gradient_vals)}

LABELS = {
    "cpu_4": "cpu_4 (4 cores)",
    "cpu_16": "cpu_16 (16 cores)",
    "cpu_48": "cpu_48 (48 cores)",
    "gpu_25": "gpu_25 (25% GPU)",
    "gpu_100": "gpu_100 (100% GPU)",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


# ---------------------------------------------------------------------------
# Figure 1: Throughput vs batch size (line plot)
# ---------------------------------------------------------------------------
def fig_throughput_vs_batch():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cfg in ["cpu_4", "cpu_16", "cpu_48", "gpu_25", "gpu_100"]:
        data = bench["results"][cfg]
        batches = sorted(int(k) for k in data if k != "peak_tokens_per_second")
        tps = [data[str(b)]["tokens_per_second"] for b in batches]
        ax.plot(batches, tps, "o-", color=COLORS[cfg], label=LABELS[cfg],
                markersize=4, linewidth=1.5)
    ax.set_xlabel("Batch Size (concurrent requests)")
    ax.set_ylabel("Aggregate Throughput (tok/s)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Throughput Scaling with Batch Size")
    _save(fig, "eval_throughput_vs_batch")


# ---------------------------------------------------------------------------
# Figure 2: Peak throughput bar chart
# ---------------------------------------------------------------------------
def fig_peak_throughput():
    fig, ax = plt.subplots(figsize=(6, 3))
    cfgs = ["cpu_4", "cpu_16", "cpu_48", "gpu_25", "gpu_100"]
    peaks = [MEASURED[c] for c in cfgs]
    bars = ax.bar(range(len(cfgs)), peaks,
                  color=[COLORS[c] for c in cfgs], edgecolor="white", width=0.6)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([LABELS[c] for c in cfgs], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Peak Throughput (tok/s)")
    ax.set_title("Peak Aggregate Throughput by Configuration")
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "eval_peak_throughput")


# ---------------------------------------------------------------------------
# Figure 3: Cost per token vs demand level
# ---------------------------------------------------------------------------
def fig_cost_per_token_vs_demand():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    demand_range = np.linspace(1, 1600, 500)
    for cfg in ["cpu_4", "cpu_16", "cpu_48", "gpu_25", "gpu_100"]:
        cost_h = CONFIGS[cfg]["hourly_cost"]
        peak = MEASURED[cfg]
        # Cost per token only valid up to peak throughput
        valid = demand_range <= peak
        cpt = np.where(valid, cost_h / (demand_range * 3600) * 1e6, np.nan)
        ax.plot(demand_range, cpt, color=COLORS[cfg], label=LABELS[cfg], linewidth=1.5)
        # Mark peak with a dot
        ax.plot(peak, cost_h / (peak * 3600) * 1e6, "o",
                color=COLORS[cfg], markersize=5)
    ax.set_xlabel("Demand (tok/s)")
    ax.set_ylabel("Cost per Token (μ\$)")
    ax.set_title("Cost per Token vs. Demand Level")
    ax.set_xlim(0, 1650)
    ax.set_ylim(0, 3.0)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, "eval_cost_per_token_vs_demand")


# ---------------------------------------------------------------------------
# Figure 4: Tier-ordered scaling ladder
# ---------------------------------------------------------------------------
def fig_scaling_ladder():
    fig, ax = plt.subplots(figsize=(6, 3))
    costs = [CONFIGS[c]["hourly_cost"] for c in TIER_ORDER]
    peaks = [MEASURED[c] for c in TIER_ORDER]
    colors = [COLORS[c] for c in TIER_ORDER]
    labels = [LABELS[c] for c in TIER_ORDER]

    x = range(len(TIER_ORDER))
    ax2 = ax.twinx()

    bars = ax.bar(x, costs, color=[COLORS[c] for c in TIER_ORDER], edgecolor="white", width=0.5,
                  label="Hourly Cost")
    ax2.plot(x, peaks, "ks-", markersize=6, linewidth=1.5, label="Peak Throughput")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Hourly Cost (\\$)")
    ax2.set_ylabel("Peak Throughput (tok/s)")
    ax.set_title("Scaling Ladder (by tier order)")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "eval_scaling_ladder")


# ---------------------------------------------------------------------------
# Figure 5: Scaling demo workload profile
# ---------------------------------------------------------------------------
def fig_scaling_demo():
    with open(ROOT / "benchmarks" / "thesis_figures" / "scaling_demo_data.json") as f:
        demo = json.load(f)

    elapsed = [e["elapsed"] / 60.0 for e in demo]  # minutes
    tps = [e["throughput_tps"] for e in demo]
    active = [e["active_requests"] for e in demo]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax2 = ax.twinx()

    # Draw background spans colored by active config
    prev_cfg = demo[0]["config_id"]
    span_start = elapsed[0]
    # Track which configs we've already added to legend
    legend_added: set[str] = set()
    for i in range(1, len(demo)):
        cur_cfg = demo[i]["config_id"]
        if cur_cfg != prev_cfg or i == len(demo) - 1:
            lbl = LABELS[prev_cfg] if prev_cfg not in legend_added else None
            ax.axvspan(span_start, elapsed[i], alpha=0.25, color=COLORS[prev_cfg],
                       label=lbl)
            legend_added.add(prev_cfg)
            span_start = elapsed[i]
            prev_cfg = cur_cfg
    # Final span
    if prev_cfg not in legend_added:
        lbl = LABELS[prev_cfg]
    else:
        lbl = None
    ax.axvspan(span_start, elapsed[-1], alpha=0.25, color=COLORS[prev_cfg],
               label=lbl)

    # Plot aggregate throughput line
    ax.plot(elapsed, tps, color="black", linewidth=0.8, alpha=0.9,
            label="Aggregate tok/s")
    # Plot active requests on secondary axis
    ax2.plot(elapsed, active, color="gray", linewidth=0.6, alpha=0.5,
             linestyle="--", label="Active requests")
    ax2.set_ylabel("Active Requests", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel("Aggregate Throughput (tok/s)")
    ax.set_title("Scaling Demo: Throughput and Hardware Transitions")
    ax.set_xlim(elapsed[0], elapsed[-1])
    ax.set_ylim(bottom=0)
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, "eval_scaling_demo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating evaluation figures...")
    fig_throughput_vs_batch()
    fig_peak_throughput()
    fig_cost_per_token_vs_demand()
    fig_scaling_ladder()
    fig_scaling_demo()
    print(f"Done. Figures saved to {OUT}")
