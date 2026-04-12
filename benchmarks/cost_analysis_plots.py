"""Alternative cost-analysis plots for the thesis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "thesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "hardware_configs.json") as f:
    hw = json.load(f)

CONFIGS = {c["config_id"]: c for c in hw["configs"]}
MEASURED = hw["measured_throughput"]
TIER_ORDER = sorted(CONFIGS.keys(), key=lambda c: CONFIGS[c]["tier_order"])

_cmap = plt.cm.Blues
_n = len(TIER_ORDER)
_gradient_vals = [0.3 + 0.55 * i / max(_n - 1, 1) for i in range(_n)]
COLORS = {c: _cmap(v) for c, v in zip(TIER_ORDER, _gradient_vals)}
LABELS = {
    "cpu_4": "cpu_4 (4 cores)",
    "cpu_16": "cpu_16 (16 cores)",
    "cpu_48": "cpu_48 (48 cores)",
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
# Plot 1: Hourly cost of optimal config vs demand
# Shows the autoscaler stepping up only when forced by capacity limits.
# ---------------------------------------------------------------------------
def plot_hourly_cost_vs_demand():
    demand = np.linspace(1, 1700, 2000)

    # For each demand level, find cheapest config that can handle it
    optimal_cost = []
    optimal_cfg = []
    for d in demand:
        viable = [(cid, CONFIGS[cid]["hourly_cost"])
                  for cid in TIER_ORDER if MEASURED[cid] >= d]
        if viable:
            best = min(viable, key=lambda x: x[1])
            optimal_cost.append(best[1])
            optimal_cfg.append(best[0])
        else:
            optimal_cost.append(np.nan)
            optimal_cfg.append(None)

    # Also show static GPU cost as baseline
    gpu_cost = CONFIGS["gpu_100"]["hourly_cost"]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Color the background by which config is selected
    prev_cfg = optimal_cfg[0]
    start_idx = 0
    for i in range(1, len(demand)):
        if optimal_cfg[i] != prev_cfg or i == len(demand) - 1:
            if prev_cfg is not None:
                ax.axvspan(demand[start_idx], demand[i],
                           alpha=0.15, color=COLORS[prev_cfg])
            start_idx = i
            prev_cfg = optimal_cfg[i]

    ax.plot(demand, optimal_cost, color="#333", linewidth=2,
            label="Dynamic (autoscaler)")
    ax.axhline(gpu_cost, color=COLORS["gpu_100"], linestyle="--",
               linewidth=1.5, alpha=0.7, label=f"Static GPU (${ gpu_cost}/hr)")

    # Mark transition points
    for i in range(1, len(optimal_cfg)):
        if optimal_cfg[i] != optimal_cfg[i-1] and optimal_cfg[i] is not None:
            ax.axvline(demand[i], color="#888", linestyle=":", alpha=0.5, linewidth=1)
            ax.annotate(LABELS[optimal_cfg[i]],
                        xy=(demand[i], optimal_cost[i]),
                        xytext=(demand[i] + 30, optimal_cost[i] + 0.05),
                        fontsize=7, color=COLORS[optimal_cfg[i]],
                        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))

    ax.set_xlabel("Demand (tok/s)")
    ax.set_ylabel("Hourly Cost (\\$)")
    ax.set_title("Autoscaler Hourly Cost vs. Demand")
    ax.set_xlim(0, 1700)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    _save(fig, "alt_hourly_cost_vs_demand")


# ---------------------------------------------------------------------------
# Plot 2: Cost per token with overprovisioning waste
# When you provision a config but demand is below its peak, you're paying
# for unused capacity. cost_per_token = hourly / (peak * 3600) is the
# floor, but if demand < peak you're wasting the difference.
# This shows: for each config, cost_per_token = hourly / (demand * 3600)
# when demand <= peak (you pay full price, get fewer tokens).
# Overlaid: the OPTIMAL envelope (cheapest viable config at each demand).
# ---------------------------------------------------------------------------
def plot_cost_envelope():
    demand = np.linspace(1, 1700, 2000)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Individual config curves
    for cfg in TIER_ORDER:
        cost_h = CONFIGS[cfg]["hourly_cost"]
        peak = MEASURED[cfg]
        valid = demand <= peak
        cpt = np.where(valid, cost_h / (demand * 3600) * 1e6, np.nan)
        ax.plot(demand, cpt, color=COLORS[cfg], label=LABELS[cfg],
                linewidth=1.2, alpha=0.5)
        ax.plot(peak, cost_h / (peak * 3600) * 1e6, "o",
                color=COLORS[cfg], markersize=5)

    # Optimal envelope: at each demand, cheapest viable config
    envelope = []
    for d in demand:
        viable = [(cid, CONFIGS[cid]["hourly_cost"] / (d * 3600) * 1e6)
                  for cid in TIER_ORDER if MEASURED[cid] >= d]
        if viable:
            envelope.append(min(viable, key=lambda x: x[1])[1])
        else:
            envelope.append(np.nan)

    ax.plot(demand, envelope, color="#222", linewidth=2.5,
            label="Optimal (autoscaler)", zorder=5)

    # Static GPU line for comparison
    gpu_static = CONFIGS["gpu_100"]["hourly_cost"] / (demand * 3600) * 1e6
    ax.plot(demand, gpu_static, color=COLORS["gpu_100"], linewidth=1.5,
            linestyle="--", alpha=0.7, label="Static GPU 100%")

    ax.set_xlabel("Demand (tok/s)")
    ax.set_ylabel("Cost per Token (μ\\$)")
    ax.set_title("Cost per Token: Autoscaler Envelope vs. Static Provisioning")
    ax.set_xlim(0, 1700)
    ax.set_ylim(0, 3.0)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, "alt_cost_envelope")


# ---------------------------------------------------------------------------
# Plot 3: Cost savings percentage (dynamic vs static GPU)
# ---------------------------------------------------------------------------
def plot_savings_vs_demand():
    max_demand = MEASURED["gpu_100"]  # clip at GPU peak — nothing works beyond
    demand = np.linspace(1, max_demand, 2000)

    static_cpt = CONFIGS["gpu_100"]["hourly_cost"] / (demand * 3600)

    # Autoscaler picks cheapest hourly config that can handle the demand
    dynamic_cpt = []
    for d in demand:
        viable = [(cid, CONFIGS[cid]["hourly_cost"] / (d * 3600))
                  for cid in TIER_ORDER if MEASURED[cid] >= d]
        if viable:
            dynamic_cpt.append(min(viable, key=lambda x: x[1])[1])
        else:
            # At GPU peak, use GPU cost (shouldn't happen with clipped range)
            dynamic_cpt.append(CONFIGS["gpu_100"]["hourly_cost"] / (d * 3600))
    dynamic_cpt = np.array(dynamic_cpt)

    savings = (1 - dynamic_cpt / static_cpt) * 100

    fig, ax = plt.subplots(figsize=(7, 4))

    # Build segments per config for colored fills
    segments = []  # (start_idx, end_idx, config_id)
    prev_best = None
    seg_start = 0
    for i, d in enumerate(demand):
        viable = [(cid, CONFIGS[cid]["hourly_cost"])
                  for cid in TIER_ORDER if MEASURED[cid] >= d]
        if viable:
            best = min(viable, key=lambda x: x[1])[0]
        else:
            best = "gpu_100"
        if best != prev_best:
            if prev_best is not None:
                segments.append((seg_start, i, prev_best))
            seg_start = i
            prev_best = best
    segments.append((seg_start, len(demand) - 1, prev_best))

    # Colored fill per config region
    for s_start, s_end, cfg in segments:
        ax.fill_between(demand[s_start:s_end+1], savings[s_start:s_end+1],
                        alpha=0.25, color=COLORS[cfg])
        ax.plot(demand[s_start:s_end+1], savings[s_start:s_end+1],
                color=COLORS[cfg], linewidth=2)
        # Label in the middle of the region
        mid_idx = (s_start + s_end) // 2
        mid_y = savings[mid_idx]
        if mid_y > 5:  # only label if there's visible savings
            # Place label to the right of the region midpoint to avoid y-axis clipping
            label_x = max(demand[mid_idx], 120)
            ax.annotate(LABELS[cfg], xy=(label_x, mid_y),
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=COLORS[cfg])

    # Vertical lines at transitions
    for s_start, s_end, cfg in segments[1:]:
        ax.axvline(demand[s_start], color="#888", linestyle=":", alpha=0.5)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Demand (tok/s)")
    ax.set_ylabel("Cost Savings vs. Static GPU (%)")
    ax.set_title("Cost Savings from Dynamic Scaling vs. Always-On GPU")
    ax.set_xlim(0, max_demand)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    _save(fig, "alt_savings_vs_demand")


# ---------------------------------------------------------------------------
# Plot 4: Efficiency ratio (throughput per dollar)
# ---------------------------------------------------------------------------
def plot_throughput_per_dollar():
    fig, ax = plt.subplots(figsize=(6, 4))

    cfgs = TIER_ORDER
    tpd = [MEASURED[c] / CONFIGS[c]["hourly_cost"] for c in cfgs]
    bars = ax.bar(range(len(cfgs)), tpd,
                  color=[COLORS[c] for c in cfgs], edgecolor="white", width=0.6)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([LABELS[c] for c in cfgs], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Throughput per Dollar (tok/s per \\$/hr)")
    ax.set_title("Throughput Efficiency by Configuration")
    for bar, val in zip(bars, tpd):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "alt_throughput_per_dollar")


if __name__ == "__main__":
    print("Generating alternative cost analysis plots...\n")
    plot_hourly_cost_vs_demand()
    plot_cost_envelope()
    plot_savings_vs_demand()
    plot_throughput_per_dollar()
    print(f"\nDone. Check {OUT}/")
