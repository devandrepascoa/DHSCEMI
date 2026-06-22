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
# Figure 1: Throughput vs batch size (line plot)
# ---------------------------------------------------------------------------
def fig_throughput_vs_batch():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cfg in TIER_ORDER:
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
    cfgs = TIER_ORDER
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
# Figure 3: Cost savings from dynamic scaling vs always-on GPU
# ---------------------------------------------------------------------------
def fig_cost_savings_vs_demand():
    max_demand = MEASURED["gpu_100"]
    demand = np.linspace(1, max_demand, 2000)

    static_cpt = CONFIGS["gpu_100"]["hourly_cost"] / (demand * 3600)

    dynamic_cpt = []
    for d in demand:
        viable = [(cid, CONFIGS[cid]["hourly_cost"] / (d * 3600))
                  for cid in TIER_ORDER if MEASURED[cid] >= d]
        if viable:
            dynamic_cpt.append(min(viable, key=lambda x: x[1])[1])
        else:
            dynamic_cpt.append(CONFIGS["gpu_100"]["hourly_cost"] / (d * 3600))
    dynamic_cpt = np.array(dynamic_cpt)

    savings = (1 - dynamic_cpt / static_cpt) * 100

    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Build segments per config for colored fills
    segments = []
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

    for s_start, s_end, cfg in segments:
        ax.fill_between(demand[s_start:s_end+1], savings[s_start:s_end+1],
                        alpha=0.25, color=COLORS[cfg])
        ax.plot(demand[s_start:s_end+1], savings[s_start:s_end+1],
                color=COLORS[cfg], linewidth=2)
        mid_idx = (s_start + s_end) // 2
        mid_y = savings[mid_idx]
        if mid_y > 5:
            label_x = max(demand[mid_idx], 120)
            ax.annotate(LABELS[cfg], xy=(label_x, mid_y),
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=COLORS[cfg])

    for s_start, s_end, cfg in segments[1:]:
        ax.axvline(demand[s_start], color="#888", linestyle=":", alpha=0.5)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Demand (tok/s)")
    ax.set_ylabel("Cost Savings vs. Static GPU (%)")
    ax.set_title("Cost Savings from Dynamic Scaling vs. Always-On GPU")
    ax.set_xlim(0, max_demand)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    _save(fig, "eval_cost_savings_vs_demand")


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
# Real autoscaler path: steps through tiers by tier_order (incl. cpu_48)
# ---------------------------------------------------------------------------
def fig_savings_real_path():
    max_demand = MEASURED["gpu_100"]
    demand = np.linspace(1, max_demand, 2000)
    gpu_hourly = CONFIGS["gpu_100"]["hourly_cost"]

    def savings(cfg):
        return (1 - CONFIGS[cfg]["hourly_cost"] / gpu_hourly) * 100

    # Autoscaler path: climbs the tier_order ladder, stops at the lowest tier
    # whose peak throughput can serve the demand.
    path_cfg = [None] * len(demand)
    for i, d in enumerate(demand):
        viable = [c for c in TIER_ORDER if MEASURED[c] >= d]
        if viable:
            path_cfg[i] = viable[0]

    # Collapse into contiguous segments
    segs = []
    start = 0
    for i in range(1, len(demand)):
        if path_cfg[i] != path_cfg[i - 1]:
            segs.append((start, i - 1, path_cfg[i - 1]))
            start = i
    segs.append((start, len(demand) - 1, path_cfg[-1]))
    segs = [(a, b, c) for a, b, c in segs if c is not None]

    fig, ax = plt.subplots(figsize=(7, 3.6))

    # Crop the view: gpu_100 is flat at 0 % out to its 1574 tok/s peak, so the
    # far-right band is dead space. Show just past cpu_48's peak.
    non_gpu_peak = max(MEASURED[c] for c in TIER_ORDER if c != "gpu_100")
    x_view_max = non_gpu_peak * 1.4

    prev = None
    for a, b, cfg in segs:
        x0, x1, y = demand[a], demand[b], savings(cfg)
        # save/loss shading under the step (green = saves, red = loses vs GPU)
        ax.fill_between([x0, x1], 0, y,
                        color=("#82B366" if y >= 0 else "#B85450"), alpha=0.18)
        # single neutral step line for the autoscaler path
        ax.plot([x0, x1], [y, y], color="#2b2b2b", linewidth=2.4, solid_capstyle="butt")
        if prev is not None:
            ax.plot([x0, x0], [prev, y], color="#2b2b2b", linewidth=1.2)
        # label centered within the *visible* portion of the step
        vx1 = min(x1, x_view_max)
        ax.annotate(f"{cfg}  {y:+.0f}%", xy=((x0 + vx1) / 2, y),
                    xytext=(0, -11 if y >= 0 else 11), textcoords="offset points",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#2b2b2b",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        prev = y

    ax.axhline(0, color="#555555", linewidth=1.0)
    ax.text(x_view_max, 3, "always-on GPU", ha="right", va="bottom",
            fontsize=7, color="#555555")

    ax.set_xlabel("Demand (tok/s)")
    ax.set_ylabel("Cost Savings vs. GPU (%)")
    ax.set_title("Autoscaler Cost Savings vs. Demand")
    ax.set_xlim(0, x_view_max)
    ax.set_ylim(-55, 100)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "eval_savings_real_path")


# ---------------------------------------------------------------------------
# Figure 5: Scaling demo workload profile
# ---------------------------------------------------------------------------
def fig_scaling_demo():
    import matplotlib.patches as mpatches

    with open(ROOT / "benchmarks" / "thesis_figures" / "scaling_demo_data.json") as f:
        demo = json.load(f)

    t = np.array([e["elapsed"] / 60.0 for e in demo])  # minutes
    active = np.array([e.get("active_requests", 0) for e in demo], dtype=float)
    configs = [e["config_id"] for e in demo]
    config_idx = np.array([TIER_ORDER.index(c) if c in TIER_ORDER else -1
                           for c in configs])

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Config step function with colored fill (left axis)
    for i in range(len(t) - 1):
        c = configs[i]
        ax.fill_between([t[i], t[i + 1]], [config_idx[i], config_idx[i + 1]],
                        alpha=0.35, color=COLORS[c], step="post", zorder=2)
    ax.step(t, config_idx, where="post", color="#333", linewidth=1.8, zorder=3)

    # Active requests (concurrency) on twin axis -- the scaling trigger
    ax2 = ax.twinx()
    ax2.plot(t, active, color="#B85450", linewidth=1.3, alpha=0.9, zorder=4,
             label="Active requests")
    ax2.fill_between(t, active, alpha=0.08, color="#B85450")
    ax2.set_ylabel("Active Requests (concurrency)", color="#B85450")
    ax2.tick_params(axis="y", labelcolor="#B85450")

    # Scaling event vertical lines
    for i in range(1, len(configs)):
        if configs[i] != configs[i - 1]:
            ax.axvline(t[i], color="#888888", linestyle="--", alpha=0.4,
                       linewidth=1, zorder=1)

    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel("Hardware Configuration")
    ax.set_yticks(range(len(TIER_ORDER)))
    ax.set_yticklabels([LABELS[c] for c in TIER_ORDER], fontsize=8)
    ax.set_ylim(-0.5, len(TIER_ORDER) - 0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_title("Scaling Experiment: Hardware Transitions vs. Concurrency")
    ax.grid(True, alpha=0.3)
    _save(fig, "eval_scaling_demo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating evaluation figures...")
    fig_throughput_vs_batch()
    fig_peak_throughput()
    fig_cost_savings_vs_demand()
    fig_savings_real_path()
    fig_scaling_ladder()
    fig_scaling_demo()
    print(f"Done. Figures saved to {OUT}")
