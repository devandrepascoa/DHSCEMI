#!/usr/bin/env python3
"""
Generate thesis-quality plots from a full scaling simulation
(cpu_4 → cpu_12 → gpu_25 → gpu_100).

Uses the same /metrics-based throughput threshold model as
scaling_simulation.py and scaling_demo_server.py.

Scaling logic:
  - Scale UP:   throughput_ema >= SCALE_UP_MULT * measured_throughput[current]
  - Scale DOWN: throughput_ema <= SCALE_DOWN_MULT * measured_throughput[cheaper]

Plot 1: Hardware config + demand (tok/s) vs time
Plot 2: Cost/hour vs time
Plot 3: Cost per token vs time
Plot 4: Dynamic vs static provisioning cost

Usage:
    uv run python benchmarks/scaling_simulation_plots.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from pathlib import Path
from main_cost_aware import HardwareConfig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Hardware configs — ordered by cost (cheapest first)
# ---------------------------------------------------------------------------
CONFIGS = [
    HardwareConfig(cpu_cores=4,  memory="8g",  hourly_cost=0.05),
    HardwareConfig(cpu_cores=12, memory="8g",  hourly_cost=0.12),
    HardwareConfig(cpu_cores=2,  memory="8g",  gpu_percentage=25,  hourly_cost=0.50),
    HardwareConfig(cpu_cores=2,  memory="16g", gpu_percentage=100, hourly_cost=4.00),
]
CONFIGS_BY_COST = sorted(CONFIGS, key=lambda c: c.hourly_cost)

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
COOLDOWN = 300
SCALING_CHECK_INTERVAL = 10

# ---------------------------------------------------------------------------
# Metrics-based scaling parameters (must match scaling_simulation.py)
# ---------------------------------------------------------------------------
SCALE_UP_MULT = 0.8
SCALE_DOWN_MULT = 0.3
EMA_ALPHA = 2.0 / (60 + 1)  # ~60s EMA window

# Measured aggregate throughput per config (from benchmarks)
MEASURED_THROUGHPUT = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}

# Single-request throughput (batch=1) — for simulation duration calc
SINGLE_REQUEST_TPS = {
    "cpu_4":   9.0,
    "cpu_12":  15.4,
    "gpu_25":  13.3,
    "gpu_100": 152.9,
}

TOKENS_PER_REQUEST = 140

# ---------------------------------------------------------------------------
# Workload phases: (name, duration_s, concurrency, rpm)
# ---------------------------------------------------------------------------
PHASES = [
    ("low load",       900,  1,   3),
    ("medium load",    900,  4,  15),
    ("high load",      900,  8,   0),
    ("peak load",      900, 30,   0),
    ("sustain gpu",    600, 30,   0),
    ("ramp-down 1",    900,  4,  43),
    ("ramp-down 2",    900,  4,  15),
    ("ramp-down 3",    900,  1,   3),
    ("low load",       600,  1,   3),
]

CONFIG_ORDER = ["cpu_4", "cpu_12", "gpu_25", "gpu_100"]
CONFIG_COLORS = {
    "cpu_4":   "#4e79a7",
    "cpu_12":  "#f28e2b",
    "gpu_25":  "#e15759",
    "gpu_100": "#b07aa1",
}

PHASE_COLORS = {
    "low load":     "#e8f4f8",
    "medium load":  "#fff3e0",
    "high load":    "#fce4ec",
    "peak load":    "#f8d7da",
    "sustain gpu":  "#f3e5f5",
    "ramp-down 1":  "#e8f5e9",
    "ramp-down 2":  "#e0f2f1",
    "ramp-down 3":  "#ede7f6",
}

OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Scaling decision (same as scaling_simulation.py)
# ---------------------------------------------------------------------------

def select_config(
    current_config: HardwareConfig,
    throughput_ema: float,
) -> HardwareConfig:
    current_id = current_config.config_id()
    current_idx = next(
        i for i, c in enumerate(CONFIGS_BY_COST) if c.config_id() == current_id
    )
    current_capacity = MEASURED_THROUGHPUT[current_id]

    if throughput_ema >= SCALE_UP_MULT * current_capacity:
        if current_idx + 1 < len(CONFIGS_BY_COST):
            return CONFIGS_BY_COST[current_idx + 1]

    if current_idx > 0:
        cheaper = CONFIGS_BY_COST[current_idx - 1]
        cheaper_capacity = MEASURED_THROUGHPUT[cheaper.config_id()]
        if throughput_ema <= SCALE_DOWN_MULT * cheaper_capacity:
            return cheaper

    return current_config


# ---------------------------------------------------------------------------
# Simulation (same model as scaling_simulation.py)
# ---------------------------------------------------------------------------

def simulate_timeseries() -> list:
    current_config = CONFIGS_BY_COST[0]
    last_scale_time = 0.0
    throughput_ema = 0.0
    last_ema_time = 0.0

    samples = []
    total_elapsed = 0.0

    for phase_name, duration, concurrency, rpm in PHASES:
        if concurrency > 0 and rpm > 0:
            worker_interval = 60.0 * concurrency / rpm
        else:
            worker_interval = 0

        workers = [{"next_start": total_elapsed, "busy_until": 0.0}
                   for _ in range(concurrency)]

        t = total_elapsed
        end_t = total_elapsed + duration

        while t < end_t:
            config_id = current_config.config_id()
            single_tps = SINGLE_REQUEST_TPS.get(config_id, 9.0)

            active_count = 0
            for w in workers:
                if w["busy_until"] > 0 and t >= w["busy_until"]:
                    w["busy_until"] = 0.0

                if w["busy_until"] == 0 and t >= w["next_start"]:
                    base_dur = TOKENS_PER_REQUEST / single_tps
                    w["busy_until"] = t + base_dur
                    if worker_interval > 0:
                        w["next_start"] = t + worker_interval
                    else:
                        w["next_start"] = t

                if w["busy_until"] > t:
                    active_count += 1

            aggregate_tps = single_tps * active_count if active_count > 0 else 0.0

            dt = t - last_ema_time
            if dt > 0:
                decay = (1.0 - EMA_ALPHA) ** dt
                throughput_ema = throughput_ema * decay + EMA_ALPHA * aggregate_tps
                last_ema_time = t

            if int(t) % SCALING_CHECK_INTERVAL == 0 and t > total_elapsed:
                if t - last_scale_time >= COOLDOWN:
                    optimal = select_config(current_config, throughput_ema)
                    if optimal.config_id() != current_config.config_id():
                        current_config = optimal
                        last_scale_time = t

            if int(t) % 5 == 0:
                cid = current_config.config_id()
                throughput = MEASURED_THROUGHPUT.get(cid, 1.0)
                cost_per_token = (
                    current_config.hourly_cost / (throughput * 3600)
                    if throughput > 0 else float('inf')
                )
                samples.append({
                    "time_min": t / 60.0,
                    "phase": phase_name,
                    "config_id": cid,
                    "demand_tps": throughput_ema,
                    "hourly_cost": current_config.hourly_cost,
                    "throughput": throughput,
                    "cost_per_token": cost_per_token,
                })

            t += 1.0
        total_elapsed = end_t

    return samples


def _phase_spans(samples: list) -> list:
    spans = []
    prev = samples[0]["phase"]
    start = samples[0]["time_min"]
    for s in samples[1:]:
        if s["phase"] != prev:
            spans.append((start, s["time_min"], prev))
            start = s["time_min"]
            prev = s["phase"]
    spans.append((start, samples[-1]["time_min"], prev))
    return spans


def _common_rc():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def generate_plots(samples: list) -> None:
    _common_rc()

    t = np.array([s["time_min"] for s in samples])
    demand = np.array([s["demand_tps"] for s in samples])
    configs = [s["config_id"] for s in samples]
    costs = np.array([s["hourly_cost"] for s in samples])
    cpt = np.array([s["cost_per_token"] for s in samples])
    config_idx = np.array([CONFIG_ORDER.index(c) for c in configs])

    phase_spans = _phase_spans(samples)

    scale_times = []
    for i in range(1, len(configs)):
        if configs[i] != configs[i - 1]:
            scale_times.append(t[i])

    # ==================================================================
    # Plot 1: Hardware config + demand vs time
    # ==================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 5.5))

    for s, e, p in phase_spans:
        ax1.axvspan(s, e, alpha=0.12, color=PHASE_COLORS.get(p, "#f5f5f5"), zorder=0)

    for i in range(len(t) - 1):
        c = configs[i]
        ax1.fill_between([t[i], t[i + 1]], [config_idx[i], config_idx[i + 1]],
                         alpha=0.35, color=CONFIG_COLORS[c], step="post", zorder=2)
    ax1.step(t, config_idx, where="post", color="#333", linewidth=1.8, zorder=3)

    ax1b = ax1.twinx()
    ax1b.plot(t, demand, color="#e15759", linewidth=1.4, alpha=0.85, zorder=4)
    ax1b.fill_between(t, demand, alpha=0.06, color="#e15759")
    ax1b.set_ylabel("Throughput EMA (tok/s)", color="#e15759")
    ax1b.tick_params(axis="y", labelcolor="#e15759")
    ax1b.spines["right"].set_visible(True)

    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax1b.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                         alpha=0.5, linewidth=1)
            ax1b.text(t[-1] * 1.01, thr, "%d" % thr, fontsize=7,
                      va="center", color=CONFIG_COLORS[cid])

    for st in scale_times:
        ax1.axvline(st, color="red", linestyle="--", alpha=0.3, linewidth=1, zorder=1)

    ymax_demand = max(demand) if max(demand) > 0 else 1
    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax1b.text(mid, ymax_demand * 1.08, p, ha="center", va="bottom",
                  fontsize=7, fontstyle="italic", color="#555")

    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Hardware Configuration")
    ax1.set_yticks(range(len(CONFIG_ORDER)))
    ax1.set_yticklabels(CONFIG_ORDER)
    ax1.set_ylim(-0.5, len(CONFIG_ORDER) - 0.5)
    ax1.set_title("Cost-Aware Vertical Scaling Under Varying Workload")

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.5)
               for c in CONFIG_ORDER]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=len(CONFIG_ORDER))

    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(OUT_DIR / ("sim_config_vs_time.%s" % ext),
                     bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print("Saved: sim_config_vs_time")

    # ==================================================================
    # Plot 2: Cost/hour vs time
    # ==================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 4.5))

    for s, e, p in phase_spans:
        ax2.axvspan(s, e, alpha=0.12, color=PHASE_COLORS.get(p, "#f5f5f5"), zorder=0)

    for i in range(len(t) - 1):
        c = configs[i]
        ax2.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS[c], step="post", zorder=2)
    ax2.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3)

    ax2b = ax2.twinx()
    ax2b.plot(t, demand, color="#e15759", linewidth=1.2, alpha=0.6, zorder=4)
    ax2b.set_ylabel("Throughput EMA (tok/s)", color="#e15759")
    ax2b.tick_params(axis="y", labelcolor="#e15759")
    ax2b.spines["right"].set_visible(True)

    for st in scale_times:
        ax2.axvline(st, color="red", linestyle="--", alpha=0.3, linewidth=1, zorder=1)

    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax2.text(mid, max(costs) * 1.12, p, ha="center", va="bottom",
                 fontsize=7, fontstyle="italic", color="#555")

    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Hourly Cost ($)")
    ax2.set_ylim(0, max(costs) * 1.3)
    ax2.set_title("Infrastructure Cost Over Time")

    cost_legend = [
        mpatches.Patch(
            color=CONFIG_COLORS[c],
            label="%s — $%.2f/hr" % (c, CONFIGS_BY_COST[i].hourly_cost),
            alpha=0.5,
        )
        for i, c in enumerate(CONFIG_ORDER)
    ]
    ax2.legend(handles=cost_legend, loc="upper left", fontsize=8, ncol=4)

    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(OUT_DIR / ("sim_cost_vs_time.%s" % ext),
                     bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print("Saved: sim_cost_vs_time")

    # ==================================================================
    # Plot 3: Cost per token vs time
    # ==================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 4.5))

    for s, e, p in phase_spans:
        ax3.axvspan(s, e, alpha=0.12, color=PHASE_COLORS.get(p, "#f5f5f5"), zorder=0)

    cpt_micro = cpt * 1e6
    for i in range(len(t) - 1):
        c = configs[i]
        ax3.fill_between([t[i], t[i + 1]], [cpt_micro[i], cpt_micro[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS[c], step="post", zorder=2)
    ax3.step(t, cpt_micro, where="post", color="#333", linewidth=1.8, zorder=3)

    ax3b = ax3.twinx()
    ax3b.plot(t, demand, color="#e15759", linewidth=1.2, alpha=0.6, zorder=4)
    ax3b.set_ylabel("Throughput EMA (tok/s)", color="#e15759")
    ax3b.tick_params(axis="y", labelcolor="#e15759")
    ax3b.spines["right"].set_visible(True)

    for st in scale_times:
        ax3.axvline(st, color="red", linestyle="--", alpha=0.3, linewidth=1, zorder=1)

    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax3.text(mid, max(cpt_micro) * 1.12, p, ha="center", va="bottom",
                 fontsize=7, fontstyle="italic", color="#555")

    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Cost per Token (μ$)")
    ax3.set_ylim(0, max(cpt_micro) * 1.3)
    ax3.set_title("Cost per Token Over Time")
    ax3.legend(handles=cost_legend, loc="upper right", fontsize=8, ncol=4)

    fig3.tight_layout()
    for ext in ("pdf", "png"):
        fig3.savefig(OUT_DIR / ("sim_cost_per_token_vs_time.%s" % ext),
                     bbox_inches="tight", dpi=300)
    plt.close(fig3)
    print("Saved: sim_cost_per_token_vs_time")

    # ==================================================================
    # Plot 4: Dynamic vs static provisioning cost
    # ==================================================================
    fig4, ax4 = plt.subplots(figsize=(14, 4.5))

    for s, e, p in phase_spans:
        ax4.axvspan(s, e, alpha=0.12, color=PHASE_COLORS.get(p, "#f5f5f5"), zorder=0)

    for i in range(len(t) - 1):
        c = configs[i]
        ax4.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS[c], step="post", zorder=2)

    static_cost = CONFIGS_BY_COST[-1].hourly_cost  # gpu_100
    ax4.axhline(static_cost, linestyle="--", color="#b07aa1", alpha=0.6,
                linewidth=1.5, label="Static gpu_100 ($%.2f/hr)" % static_cost)
    ax4.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3,
             label="Dynamic cost")

    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("Hourly Cost ($)")
    ax4.set_ylim(0, static_cost * 1.4)
    ax4.set_title("Dynamic vs Static Provisioning Cost")
    ax4.legend(loc="upper left", fontsize=9)

    fig4.tight_layout()
    for ext in ("pdf", "png"):
        fig4.savefig(OUT_DIR / ("sim_cost_vs_demand.%s" % ext),
                     bbox_inches="tight", dpi=300)
    plt.close(fig4)
    print("Saved: sim_cost_vs_demand")


if __name__ == "__main__":
    print("Running scaling simulation (cpu_4 -> cpu_12 -> gpu_25 -> gpu_100)...")
    samples = simulate_timeseries()
    print("Generated %d samples over %.1f min" % (len(samples), samples[-1]["time_min"]))

    for cid in CONFIG_ORDER:
        pts = [s for s in samples if s["config_id"] == cid]
        if pts:
            d = [s["demand_tps"] for s in pts]
            print("  %s: %d samples, demand %.1f-%.1f tok/s" % (cid, len(pts), min(d), max(d)))
        else:
            print("  %s: never selected" % cid)

    generate_plots(samples)
    print("Done.")
