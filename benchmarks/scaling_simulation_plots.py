#!/usr/bin/env python3
"""
Generate thesis-quality plots from the CPU scaling simulation.

Plot 1: Hardware config + demand (tok/s) vs time
Plot 2: Cost/hour vs time

Usage:
    uv run python benchmarks/scaling_simulation_plots.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from pathlib import Path
from main_cost_aware import (
    DemandTracker,
    HardwareConfig,
    CostAwareAutoscaler,
    DEFAULT_THROUGHPUT,
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CONFIGS = [
    HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40),
    HardwareConfig(cpu_cores=8, memory="16g", hourly_cost=0.80),
    HardwareConfig(cpu_cores=12, memory="24g", hourly_cost=1.20),
]
DEFAULT_THROUGHPUT["cpu_12"] = 22.0

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
COOLDOWN = 300
DEMAND_WINDOW = 60
TOKENS_PER_REQUEST = 140

PHASES = [
    ("warm-up",      120, 1, 2),
    ("medium load",  420, 2, 4),
    ("high load",    420, 3, 9),
    ("sustain",      120, 3, 9),
    ("ramp-down 1",  420, 2, 4),
    ("ramp-down 2",  420, 1, 2),
    ("low load",      60, 1, 2),
]

BASE_DURATION = {"cpu_4": 20.0, "cpu_8": 10.0, "cpu_12": 8.0}
CONFIG_ORDER = ["cpu_4", "cpu_8", "cpu_12"]
CONFIG_COLORS = {"cpu_4": "#4e79a7", "cpu_8": "#59a14f", "cpu_12": "#f28e2b"}

PHASE_COLORS = {
    "warm-up": "#e8f4f8", "medium load": "#fff3e0",
    "high load": "#fce4ec", "sustain": "#fce4ec",
    "ramp-down 1": "#e8f5e9", "ramp-down 2": "#ede7f6",
    "low load": "#e8f4f8",
}

OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)


def simulate_timeseries() -> list[dict]:
    fake_time = [0.0]
    clock = lambda: fake_time[0]

    tracker = DemandTracker(window_seconds=DEMAND_WINDOW, clock=clock)
    scaler = CostAwareAutoscaler(
        configs=CONFIGS, cooldown_seconds=COOLDOWN, clock=clock,
    )
    scaler.demand_tracker = tracker
    current_config = CONFIGS[0]
    scaler.current_config[MODEL] = current_config
    scaler.last_scale_time[MODEL] = 0.0

    samples = []
    total_elapsed = 0.0

    for phase_name, duration, concurrency, rpm in PHASES:
        worker_interval = 60.0 * concurrency / rpm if concurrency > 0 and rpm > 0 else 0
        workers = [{"next_start": total_elapsed, "busy_until": total_elapsed}
                   for _ in range(concurrency)]

        t = total_elapsed
        end_t = total_elapsed + duration

        while t < end_t:
            for w in workers:
                if t >= w["busy_until"] and t >= w["next_start"]:
                    config_id = current_config.config_id()
                    dur = BASE_DURATION.get(config_id, 15.0)
                    w["busy_until"] = t + dur
                    w["next_start"] = t + worker_interval
                    completion_time = t + dur
                    if completion_time < end_t + 60:
                        saved = fake_time[0]
                        fake_time[0] = completion_time
                        tracker.record_tokens(MODEL, TOKENS_PER_REQUEST)
                        fake_time[0] = saved

            if int(t) % 5 == 0:
                fake_time[0] = t
                demand = tracker.get_demand(MODEL)
                new_config = scaler.check_scaling(MODEL)
                if new_config and new_config.config_id() != current_config.config_id():
                    current_config = new_config
                    scaler.current_config[MODEL] = new_config
                    scaler.last_scale_time[MODEL] = t

                samples.append({
                    "time_min": t / 60.0,
                    "phase": phase_name,
                    "config_id": current_config.config_id(),
                    "demand_tps": demand,
                    "hourly_cost": current_config.hourly_cost,
                })
            t += 1.0
        total_elapsed = end_t

    return samples


def _phase_spans(samples: list[dict]) -> list[tuple]:
    """Extract (start_min, end_min, phase_name) spans."""
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


def generate_plots(samples: list[dict]) -> None:
    t = np.array([s["time_min"] for s in samples])
    demand = np.array([s["demand_tps"] for s in samples])
    configs = [s["config_id"] for s in samples]
    costs = np.array([s["hourly_cost"] for s in samples])
    config_idx = np.array([CONFIG_ORDER.index(c) for c in configs])

    phase_spans = _phase_spans(samples)

    # Scaling event times
    scale_times = []
    for i in range(1, len(configs)):
        if configs[i] != configs[i - 1]:
            scale_times.append(t[i])

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # ------------------------------------------------------------------
    # Plot 1: Hardware config + demand vs time
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    # Phase backgrounds
    for s, e, p in phase_spans:
        ax1.axvspan(s, e, alpha=0.15, color=PHASE_COLORS.get(p, "#f5f5f5"), zorder=0)

    # Config as colored step fill
    for i in range(len(t) - 1):
        c = configs[i]
        ax1.fill_between([t[i], t[i + 1]], [config_idx[i], config_idx[i + 1]],
                         alpha=0.35, color=CONFIG_COLORS[c], step="post", zorder=2)
    ax1.step(t, config_idx, where="post", color="#333", linewidth=1.8, zorder=3,
             label="Active config")

    # Demand on secondary y-axis
    ax1b = ax1.twinx()
    ax1b.plot(t, demand, color="#e15759", linewidth=1.4, alpha=0.85, zorder=4,
              label="Demand (tok/s)")
    ax1b.fill_between(t, demand, alpha=0.08, color="#e15759")
    ax1b.set_ylabel("Demand (tok/s)", color="#e15759")
    ax1b.tick_params(axis="y", labelcolor="#e15759")
    ax1b.spines["right"].set_visible(True)

    # Threshold lines on demand axis
    thresholds = {"cpu_4": 12.0, "cpu_8": 18.0, "cpu_12": 22.0}
    for cid, thr in thresholds.items():
        ax1b.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                      alpha=0.6, linewidth=1)
        ax1b.text(t[-1] * 1.01, thr, f"{thr}", fontsize=8,
                  va="center", color=CONFIG_COLORS[cid])

    # Scaling event lines
    for st in scale_times:
        ax1.axvline(st, color="red", linestyle="--", alpha=0.35, linewidth=1, zorder=1)

    # Phase labels
    ymax_demand = max(demand) if max(demand) > 0 else 1
    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax1b.text(mid, ymax_demand * 1.08, p, ha="center", va="bottom",
                  fontsize=7.5, fontstyle="italic", color="#555")

    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Hardware Configuration")
    ax1.set_yticks(range(len(CONFIG_ORDER)))
    ax1.set_yticklabels(CONFIG_ORDER)
    ax1.set_ylim(-0.5, len(CONFIG_ORDER) - 0.5)
    ax1.set_title("Vertical Scaling Under Varying Workload (CPU Simulation)")

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.5)
               for c in CONFIG_ORDER]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=3)

    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(OUT_DIR / f"sim_config_vs_time.{ext}",
                     bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print(f"Saved: {OUT_DIR}/sim_config_vs_time.pdf/.png")

    # ------------------------------------------------------------------
    # Plot 2: Cost/hour vs time
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 4))

    # Phase backgrounds
    for s, e, p in phase_spans:
        ax2.axvspan(s, e, alpha=0.15, color=PHASE_COLORS.get(p, "#f5f5f5"), zorder=0)

    # Cost step
    for i in range(len(t) - 1):
        c = configs[i]
        ax2.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS[c], step="post", zorder=2)
    ax2.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3)

    # Demand overlay
    ax2b = ax2.twinx()
    ax2b.plot(t, demand, color="#e15759", linewidth=1.2, alpha=0.6, zorder=4,
              label="Demand (tok/s)")
    ax2b.set_ylabel("Demand (tok/s)", color="#e15759")
    ax2b.tick_params(axis="y", labelcolor="#e15759")
    ax2b.spines["right"].set_visible(True)

    # Scaling event lines
    for st in scale_times:
        ax2.axvline(st, color="red", linestyle="--", alpha=0.35, linewidth=1, zorder=1)

    # Phase labels
    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax2.text(mid, max(costs) * 1.15, p, ha="center", va="bottom",
                 fontsize=7.5, fontstyle="italic", color="#555")

    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Hourly Cost ($)")
    ax2.set_ylim(0, max(costs) * 1.35)
    ax2.set_title("Infrastructure Cost Over Time (CPU Simulation)")

    cost_legend = [
        mpatches.Patch(color=CONFIG_COLORS["cpu_4"], label="cpu_4 — $0.40/hr", alpha=0.5),
        mpatches.Patch(color=CONFIG_COLORS["cpu_8"], label="cpu_8 — $0.80/hr", alpha=0.5),
        mpatches.Patch(color=CONFIG_COLORS["cpu_12"], label="cpu_12 — $1.20/hr", alpha=0.5),
    ]
    ax2.legend(handles=cost_legend, loc="upper left", fontsize=8, ncol=3)

    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(OUT_DIR / f"sim_cost_vs_time.{ext}",
                     bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print(f"Saved: {OUT_DIR}/sim_cost_vs_time.pdf/.png")


if __name__ == "__main__":
    print("Running CPU scaling simulation...")
    samples = simulate_timeseries()
    print(f"Generated {len(samples)} samples")

    for cid in CONFIG_ORDER:
        pts = [s for s in samples if s["config_id"] == cid]
        if pts:
            d = [s["demand_tps"] for s in pts]
            print(f"  {cid}: {len(pts)} samples, demand {min(d):.1f}–{max(d):.1f} tok/s")

    generate_plots(samples)
    print("Done.")
