#!/usr/bin/env python3
"""
Real-time plotting of the scaling demo benchmark.

Reads the latest log file from scaling_demo_logs/ and generates
a live-updating 4-panel plot (HTML with auto-refresh).

Usage:
    uv run python benchmarks/live_plot.py

Opens an HTML file that auto-refreshes every 10 seconds.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Use non-interactive backend for file output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_DIR = Path(__file__).parent / "scaling_demo_logs"
OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

CONFIG_ORDER = ["cpu_4", "cpu_12", "gpu_25", "gpu_100"]
CONFIG_COLORS = {
    "cpu_4":   "#4e79a7",
    "cpu_12":  "#f28e2b",
    "gpu_25":  "#e15759",
    "gpu_100": "#b07aa1",
}
MEASURED_THROUGHPUT = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}
HOURLY_COSTS = {
    "cpu_4":   0.05,
    "cpu_12":  0.12,
    "gpu_25":  0.50,
    "gpu_100": 4.00,
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


def find_latest_log() -> Optional[Path]:
    if not LOG_DIR.exists():
        return None
    logs = sorted(LOG_DIR.glob("run_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_log(log_path: Path) -> Dict:
    """Parse STATUS and REQ_OK lines from the log."""
    statuses = []
    requests = []
    scaling_events = []
    phases = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse STATUS lines from client
            if "[STATUS]" in line and "elapsed_s" in line:
                try:
                    json_str = line.split("[STATUS]")[1].strip()
                    data = json.loads(json_str)
                    statuses.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

            # Parse REQ_OK lines from client
            elif "[REQ_OK]" in line and "wall_ms" in line:
                try:
                    json_str = line.split("[REQ_OK]")[1].strip()
                    data = json.loads(json_str)
                    requests.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

            # Parse SCALE_EVENT from client
            elif "[SCALE_EVENT]" in line:
                try:
                    json_str = line.split("[SCALE_EVENT]")[1].strip()
                    data = json.loads(json_str)
                    scaling_events.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

            # Parse PHASE_START
            elif "[PHASE_START]" in line:
                try:
                    json_str = line.split("[PHASE_START]")[1].strip()
                    data = json.loads(json_str)
                    phases.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

    return {
        "statuses": statuses,
        "requests": requests,
        "scaling_events": scaling_events,
        "phases": phases,
    }


def generate_live_plot(data: Dict, out_path: Path) -> None:
    """Generate the 4-panel plot from parsed data."""
    statuses = data["statuses"]
    requests = data["requests"]
    scaling_events = data["scaling_events"]

    if len(statuses) < 2:
        print("Not enough data yet (%d status samples)" % len(statuses))
        return

    t = np.array([s["elapsed_s"] / 60 for s in statuses])
    demand = np.array([s.get("demand_tps", 0) for s in statuses])
    configs = [s.get("config_id", "?") for s in statuses]
    costs = np.array([s.get("hourly_cost", 0) for s in statuses])
    config_idx = np.array([
        CONFIG_ORDER.index(c) if c in CONFIG_ORDER else -1
        for c in configs
    ])
    phases_list = [s.get("phase", "") for s in statuses]

    # Phase spans
    phase_spans = []
    prev_phase = phases_list[0]
    start = t[0]
    for i in range(1, len(phases_list)):
        if phases_list[i] != prev_phase:
            phase_spans.append((start, t[i], prev_phase))
            start = t[i]
            prev_phase = phases_list[i]
    phase_spans.append((start, t[-1], prev_phase))

    # Scaling event times
    scale_times = []
    for i in range(1, len(configs)):
        if configs[i] != configs[i - 1]:
            scale_times.append(t[i])

    # Request data
    req_t = np.array([r.get("elapsed_s", 0) / 60 for r in requests]) if requests else np.array([])
    req_gen_tps = np.array([r.get("generation_tps", 0) for r in requests]) if requests else np.array([])

    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 2, 1.5, 1.5]})

    # Phase background shading
    for s, e, p in phase_spans:
        color = PHASE_COLORS.get(p, "#f5f5f5")
        for ax in axes:
            ax.axvspan(s, e, alpha=0.12, color=color, zorder=0)

    # Scaling event lines
    for st in scale_times:
        for ax in axes:
            ax.axvline(st, color="red", linestyle="--", alpha=0.3,
                       linewidth=1, zorder=1)

    # --- Panel 1: Config + Demand ---
    ax1 = axes[0]
    for i in range(len(t) - 1):
        c = configs[i]
        ax1.fill_between([t[i], t[i + 1]], [config_idx[i], config_idx[i + 1]],
                         alpha=0.35, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax1.step(t, config_idx, where="post", color="#333", linewidth=1.8, zorder=3)

    ax1b = ax1.twinx()
    ax1b.plot(t, demand, color="#e15759", linewidth=1.4, alpha=0.85, zorder=4)
    ax1b.fill_between(t, demand, alpha=0.06, color="#e15759")
    ax1b.set_ylabel("Demand (tok/s)", color="#e15759")
    ax1b.tick_params(axis="y", labelcolor="#e15759")
    ax1b.spines["right"].set_visible(True)

    # Throughput capacity lines
    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax1b.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                         alpha=0.5, linewidth=1)
            ax1b.text(t[-1] * 1.01, thr, "%d" % thr, fontsize=7,
                      va="center", color=CONFIG_COLORS[cid])

    # Phase labels
    ymax_demand = max(demand) if len(demand) > 0 and max(demand) > 0 else 1
    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax1b.text(mid, ymax_demand * 1.12, p, ha="center", va="bottom",
                  fontsize=7, fontstyle="italic", color="#555")

    ax1.set_ylabel("Hardware Config")
    ax1.set_yticks(range(len(CONFIG_ORDER)))
    ax1.set_yticklabels(CONFIG_ORDER)
    ax1.set_ylim(-0.5, len(CONFIG_ORDER) - 0.5)

    elapsed_min = t[-1] if len(t) > 0 else 0
    total_reqs = len(requests)
    ok_reqs = sum(1 for r in requests if r.get("generation_tps", 0) > 0)
    ax1.set_title("Cost-Aware Vertical Scaling - LIVE (%.1f min, %d reqs)" % (elapsed_min, total_reqs))

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.5)
               for c in CONFIG_ORDER]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=len(CONFIG_ORDER))

    # --- Panel 2: Demand + Generation TPS scatter ---
    ax2 = axes[1]
    ax2.plot(t, demand, color="#333", linewidth=1.5, zorder=3, label="EMA demand")
    ax2.fill_between(t, demand, alpha=0.12, color="#4e79a7", zorder=2)

    if len(req_t) > 0:
        ax2.scatter(req_t, req_gen_tps, s=12, alpha=0.5, color="#e15759",
                    zorder=4, label="gen tok/s per request")

    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax2.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                        alpha=0.6, linewidth=1)
            ax2.text(t[-1] * 1.01, thr, "%s cap" % cid,
                     fontsize=7, va="center", color=CONFIG_COLORS[cid])

    ax2.set_ylabel("Tokens/sec")
    ax2.legend(loc="upper left", fontsize=8)

    # --- Panel 3: Hourly cost ---
    ax3 = axes[2]
    for i in range(len(t) - 1):
        c = configs[i]
        ax3.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax3.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3)

    static_cost = HOURLY_COSTS["gpu_100"]
    ax3.axhline(static_cost, linestyle="--", color="#b07aa1", alpha=0.6,
                linewidth=1.5, label="Static gpu_100 ($%.2f/hr)" % static_cost)
    ax3.set_ylabel("Hourly Cost ($)")
    ax3.set_ylim(0, static_cost * 1.4)
    ax3.legend(loc="upper left", fontsize=8)

    # --- Panel 4: Cost per token ---
    cpt = np.array([
        s.get("hourly_cost", 0) / (MEASURED_THROUGHPUT.get(s.get("config_id", ""), 1.0) * 3600)
        for s in statuses
    ]) * 1e6
    ax4 = axes[3]
    for i in range(len(t) - 1):
        c = configs[i]
        ax4.fill_between([t[i], t[i + 1]], [cpt[i], cpt[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax4.step(t, cpt, where="post", color="#333", linewidth=1.8, zorder=3)
    ax4.set_ylabel("Cost/Token (us)")
    ax4.set_xlabel("Time (minutes)")

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_html_wrapper(png_path: Path, html_path: Path, refresh_seconds: int = 10) -> None:
    """Generate an HTML file that auto-refreshes and shows the plot."""
    rel_png = os.path.relpath(png_path, html_path.parent)
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Scaling Demo - Live Plot</title>
    <meta http-equiv="refresh" content="%d">
    <style>
        body {
            margin: 0; padding: 20px;
            background: #1a1a2e; color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        h1 { color: #e94560; margin-bottom: 5px; }
        .meta { color: #888; font-size: 14px; margin-bottom: 15px; }
        img {
            max-width: 100%%; border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .footer { color: #555; font-size: 12px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Scaling Demo - Live</h1>
    <div class="meta">Auto-refreshes every %d seconds. Last update: <span id="ts"></span></div>
    <img src="%s?t=%d" alt="Live Plot">
    <div class="footer">
        Expected: cpu_4 → cpu_12 → gpu_25 → gpu_100 → gpu_25 → cpu_12 → cpu_4
    </div>
    <script>document.getElementById('ts').textContent = new Date().toLocaleTimeString();</script>
</body>
</html>""" % (refresh_seconds, refresh_seconds, rel_png, int(time.time()))

    with open(html_path, "w") as f:
        f.write(html)


def main() -> None:
    log_path = find_latest_log()
    if not log_path:
        print("No log files found in %s" % LOG_DIR)
        sys.exit(1)

    png_path = OUT_DIR / "scaling_demo_live.png"
    html_path = OUT_DIR / "scaling_demo_live.html"

    print("Watching: %s" % log_path)
    print("Plot: %s" % png_path)
    print("HTML: %s" % html_path)
    print("Updating every 10 seconds. Ctrl+C to stop.\n")

    while True:
        try:
            data = parse_log(log_path)
            n_status = len(data["statuses"])
            n_reqs = len(data["requests"])
            n_scale = len(data["scaling_events"])

            if n_status >= 2:
                generate_live_plot(data, png_path)
                generate_html_wrapper(png_path, html_path)

                latest = data["statuses"][-1]
                elapsed = latest.get("elapsed_s", 0)
                config = latest.get("config_id", "?")
                demand = latest.get("demand_tps", 0)
                phase = latest.get("phase", "?")

                print("[%.0fs] phase=%s config=%s demand=%.1f tps | %d samples, %d reqs, %d scale events"
                      % (elapsed, phase, config, demand, n_status, n_reqs, n_scale),
                      flush=True)
            else:
                print("Waiting for data... (%d status lines)" % n_status, flush=True)

            time.sleep(10)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print("Error: %s" % e, flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
