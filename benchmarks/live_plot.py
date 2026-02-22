#!/usr/bin/env python3
"""
Real-time plotting of the scaling demo benchmark.

Reads the latest log file from scaling_demo_logs/ and generates
a live-updating 5-panel plot (HTML with auto-refresh).

Panels:
  1. Hardware config + demand (dual axis)
  2. Throughput MA + Concurrency MA + per-request gen tok/s
  3. Demand signal (EMA demand from client STATUS)
  4. Hourly cost
  5. Cost per token

Usage:
    uv run python benchmarks/live_plot.py

Opens an HTML file that auto-refreshes every 10 seconds.
"""
from __future__ import annotations

import json
import os
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


def find_server_log(run_log: Path) -> Optional[Path]:
    """Find the server log that corresponds to a run log.

    run_20260208_003358.log -> server_20260208_003359.log (closest match)
    """
    # Extract timestamp from run log name
    stem = run_log.stem  # e.g. run_20260208_003358
    ts_part = stem.replace("run_", "")  # 20260208_003358

    # Try exact match first
    exact = run_log.parent / ("server_%s.log" % ts_part)
    if exact.exists():
        return exact

    # Find closest server log by timestamp (server starts ~1s after run)
    server_logs = sorted(run_log.parent.glob("server_*.log"))
    if not server_logs:
        return None

    # Pick the server log with the closest timestamp
    best = None
    best_diff = float("inf")
    for sl in server_logs:
        sl_ts = sl.stem.replace("server_", "")
        # Simple string comparison works since format is YYYYMMDD_HHMMSS
        diff = abs(int(sl_ts.replace("_", "")) - int(ts_part.replace("_", "")))
        if diff < best_diff:
            best_diff = diff
            best = sl

    return best


def parse_log(log_path: Path) -> Dict:
    """Parse STATUS and REQ_OK lines from the client log."""
    statuses = []
    requests = []
    scaling_events = []
    phases = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "[STATUS]" in line and "elapsed_s" in line:
                try:
                    json_str = line.split("[STATUS]")[1].strip()
                    data = json.loads(json_str)
                    statuses.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

            elif "[REQ_OK]" in line and "wall_ms" in line:
                try:
                    json_str = line.split("[REQ_OK]")[1].strip()
                    data = json.loads(json_str)
                    requests.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

            elif "[SCALE_EVENT]" in line:
                try:
                    json_str = line.split("[SCALE_EVENT]")[1].strip()
                    data = json.loads(json_str)
                    scaling_events.append(data)
                except (json.JSONDecodeError, IndexError):
                    pass

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


def parse_server_log(server_log: Path) -> List[Dict]:
    """Parse DEMAND_CHECK lines from the server log for throughput_ema/metrics_tps/active_requests."""
    demand_checks = []
    with open(server_log) as f:
        for line in f:
            line = line.strip()
            if "[DEMAND_CHECK]" not in line:
                continue
            try:
                json_str = line.split("[DEMAND_CHECK]")[1].strip()
                data = json.loads(json_str)
                demand_checks.append(data)
            except (json.JSONDecodeError, IndexError):
                pass
    return demand_checks


def generate_live_plot(data: Dict, server_metrics: List[Dict], out_path: Path) -> None:
    """Generate the 5-panel plot from parsed data."""
    statuses = data["statuses"]
    requests = data["requests"]

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

    # Try to get throughput_ema/active_requests from client STATUS lines
    thr_ema_client = [s.get("throughput_ema", 0) for s in statuses]
    con_client = [s.get("active_requests", 0) for s in statuses]
    has_client_ma = any(v > 0 for v in thr_ema_client) or any(v > 0 for v in con_client)

    # Server metrics (DEMAND_CHECK) — use elapsed field (seconds from server start)
    # We need to align server elapsed with client elapsed.
    # The server starts a few seconds before the client, so we compute an offset.
    # The first client STATUS elapsed_s and the first server DEMAND_CHECK elapsed
    # should be close. We use the difference as offset.
    srv_t = []
    srv_thr_ma = []
    srv_streaming_tps = []
    srv_predicted_gauge = []
    srv_con_ma = []
    if server_metrics:
        for dc in server_metrics:
            srv_t.append(dc.get("elapsed", 0) / 60)
            srv_thr_ma.append(dc.get("throughput_ema", 0))
            srv_streaming_tps.append(dc.get("streaming_tps", 0))
            srv_predicted_gauge.append(dc.get("predicted_tps_gauge", 0))
            srv_con_ma.append(dc.get("active_requests", 0))

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

    fig, axes_arr = plt.subplots(5, 1, figsize=(16, 16), sharex=True,
                                  gridspec_kw={"height_ratios": [2.5, 2, 1.5, 1.5, 1.5]})

    # Phase background shading + scaling event lines on all panels
    for s, e, p in phase_spans:
        color = PHASE_COLORS.get(p, "#f5f5f5")
        for ax in axes_arr:
            ax.axvspan(s, e, alpha=0.12, color=color, zorder=0)
    for st in scale_times:
        for ax in axes_arr:
            ax.axvline(st, color="red", linestyle="--", alpha=0.3,
                       linewidth=1, zorder=1)

    # --- Panel 1: Config + Demand ---
    ax1 = axes_arr[0]
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

    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax1b.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                         alpha=0.5, linewidth=1)
            ax1b.text(t[-1] * 1.01, thr, "%d" % thr, fontsize=7,
                      va="center", color=CONFIG_COLORS[cid])

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
    ax1.set_title("Cost-Aware Vertical Scaling - LIVE (%.1f min, %d reqs)" % (elapsed_min, total_reqs))

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.5)
               for c in CONFIG_ORDER]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=len(CONFIG_ORDER))

    # --- Panel 2: Throughput signals + Concurrency (from server DEMAND_CHECK) ---
    ax2 = axes_arr[1]

    # Raw aggregate streaming tok/s (orange) — real-time token counting
    if srv_t and srv_streaming_tps and any(v > 0 for v in srv_streaming_tps):
        ax2.plot(srv_t, srv_streaming_tps, color="#ff7f0e", linewidth=0.8, alpha=0.5,
                 zorder=4, label="streaming tok/s (raw)")

    # predicted_tokens_seconds gauge (blue, thin) — llama.cpp's running avg
    if srv_t and srv_predicted_gauge and any(v > 0 for v in srv_predicted_gauge):
        ax2.plot(srv_t, srv_predicted_gauge, color="#1f77b4", linewidth=1.0, alpha=0.7,
                 zorder=5, label="predicted_tokens_seconds (gauge)")

    # EMA line (green, thick) — the actual scaling signal
    if has_client_ma:
        ax2.plot(t, thr_ema_client, color="#2ca02c", linewidth=2.0, zorder=6,
                 label="throughput EMA")
    elif srv_t:
        ax2.plot(srv_t, srv_thr_ma, color="#2ca02c", linewidth=2.0, zorder=6,
                 label="throughput EMA")

    # Per-request generation tok/s scatter (small dots)
    if len(req_t) > 0:
        ax2.scatter(req_t, req_gen_tps, s=6, alpha=0.25, color="#e15759",
                    zorder=3, label="per-req gen tok/s")

    # tok/s per request line (streaming_tps / active_requests)
    if srv_t and srv_streaming_tps and srv_con_ma:
        srv_tps_per_req = [
            (stps / ar) if ar > 0 else 0.0
            for stps, ar in zip(srv_streaming_tps, srv_con_ma)
        ]
        if any(v > 0 for v in srv_tps_per_req):
            ax2.plot(srv_t, srv_tps_per_req, color="#d62728", linewidth=1.2, alpha=0.7,
                     zorder=5, label="tok/s per request")

    # Capacity lines
    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax2.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                        alpha=0.6, linewidth=1)
            ax2.text(t[-1] * 1.01, thr, "%s cap" % cid,
                     fontsize=7, va="center", color=CONFIG_COLORS[cid])

    # Concurrency on secondary axis
    ax2b = ax2.twinx()
    if has_client_ma:
        ax2b.plot(t, con_client, color="#9467bd", linewidth=1.4, alpha=0.8,
                  zorder=3, label="active requests")
    elif srv_t:
        ax2b.plot(srv_t, srv_con_ma, color="#9467bd", linewidth=1.4, alpha=0.8,
                  zorder=3, label="active requests")
    ax2b.set_ylabel("Concurrency (active reqs)", color="#9467bd")
    ax2b.tick_params(axis="y", labelcolor="#9467bd")
    ax2b.spines["right"].set_visible(True)

    ax2.set_ylabel("Throughput (tok/s)")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # --- Panel 3: Demand signal ---
    ax3 = axes_arr[2]
    ax3.plot(t, demand, color="#333", linewidth=1.5, zorder=3, label="demand (scaling signal)")
    ax3.fill_between(t, demand, alpha=0.12, color="#4e79a7", zorder=2)
    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax3.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                        alpha=0.6, linewidth=1)
    ax3.set_ylabel("Demand (tok/s)")
    ax3.legend(loc="upper left", fontsize=8)

    # --- Panel 4: Hourly cost ---
    ax4 = axes_arr[3]
    for i in range(len(t) - 1):
        c = configs[i]
        ax4.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax4.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3)

    static_cost = HOURLY_COSTS["gpu_100"]
    ax4.axhline(static_cost, linestyle="--", color="#b07aa1", alpha=0.6,
                linewidth=1.5, label="Static gpu_100 ($%.2f/hr)" % static_cost)
    ax4.set_ylabel("Hourly Cost ($)")
    ax4.set_ylim(0, static_cost * 1.4)
    ax4.legend(loc="upper left", fontsize=8)

    # --- Panel 5: Cost per token ---
    cpt = np.array([
        s.get("hourly_cost", 0) / (MEASURED_THROUGHPUT.get(s.get("config_id", ""), 1.0) * 3600)
        for s in statuses
    ]) * 1e6
    ax5 = axes_arr[4]
    for i in range(len(t) - 1):
        c = configs[i]
        ax5.fill_between([t[i], t[i + 1]], [cpt[i], cpt[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax5.step(t, cpt, where="post", color="#333", linewidth=1.8, zorder=3)
    ax5.set_ylabel("Cost/Token (us)")
    ax5.set_xlabel("Time (minutes)")

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

    server_log = find_server_log(log_path)

    png_path = OUT_DIR / "scaling_demo_live.png"
    html_path = OUT_DIR / "scaling_demo_live.html"

    print("Watching: %s" % log_path)
    if server_log:
        print("Server log: %s" % server_log)
    else:
        print("No server log found (throughput_ema/active_requests from client only)")
    print("Plot: %s" % png_path)
    print("HTML: %s" % html_path)
    print("Updating every 10 seconds. Ctrl+C to stop.\n")

    while True:
        try:
            data = parse_log(log_path)
            server_metrics = parse_server_log(server_log) if server_log else []
            n_status = len(data["statuses"])
            n_reqs = len(data["requests"])
            n_scale = len(data["scaling_events"])

            if n_status >= 2:
                generate_live_plot(data, server_metrics, png_path)
                generate_html_wrapper(png_path, html_path)

                latest = data["statuses"][-1]
                elapsed = latest.get("elapsed_s", 0)
                config = latest.get("config_id", "?")
                demand = latest.get("demand_tps", 0)
                phase = latest.get("phase", "?")

                srv_info = ""
                if server_metrics:
                    last_dc = server_metrics[-1]
                    srv_info = " ema=%.1f streaming=%.1f gauge=%.1f active=%d" % (
                        last_dc.get("throughput_ema", 0),
                        last_dc.get("streaming_tps", 0),
                        last_dc.get("predicted_tps_gauge", 0),
                        last_dc.get("active_requests", 0),
                    )

                print("[%.0fs] phase=%s config=%s demand=%.1f tps%s | %d samples, %d reqs, %d scale events"
                      % (elapsed, phase, config, demand, srv_info, n_status, n_reqs, n_scale),
                      flush=True)
            else:
                print("Waiting for data... (%d status lines)" % n_status, flush=True)

            time.sleep(10)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print("Error: %s" % e, flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(10)


if __name__ == "__main__":
    main()
