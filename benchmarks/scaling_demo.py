#!/usr/bin/env python3
"""
Scaling demo benchmark for thesis.

Starts a CPU-only scaling demo server (cpu_4 → cpu_8 → cpu_12),
sends real inference requests in phases with controlled request rates
to trigger vertical scaling up and down, collects metrics, and
generates a thesis-quality 3-panel plot.

Server config: cooldown=300s (5 min), demand_window=60s
Configs: cpu_4 (12 tok/s), cpu_8 (18 tok/s), cpu_12 (22 tok/s)

Rate control: Each phase specifies concurrency AND requests-per-minute
(rpm). Workers pace themselves so the aggregate rate matches the target.
Demand (tok/s) ≈ rpm * ~140 tokens / 60 ≈ rpm * 2.33.

Phases (~33 min total):
  1. Warm-up      (2 min):  1 worker, rpm=2   → ~5 tok/s   (cpu_4 holds)
  2. Medium load  (7 min):  2 workers, rpm=4  → ~13 tok/s  (>12 → scale to cpu_8)
  3. High load    (7 min):  3 workers, rpm=9  → ~28 tok/s  (>18 → scale to cpu_12)
  4. Sustain      (2 min):  3 workers, rpm=9  → stable on cpu_12
  5. Ramp-down 1  (7 min):  2 workers, rpm=4  → ~13 tok/s  (after cooldown → cpu_8)
  6. Ramp-down 2  (7 min):  1 worker, rpm=2   → ~5 tok/s   (after cooldown → cpu_4)
  7. Low load     (1 min):  1 worker, rpm=2   → confirm cpu_4

Usage:
    uv run python benchmarks/scaling_demo.py
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MAX_TOKENS = 128
SERVER_STARTUP_TIMEOUT = 180
OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class Sample:
    elapsed: float
    phase: str
    config_id: str
    demand_tps: float
    throughput_tps: float
    hourly_cost: float
    active_requests: int
    total_requests: int


@dataclass
class Collector:
    start_time: float = field(default_factory=time.time)
    samples: List[Sample] = field(default_factory=list)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


async def _poll_status(base_url: str) -> Optional[Dict]:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(f"{base_url}/status") as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None


async def _send_request(session: aiohttp.ClientSession, base_url: str) -> bool:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Explain vertical scaling in one paragraph."}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status == 200:
                await resp.json()
                return True
    except Exception:
        pass
    return False


def _record(collector: Collector, status: Dict, phase: str) -> None:
    model_info = status.get("models", {}).get(MODEL, {})
    if not model_info:
        return
    collector.samples.append(Sample(
        elapsed=time.time() - collector.start_time,
        phase=phase,
        config_id=model_info.get("config_id", "?"),
        demand_tps=model_info.get("demand_tps", 0),
        throughput_tps=model_info.get("throughput_tps", 0),
        hourly_cost=model_info.get("hourly_cost", 0),
        active_requests=model_info.get("active_requests", 0),
        total_requests=model_info.get("total_requests", 0),
    ))


async def _run_phase(
    base_url: str,
    collector: Collector,
    phase_name: str,
    duration: int,
    concurrency: int = 0,
    rpm: float = 0,
) -> None:
    """Run a benchmark phase with controlled request rate.

    Args:
        base_url: Server URL.
        collector: Metrics collector.
        phase_name: Name for logging/plotting.
        duration: Phase duration in seconds.
        concurrency: Number of parallel workers sending requests.
        rpm: Target aggregate requests per minute. Each worker paces
             itself to send at most rpm/concurrency requests per minute.
             If 0 and concurrency > 0, workers send as fast as possible.
    """
    if concurrency > 0 and rpm > 0:
        # Interval between request *starts* per worker
        worker_interval = 60.0 * concurrency / rpm
    else:
        worker_interval = 0.0

    log(f"PHASE: {phase_name} ({duration}s, workers={concurrency}, "
        f"rpm={rpm}, interval={worker_interval:.1f}s/worker)")

    deadline = time.time() + duration
    stop = asyncio.Event()

    async def worker(session: aiohttp.ClientSession) -> None:
        while not stop.is_set() and time.time() < deadline:
            req_start = time.time()
            await _send_request(session, base_url)
            # Pace: wait until at least worker_interval has passed since
            # this request started, so we control the *rate* not just the
            # gap after completion.
            if worker_interval > 0:
                elapsed_req = time.time() - req_start
                wait = max(0, worker_interval - elapsed_req)
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                await asyncio.sleep(0.1)

    async def monitor() -> None:
        while not stop.is_set() and time.time() < deadline:
            status = await _poll_status(base_url)
            if status:
                _record(collector, status, phase_name)
                mi = status.get("models", {}).get(MODEL, {})
                log(f"  [{phase_name}] config={mi.get('config_id')} "
                    f"demand={mi.get('demand_tps', 0):.1f} tok/s "
                    f"cost=${mi.get('hourly_cost', 0):.2f}/hr "
                    f"active={mi.get('active_requests', 0)}")
            await asyncio.sleep(5)

    tasks = [asyncio.create_task(monitor())]

    session = None
    if concurrency > 0:
        session = aiohttp.ClientSession()
        for _ in range(concurrency):
            tasks.append(asyncio.create_task(worker(session)))

    remaining = deadline - time.time()
    if remaining > 0:
        await asyncio.sleep(remaining)

    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if session:
        await session.close()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
CONFIG_COLORS = {
    "cpu_4": "#4e79a7",
    "cpu_8": "#59a14f",
    "cpu_12": "#f28e2b",
}
CONFIG_ORDER = ["cpu_4", "cpu_8", "cpu_12"]

PHASE_COLORS = {
    "warm-up": "#e8f4f8",
    "medium load": "#fff3e0",
    "high load": "#fce4ec",
    "sustain": "#fce4ec",
    "ramp-down 1": "#e8f5e9",
    "ramp-down 2": "#ede7f6",
    "low load": "#e8f4f8",
}


def generate_plot(collector: Collector) -> None:
    if not collector.samples:
        log("No samples, skipping plot")
        return

    elapsed = [s.elapsed for s in collector.samples]
    demand = [s.demand_tps for s in collector.samples]
    configs = [s.config_id for s in collector.samples]
    costs = [s.hourly_cost for s in collector.samples]
    phases = [s.phase for s in collector.samples]

    elapsed_min = [e / 60 for e in elapsed]
    config_idx = [CONFIG_ORDER.index(c) if c in CONFIG_ORDER else -1 for c in configs]

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.grid": True, "grid.alpha": 0.2,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1.2]})

    # Phase backgrounds
    prev_phase = phases[0]
    start_e = elapsed_min[0]
    phase_spans = []
    for i in range(1, len(phases)):
        if phases[i] != prev_phase or i == len(phases) - 1:
            end_e = elapsed_min[i]
            phase_spans.append((start_e, end_e, prev_phase))
            start_e = end_e
            prev_phase = phases[i]

    for s, e, p in phase_spans:
        color = PHASE_COLORS.get(p, "#f5f5f5")
        for ax in axes:
            ax.axvspan(s, e, alpha=0.2, color=color, zorder=0)

    # --- Panel 1: Demand ---
    ax1 = axes[0]
    ax1.plot(elapsed_min, demand, color="#333", linewidth=1.5, zorder=3)
    ax1.fill_between(elapsed_min, demand, alpha=0.12, color="#4e79a7", zorder=2)
    ax1.set_ylabel("Demand (tok/s)")
    ax1.set_title("Cost-Aware Autoscaler: Vertical Scaling Under Varying Workload")

    thresholds = {"cpu_4": 12, "cpu_8": 18, "cpu_12": 22}
    for label, thr in thresholds.items():
        ax1.axhline(thr, linestyle=":", color=CONFIG_COLORS.get(label, "gray"),
                     alpha=0.6, linewidth=1)
        ax1.text(elapsed_min[-1] * 1.01, thr, f"{label} cap",
                 fontsize=7, va="center", color=CONFIG_COLORS.get(label, "gray"))

    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ymax = max(demand) if max(demand) > 0 else 1
        ax1.text(mid, ymax * 1.08, p, ha="center", va="bottom",
                 fontsize=8, fontstyle="italic", color="#555")

    # --- Panel 2: Active config ---
    ax2 = axes[1]
    for i in range(len(elapsed_min) - 1):
        c = configs[i]
        color = CONFIG_COLORS.get(c, "gray")
        ax2.fill_between(
            [elapsed_min[i], elapsed_min[i + 1]],
            [config_idx[i], config_idx[i + 1]],
            alpha=0.4, color=color, step="post", zorder=2,
        )
    ax2.step(elapsed_min, config_idx, where="post", color="#333", linewidth=1.5, zorder=3)
    ax2.set_ylabel("Hardware Config")
    ax2.set_yticks(range(len(CONFIG_ORDER)))
    ax2.set_yticklabels(CONFIG_ORDER)
    ax2.set_ylim(-0.5, len(CONFIG_ORDER) - 0.5)

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.6)
               for c in CONFIG_ORDER]
    ax2.legend(handles=patches, loc="upper left", fontsize=8, ncol=3)

    # Scaling event lines
    for i in range(1, len(configs)):
        if configs[i] != configs[i - 1]:
            for ax in axes:
                ax.axvline(elapsed_min[i], color="red", linestyle="--",
                           alpha=0.4, linewidth=1, zorder=1)

    # --- Panel 3: Hourly cost ---
    ax3 = axes[2]
    ax3.fill_between(elapsed_min, costs, alpha=0.25, color="#59a14f", step="post")
    ax3.step(elapsed_min, costs, where="post", color="#59a14f", linewidth=1.5)
    ax3.set_ylabel("Hourly Cost ($)")
    ax3.set_xlabel("Time (minutes)")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"scaling_demo.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log(f"Plot saved to {OUT_DIR}/scaling_demo.pdf and .png")

    data_path = OUT_DIR / "scaling_demo_data.json"
    with open(data_path, "w") as f:
        json.dump([{
            "elapsed": s.elapsed, "phase": s.phase, "config_id": s.config_id,
            "demand_tps": s.demand_tps, "throughput_tps": s.throughput_tps,
            "hourly_cost": s.hourly_cost, "active_requests": s.active_requests,
            "total_requests": s.total_requests,
        } for s in collector.samples], f, indent=2)
    log(f"Raw data saved to {data_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())

    log(f"Starting scaling demo server on port {port}...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "benchmarks.scaling_demo_server:app",
         "--host", "0.0.0.0", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    healthy = False
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"{base_url}/health") as resp:
                    data = await resp.json()
                    if data.get("status") == "healthy":
                        healthy = True
                        break
        except Exception:
            pass
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            print(f"Server died:\n{stdout}")
            return
        await asyncio.sleep(3)

    if not healthy:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        proc.kill()
        proc.wait()
        print(f"Server not healthy in {SERVER_STARTUP_TIMEOUT}s:\n{stdout}")
        return

    log(f"Server healthy at {base_url}")
    collector = Collector(start_time=time.time())

    try:
        # Phase 1: warm-up — 1 worker, rpm=2 → ~5 tok/s on cpu_4 (2 min)
        await _run_phase(base_url, collector, "warm-up",
                         duration=120, concurrency=1, rpm=2)

        # Phase 2: medium load — 2 workers, rpm=4 → ~13 tok/s → scale to cpu_8 (7 min)
        await _run_phase(base_url, collector, "medium load",
                         duration=420, concurrency=2, rpm=4)

        # Phase 3: high load — 3 workers, rpm=9 → ~28 tok/s → scale to cpu_12 (7 min)
        await _run_phase(base_url, collector, "high load",
                         duration=420, concurrency=3, rpm=9)

        # Phase 4: sustain on cpu_12 (2 min)
        await _run_phase(base_url, collector, "sustain",
                         duration=120, concurrency=3, rpm=9)

        # Phase 5: ramp-down 1 — 2 workers, rpm=4 → ~13 tok/s
        # After cooldown (300s), demand < cpu_12 threshold → scale to cpu_8 (7 min)
        await _run_phase(base_url, collector, "ramp-down 1",
                         duration=420, concurrency=2, rpm=4)

        # Phase 6: ramp-down 2 — 1 worker, rpm=2 → ~5 tok/s
        # After cooldown (300s), demand < cpu_8 threshold → scale to cpu_4 (7 min)
        await _run_phase(base_url, collector, "ramp-down 2",
                         duration=420, concurrency=1, rpm=2)

        # Phase 7: low load — confirm stable on cpu_4 (1 min)
        await _run_phase(base_url, collector, "low load",
                         duration=60, concurrency=1, rpm=2)

    except KeyboardInterrupt:
        log("Interrupted")
    finally:
        log("Stopping server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        for name in [n.strip() for n in result.stdout.splitlines() if n.strip()]:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    log(f"Collected {len(collector.samples)} samples")
    generate_plot(collector)
    log("Done")


if __name__ == "__main__":
    asyncio.run(main())
