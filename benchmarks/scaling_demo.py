#!/usr/bin/env python3
"""
5-config scaling demo benchmark for thesis.

Starts the main cost-aware server and drives it through the full
vertical scaling staircase:
  cpu_4 → cpu_16 → cpu_48 → gpu_25 → gpu_100 → gpu_25 → cpu_48 → cpu_16 → cpu_4

Server config: cooldown=60s (up), cooldown_down=240s (down), EMA ~30s window, MIN_TPS=10.0, SCALE_DOWN_CONCURRENCY=5.0

Configs (measured throughput from benchmark):
  cpu_4:    92.2 tok/s  ($0.05/hr)  — peak at batch=32
  cpu_16:  291.2 tok/s  ($0.15/hr)  — peak at batch=32
  cpu_48:  446.2 tok/s  ($0.45/hr)  — peak at batch=64
  gpu_25:  335.8 tok/s  ($0.50/hr)  — peak at batch=16
  gpu_100: 1573.9 tok/s ($4.00/hr)  — peak at batch=64

Scaling signal: per-request tok/s EMA.
  Scale UP:   per_request_tps_ema < MIN_TPS (10.0)
  Scale DOWN: per_request_tps_ema >= MIN_TPS AND active_requests_ema <= 5.0

Per-request tok/s ≈ capacity / concurrency.
  To overwhelm config X: need concurrency > capacity_X / MIN_TPS

Phase design (3 min each, 27 min total):
  Each scale-up phase uses enough concurrency to overwhelm the current config
  but not the next one, so it scales up exactly one step and stabilizes.
  Each scale-down phase uses low enough concurrency that per-request tok/s
  is well above threshold on the lower config.

Usage:
    uv run python benchmarks/scaling_demo.py 2>&1 | tee benchmarks/scaling_demo_logs/run.log
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MAX_TOKENS = 256
SERVER_STARTUP_TIMEOUT = 180
OUT_DIR = Path(__file__).parent / "thesis_figures"
OUT_DIR.mkdir(exist_ok=True)

# Load configs from hardware_configs.json
_CONFIG_PATH = Path(__file__).parent.parent / "hardware_configs.json"


def _load_from_json() -> tuple:
    with open(_CONFIG_PATH) as f:
        data = json.load(f)
    config_order = [c["config_id"] for c in data["configs"]]
    hourly_costs = {c["config_id"]: c["hourly_cost"] for c in data["configs"]}
    measured_throughput = data.get("measured_throughput", {})
    return config_order, hourly_costs, measured_throughput


CONFIG_ORDER, HOURLY_COSTS, MEASURED_THROUGHPUT = _load_from_json()

CONFIG_COLORS = {
    "cpu_4":   "#4e79a7",
    "cpu_16":  "#59a14f",
    "cpu_48":  "#f28e2b",
    "gpu_25":  "#e15759",
    "gpu_100": "#b07aa1",
}

PHASE_COLORS = {
    "idle":         "#f0f0f0",
    "low load":     "#e8f4f8",
    "medium load":  "#fff3e0",
    "high load":    "#fce4ec",
    "very high":    "#f8d7da",
    "peak load":    "#f3e5f5",
    "ramp-down 1":  "#e8f5e9",
    "ramp-down 2":  "#e0f2f1",
    "ramp-down 3":  "#ede7f6",
    "ramp-down 4":  "#fafafa",
}


# ---------------------------------------------------------------------------
# Phase design
# ---------------------------------------------------------------------------
# Per-request tok/s ≈ capacity / concurrency
# Scale-up triggers when per_request_tps < 10.0
# Concurrency to overwhelm each config:
#   cpu_4  (92.2):  >9.2  → use 12 workers
#   cpu_16 (291.2): >29.1 → use 35 workers
#   gpu_25 (335.8): >33.6 → use 35 workers (already >33.6 from prev phase)
#   cpu_48 (446.2): >44.6 → use 50 workers
#   gpu_100 (1573.9): not overwhelmed
#
# Throughput-sorted order: cpu_4 → cpu_16 → gpu_25 → cpu_48 → gpu_100
#
# Scale-down triggers when per_request_tps >= 10.0
#   AND lower config can serve current concurrency above threshold (1.5x margin)
# For scale-down we use 6 workers (phases 5-6), 8 workers (phase 7),
# and 1 worker at 2 rpm (phase 8).
#
# Viability checks for each step down:
#   gpu_100 → cpu_48: 446.2/6 = 74.4 >= 15 ✓  (6 workers)
#   cpu_48  → gpu_25: 335.8/6 = 56.0 >= 15 ✓  (6 workers)
#   gpu_25  → cpu_16: 291.2/8 = 36.4 >= 15 ✓  (8 workers)
#   cpu_16  → cpu_4:  92.2/8  = 11.5 <  15 ✗  (8 workers — blocked!)
#   cpu_16  → cpu_4:  92.2/1  = 92.2 >= 15 ✓  (1 worker, rate-limited)
#
# For the scale-up phases, we use 4 min (240s) to allow time for:
#   - Initial detection (~10-30s)
#   - Container swap (~30-60s)
#   - Cooldown (60s)
#   - Second transition if needed
#   - Stabilization on target config
#
# For scale-down, we use 6 min (360s) to allow time for:
#   - Cooldown (240s)
#   - Container swap (~30-60s)
#   - EMA convergence on new tier

PHASES = [
    # Phase 1: Low load — stay on cpu_4
    # 1 worker, rate-limited to ~3 rpm → per-req tok/s ≈ 92.2/1 = 92.2 (well above 10)
    ("low load",       240,   1,   3),

    # Phase 2: Medium load — overwhelm cpu_4, scale to cpu_16
    # 12 workers back-to-back → cpu_4: 92.2/12 ≈ 7.7 tok/s < 10 → scale up
    # cpu_16: 291.2/12 ≈ 24.3 tok/s > 10 → stable
    ("medium load",    240,  12,   0),

    # Phase 3: High load — overwhelm cpu_16 and gpu_25, scale to cpu_48
    # 35 workers back-to-back → cpu_16: 291.2/35 ≈ 8.3 tok/s < 10 → scale up
    # gpu_25: 335.8/35 ≈ 9.6 tok/s < 10 → after cooldown, scale up again
    # cpu_48: 446.2/35 ≈ 12.7 tok/s > 10 → stable
    ("high load",      300,  35,   0),

    # Phase 4: Very high load — overwhelm cpu_48, scale to gpu_100
    # 55 workers back-to-back → cpu_48: 446.2/55 ≈ 8.1 tok/s < 10 → scale up
    # gpu_100: 1573.9/55 ≈ 28.6 tok/s > 10 → stable
    ("peak load",      300,  55,   0),

    # Phase 5: Ramp-down — scale from gpu_100 to cpu_48
    # 6 workers back-to-back → gpu_100: ~1201 tok/s, per-req ~200 >> 10
    # Viability: cpu_48 446.2/6 = 74.4 >= 15 → scale down
    ("ramp-down 1",    360,   6,   0),

    # Phase 6: Ramp-down — scale from cpu_48 to gpu_25
    # 6 workers back-to-back → cpu_48: ~303 tok/s, per-req ~50 >> 10
    # Viability: gpu_25 335.8/6 = 56.0 >= 15 → scale down
    ("ramp-down 2",    360,   6,   0),

    # Phase 7: Ramp-down — scale from gpu_25 to cpu_16
    # 8 workers back-to-back → gpu_25: ~335 tok/s, per-req ~42 >> 10
    # Viability: cpu_16 291.2/8 = 36.4 >= 15 → scale down ✓
    # Viability: cpu_4  92.2/8  = 11.5 <  15 → blocked ✗ (prevents premature drop)
    ("ramp-down 3",    360,   8,   0),

    # Phase 8: Ramp-down — scale from cpu_16 to cpu_4
    # 1 worker at 2 rpm → cpu_16: per-req ~63 tok/s >> 10
    # Viability: cpu_4 92.2/1 = 92.2 >= 15 → scale down
    # NOTE: Using rate-limited requests (not back-to-back) to keep cpu_4 stable.
    # Back-to-back workers cause per-req EMA to dip near threshold during
    # container swap, triggering a re-scale-up.
    # 2 rpm (30s interval) gives ~11s window where recently_active=false,
    # enough for the 10s scaling check to trigger scale-down.
    ("ramp-down 4",    360,   1,   2),
]

EXPECTED_SEQUENCE = [
    "cpu_4", "cpu_16", "gpu_25", "cpu_48", "gpu_100",
    "cpu_48", "gpu_25", "cpu_16", "cpu_4",
]


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
class RequestResult:
    elapsed: float
    phase: str
    success: bool
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    prompt_eval_ms: float
    generation_ms: float
    prompt_tps: float
    generation_tps: float
    config_id: str
    error: str = ""


@dataclass
class Collector:
    start_time: float = field(default_factory=time.time)
    samples: List[Sample] = field(default_factory=list)
    requests: List[RequestResult] = field(default_factory=list)
    scaling_events: List[Dict] = field(default_factory=list)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def log_json(tag: str, data: Dict) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{tag}] {json.dumps(data)}", flush=True)


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


async def _send_request(
    session: aiohttp.ClientSession, base_url: str, collector: Collector, phase: str
) -> bool:
    """Send a request and log every detail of the response."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Explain vertical scaling in one paragraph."}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    req_start = time.time()
    elapsed = req_start - collector.start_time

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            raw = await resp.json()
            req_end = time.time()
            wall_ms = (req_end - req_start) * 1000

            if resp.status != 200:
                rr = RequestResult(
                    elapsed=elapsed, phase=phase, success=False,
                    prompt_tokens=0, completion_tokens=0,
                    wall_ms=wall_ms, prompt_eval_ms=0, generation_ms=0,
                    prompt_tps=0, generation_tps=0, config_id="?",
                    error=f"HTTP {resp.status}: {raw}",
                )
                collector.requests.append(rr)
                log_json("REQ_FAIL", {
                    "phase": phase, "elapsed": round(elapsed, 1),
                    "status": resp.status, "wall_ms": round(wall_ms, 1),
                    "error": str(raw)[:200],
                })
                return False

            usage = raw.get("usage", {})
            timings = raw.get("timings", {})

            prompt_tokens = timings.get("prompt_n", usage.get("prompt_tokens", 0))
            completion_tokens = timings.get("predicted_n", usage.get("completion_tokens", 0))
            prompt_ms = timings.get("prompt_ms", 0)
            predicted_ms = timings.get("predicted_ms", 0)
            prompt_per_second = timings.get("prompt_per_second", 0)
            predicted_per_second = timings.get("predicted_per_second", 0)
            prompt_per_token_ms = timings.get("prompt_per_token_ms", 0)
            predicted_per_token_ms = timings.get("predicted_per_token_ms", 0)

            rr = RequestResult(
                elapsed=elapsed, phase=phase, success=True,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                wall_ms=wall_ms, prompt_eval_ms=prompt_ms, generation_ms=predicted_ms,
                prompt_tps=prompt_per_second, generation_tps=predicted_per_second,
                config_id="",
            )
            collector.requests.append(rr)

            log_json("REQ_OK", {
                "phase": phase,
                "elapsed_s": round(elapsed, 1),
                "wall_ms": round(wall_ms, 1),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_eval_ms": round(prompt_ms, 1),
                "generation_ms": round(predicted_ms, 1),
                "prompt_ms_per_token": round(prompt_per_token_ms, 3),
                "generation_ms_per_token": round(predicted_per_token_ms, 3),
                "prompt_tps": round(prompt_per_second, 1),
                "generation_tps": round(predicted_per_second, 1),
                "ttft_ms": round(prompt_ms, 1),
                "raw_usage": usage,
                "raw_timings": timings,
            })
            return True

    except Exception as e:
        req_end = time.time()
        wall_ms = (req_end - req_start) * 1000
        rr = RequestResult(
            elapsed=elapsed, phase=phase, success=False,
            prompt_tokens=0, completion_tokens=0,
            wall_ms=wall_ms, prompt_eval_ms=0, generation_ms=0,
            prompt_tps=0, generation_tps=0, config_id="?",
            error=str(e),
        )
        collector.requests.append(rr)
        log_json("REQ_ERR", {
            "phase": phase, "elapsed": round(elapsed, 1),
            "wall_ms": round(wall_ms, 1), "error": str(e)[:200],
        })
        return False


def _record(collector: Collector, status: Dict, phase: str) -> None:
    model_info = status.get("models", {}).get(MODEL, {})
    if not model_info:
        return

    config_id = model_info.get("config_id", "?")
    demand = model_info.get("demand_tps", 0)
    throughput = model_info.get("throughput_ema", model_info.get("throughput_tps", 0))
    cost = model_info.get("hourly_cost", 0)
    active = model_info.get("active_requests", 0)
    total = model_info.get("total_requests", 0)

    sample = Sample(
        elapsed=time.time() - collector.start_time,
        phase=phase,
        config_id=config_id,
        demand_tps=demand,
        throughput_tps=throughput,
        hourly_cost=cost,
        active_requests=active,
        total_requests=total,
    )
    collector.samples.append(sample)

    # Detect scaling events
    if len(collector.samples) >= 2:
        prev = collector.samples[-2]
        if prev.config_id != config_id:
            event = {
                "elapsed": round(sample.elapsed, 1),
                "phase": phase,
                "from": prev.config_id,
                "to": config_id,
                "demand_tps": round(demand, 2),
                "from_cost": HOURLY_COSTS.get(prev.config_id, 0),
                "to_cost": HOURLY_COSTS.get(config_id, 0),
                "from_throughput": MEASURED_THROUGHPUT.get(prev.config_id, 0),
                "to_throughput": MEASURED_THROUGHPUT.get(config_id, 0),
            }
            collector.scaling_events.append(event)
            log_json("SCALE_EVENT", event)

    log_json("STATUS", {
        "phase": phase,
        "elapsed_s": round(sample.elapsed, 1),
        "config_id": config_id,
        "demand_tps": round(demand, 4),
        "throughput_ema": throughput,
        "hourly_cost": cost,
        "active_requests": active,
        "total_requests": total,
        "cost_per_token": model_info.get("cost_per_token", 0),
        "cpu_cores": model_info.get("cpu_cores"),
        "gpu_percentage": model_info.get("gpu_percentage"),
        "memory": model_info.get("memory"),
        "port": model_info.get("port"),
        "is_ready": model_info.get("is_ready"),
        "capacity": model_info.get("capacity", 0),
        "per_request_tps_ema": model_info.get("per_request_tps_ema", 0),
        "active_requests_ema": model_info.get("active_requests_ema", 0),
        "server_uptime": status.get("server_uptime_seconds", 0),
    })


async def _run_phase(
    base_url: str,
    collector: Collector,
    phase_name: str,
    duration: int,
    concurrency: int = 0,
    rpm: float = 0,
) -> None:
    """Run a benchmark phase with controlled request rate."""
    if concurrency > 0 and rpm > 0:
        worker_interval = 60.0 * concurrency / rpm
    else:
        worker_interval = 0.0

    log_json("PHASE_START", {
        "phase": phase_name,
        "duration_s": duration,
        "concurrency": concurrency,
        "rpm": rpm,
        "worker_interval_s": round(worker_interval, 2),
        "elapsed_s": round(time.time() - collector.start_time, 1),
    })

    deadline = time.time() + duration
    stop = asyncio.Event()

    async def worker(session: aiohttp.ClientSession) -> None:
        while not stop.is_set() and time.time() < deadline:
            req_start = time.time()
            await _send_request(session, base_url, collector, phase_name)
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

    # Phase summary
    phase_reqs = [r for r in collector.requests if r.phase == phase_name]
    ok = sum(1 for r in phase_reqs if r.success)
    fail = sum(1 for r in phase_reqs if not r.success)
    avg_wall = (sum(r.wall_ms for r in phase_reqs if r.success) / ok) if ok else 0
    avg_gen_tps = (sum(r.generation_tps for r in phase_reqs if r.success) / ok) if ok else 0
    total_prompt = sum(r.prompt_tokens for r in phase_reqs if r.success)
    total_completion = sum(r.completion_tokens for r in phase_reqs if r.success)

    log_json("PHASE_END", {
        "phase": phase_name,
        "elapsed_s": round(time.time() - collector.start_time, 1),
        "requests_ok": ok,
        "requests_fail": fail,
        "avg_wall_ms": round(avg_wall, 1),
        "avg_generation_tps": round(avg_gen_tps, 1),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    })


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _phase_spans(samples: List[Sample]) -> List[tuple]:
    spans = []
    prev = samples[0].phase
    start = samples[0].elapsed / 60
    for s in samples[1:]:
        if s.phase != prev:
            spans.append((start, s.elapsed / 60, prev))
            start = s.elapsed / 60
            prev = s.phase
    spans.append((start, samples[-1].elapsed / 60, prev))
    return spans


def generate_plot(collector: Collector) -> None:
    if not collector.samples:
        log("No samples, skipping plot")
        return

    samples = collector.samples
    t = np.array([s.elapsed / 60 for s in samples])
    demand = np.array([s.demand_tps for s in samples])
    configs = [s.config_id for s in samples]
    costs = np.array([s.hourly_cost for s in samples])
    active = np.array([s.active_requests for s in samples])
    config_idx = np.array([CONFIG_ORDER.index(c) if c in CONFIG_ORDER else -1
                           for c in configs])

    phase_spans = _phase_spans(samples)

    scale_times = []
    scale_labels = []
    for i in range(1, len(configs)):
        if configs[i] != configs[i - 1]:
            scale_times.append(t[i])
            scale_labels.append(f"{configs[i-1]}→{configs[i]}")

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 2, 1.5, 1.5]})

    # Phase background shading
    for s, e, p in phase_spans:
        color = PHASE_COLORS.get(p, "#f5f5f5")
        for ax in axes:
            ax.axvspan(s, e, alpha=0.12, color=color, zorder=0)

    # Scaling event vertical lines
    for st in scale_times:
        for ax in axes:
            ax.axvline(st, color="red", linestyle="--", alpha=0.3,
                       linewidth=1, zorder=1)

    # Panel 1: Config + Demand
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
        if thr < 500:
            ax1b.axhline(thr, linestyle=":", color=CONFIG_COLORS.get(cid, "gray"),
                         alpha=0.5, linewidth=1)
            ax1b.text(t[-1] * 1.01, thr, f"{thr:.0f}", fontsize=7,
                      va="center", color=CONFIG_COLORS.get(cid, "gray"))

    ymax_demand = max(demand) if len(demand) > 0 and max(demand) > 0 else 1
    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax1b.text(mid, ymax_demand * 1.08, p, ha="center", va="bottom",
                  fontsize=7, fontstyle="italic", color="#555")

    ax1.set_ylabel("Hardware Configuration")
    ax1.set_yticks(range(len(CONFIG_ORDER)))
    ax1.set_yticklabels(CONFIG_ORDER)
    ax1.set_ylim(-0.5, len(CONFIG_ORDER) - 0.5)
    ax1.set_title("Cost-Aware Vertical Scaling — Real Hardware Demo (5 Configs)")

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.5)
               for c in CONFIG_ORDER]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=len(CONFIG_ORDER))

    # Panel 2: Active requests (concurrency)
    ax2 = axes[1]
    ax2.plot(t, active, color="#333", linewidth=1.2, zorder=3, alpha=0.8)
    ax2.fill_between(t, active, alpha=0.15, color="#4e79a7", zorder=2)
    ax2.set_ylabel("Active Requests")
    ax2.axhline(5.0, linestyle="--", color="#999", alpha=0.5, linewidth=1,
                label="Scale-down threshold (5)")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Hourly cost
    ax3 = axes[2]
    for i in range(len(t) - 1):
        c = configs[i]
        ax3.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax3.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3)

    static_cost = HOURLY_COSTS.get("gpu_100", 4.0)
    ax3.axhline(static_cost, linestyle="--", color="#b07aa1", alpha=0.6,
                linewidth=1.5, label=f"Static gpu_100 (${static_cost:.2f}/hr)")
    ax3.set_ylabel("Hourly Cost ($)")
    ax3.set_ylim(0, static_cost * 1.4)
    ax3.legend(loc="upper left", fontsize=8)

    # Panel 4: Cost per token
    cpt = np.array([
        s.hourly_cost / (MEASURED_THROUGHPUT.get(s.config_id, 1.0) * 3600)
        for s in samples
    ]) * 1e6
    ax4 = axes[3]
    for i in range(len(t) - 1):
        c = configs[i]
        ax4.fill_between([t[i], t[i + 1]], [cpt[i], cpt[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax4.step(t, cpt, where="post", color="#333", linewidth=1.8, zorder=3)
    ax4.set_ylabel("Cost/Token (μ$)")
    ax4.set_xlabel("Time (minutes)")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"scaling_demo.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    log(f"Plot saved to {OUT_DIR}/scaling_demo.pdf and .png")

    # Save raw data
    data_path = OUT_DIR / "scaling_demo_data.json"
    with open(data_path, "w") as f:
        json.dump([{
            "elapsed": s.elapsed, "phase": s.phase, "config_id": s.config_id,
            "demand_tps": s.demand_tps, "throughput_tps": s.throughput_tps,
            "hourly_cost": s.hourly_cost, "active_requests": s.active_requests,
            "total_requests": s.total_requests,
        } for s in collector.samples], f, indent=2)
    log(f"Raw data saved to {data_path}")

    # Check scaling sequence
    seen = [samples[0].config_id]
    for i in range(1, len(samples)):
        if samples[i].config_id != samples[i - 1].config_id:
            seen.append(samples[i].config_id)

    log(f"Observed sequence: {' → '.join(seen)}")
    log(f"Expected sequence: {' → '.join(EXPECTED_SEQUENCE)}")
    if seen == EXPECTED_SEQUENCE:
        log("✓ PASS — perfect staircase achieved")
    else:
        log("✗ FAIL — sequence mismatch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())
    env["E2E_INITIAL_CONFIG"] = "cpu_4"
    env["E2E_COOLDOWN"] = "60"
    env["E2E_COOLDOWN_DOWN"] = "240"
    env["E2E_EMA_WINDOW"] = "30"
    env["E2E_MIN_TPS"] = "10.0"
    env["E2E_SCALE_DOWN_CONCURRENCY"] = "5.0"
    env["E2E_RECENT_ACTIVITY_WINDOW"] = "15.0"

    total_duration = sum(p[1] for p in PHASES)

    log_json("BENCHMARK_START", {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "port": port,
        "total_duration_min": round(total_duration / 60, 1),
        "phases": [{
            "name": p[0], "duration_s": p[1],
            "concurrency": p[2], "rpm": p[3],
        } for p in PHASES],
        "expected_sequence": EXPECTED_SEQUENCE,
        "configs": {
            cid: {"throughput_tps": t, "hourly_cost": HOURLY_COSTS[cid]}
            for cid, t in MEASURED_THROUGHPUT.items()
        },
        "cooldown_s": 60,
        "cooldown_down_s": 240,
        "ema_window_s": 30,
        "min_tps_threshold": 10.0,
        "scale_down_concurrency": 5.0,
        "recent_activity_window": 15.0,
    })

    # Write server output to a separate log file
    server_log_path = (
        OUT_DIR.parent / "scaling_demo_logs"
        / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    server_log_file = open(server_log_path, "w")
    log(f"Server log → {server_log_path}")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "main_cost_aware:app",
         "--host", "0.0.0.0", "--port", str(port)],
        env=env, stdout=server_log_file, stderr=subprocess.STDOUT,
    )

    log(f"Server PID={proc.pid}, waiting for health...")

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
            server_log_file.flush()
            tail = Path(server_log_path).read_text()[-2000:]
            log(f"Server died (see {server_log_path}):\n{tail}")
            server_log_file.close()
            return
        await asyncio.sleep(3)

    if not healthy:
        server_log_file.flush()
        tail = Path(server_log_path).read_text()[-2000:]
        proc.kill()
        proc.wait()
        log(f"Server not healthy in {SERVER_STARTUP_TIMEOUT}s (see {server_log_path}):\n{tail}")
        server_log_file.close()
        return

    log(f"Server healthy at {base_url}")

    # Log initial status
    status = await _poll_status(base_url)
    if status:
        log_json("INITIAL_STATUS", status)

    collector = Collector(start_time=time.time())

    try:
        for phase_name, duration, concurrency, rpm in PHASES:
            await _run_phase(base_url, collector, phase_name,
                             duration=duration, concurrency=concurrency, rpm=rpm)

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

        server_log_file.close()
        log(f"Server log saved to {server_log_path}")

        # Clean up leftover containers
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        for name in [n.strip() for n in result.stdout.splitlines() if n.strip()]:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    # Final summary
    total_reqs = len(collector.requests)
    ok_reqs = sum(1 for r in collector.requests if r.success)
    fail_reqs = total_reqs - ok_reqs
    total_prompt_tok = sum(r.prompt_tokens for r in collector.requests if r.success)
    total_completion_tok = sum(r.completion_tokens for r in collector.requests if r.success)

    log_json("BENCHMARK_END", {
        "total_samples": len(collector.samples),
        "total_requests": total_reqs,
        "successful_requests": ok_reqs,
        "failed_requests": fail_reqs,
        "total_prompt_tokens": total_prompt_tok,
        "total_completion_tokens": total_completion_tok,
        "total_tokens": total_prompt_tok + total_completion_tok,
        "scaling_events": collector.scaling_events,
        "scaling_event_count": len(collector.scaling_events),
        "duration_minutes": round((time.time() - collector.start_time) / 60, 1),
    })

    generate_plot(collector)
    log("Done")


if __name__ == "__main__":
    asyncio.run(main())
