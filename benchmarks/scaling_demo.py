#!/usr/bin/env python3
"""
4-config scaling demo benchmark for thesis.

Starts the main cost-aware server and drives it through the full
vertical scaling staircase:
  cpu_4 → cpu_12 → gpu_25 → gpu_100 → gpu_25 → cpu_12 → cpu_4

Server config: cooldown=120s, EMA ~1min window, MIN_TPS=10.0, SCALE_DOWN_CONCURRENCY=5.0
Configs (measured throughput):
  cpu_4:    32 tok/s  ($0.05/hr)
  cpu_12:   47 tok/s  ($0.12/hr)
  gpu_25:  147 tok/s  ($0.50/hr)
  gpu_100: 1064 tok/s ($4.00/hr)

Phases (~21 min total):
  1. low load     (3 min):  1 worker,  rpm=3   → ~18 tok/s  (cpu_4 stays)
  2. medium load  (3 min):  3 workers, rpm=20  → ~8 tok/s   (→ cpu_12)
  3. high load    (3 min):  8 workers, back-to-back          (→ gpu_25)
  4. peak load    (3 min): 20 workers, back-to-back          (→ gpu_100)
  5. ramp-down 1  (3 min):  3 workers, rpm=20  → ~355 tok/s (→ gpu_25)
  6. ramp-down 2  (3 min):  2 workers, rpm=15  → ~24 tok/s  (→ cpu_12)
  7. ramp-down 3  (3 min):  1 worker,  rpm=3   → ~32 tok/s  (→ cpu_4)

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

# Phases — 3 min each (~21 min total)
PHASES = [
    ("low load",       180,  1,   3),
    ("medium load",    180,  3,  20),
    ("high load",      180,  8,   0),   # back-to-back to saturate cpu_12
    ("peak load",      180, 20,   0),   # back-to-back; gpu_25 has --parallel 4, so queuing drives TPS down
    ("ramp-down 1",    180,  3,  20),
    ("ramp-down 2",    180,  2,  15),
    ("ramp-down 3",    180,  1,   3),
]

EXPECTED_SEQUENCE = ["cpu_4", "cpu_12", "gpu_25", "gpu_100", "gpu_25", "cpu_12", "cpu_4"]


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
                config_id="",  # filled from status polls
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
        "ema_pct": model_info.get("ema_pct", 0),
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
    config_idx = np.array([CONFIG_ORDER.index(c) if c in CONFIG_ORDER else -1
                           for c in configs])

    phase_spans = _phase_spans(samples)

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

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 2, 1.5, 1.5]})

    for s, e, p in phase_spans:
        color = PHASE_COLORS.get(p, "#f5f5f5")
        for ax in axes:
            ax.axvspan(s, e, alpha=0.12, color=color, zorder=0)

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

    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax1b.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                         alpha=0.5, linewidth=1)
            ax1b.text(t[-1] * 1.01, thr, f"{thr:.0f}", fontsize=7,
                      va="center", color=CONFIG_COLORS[cid])

    ymax_demand = max(demand) if len(demand) > 0 and max(demand) > 0 else 1
    for s, e, p in phase_spans:
        mid = (s + e) / 2
        ax1b.text(mid, ymax_demand * 1.08, p, ha="center", va="bottom",
                  fontsize=7, fontstyle="italic", color="#555")

    ax1.set_ylabel("Hardware Configuration")
    ax1.set_yticks(range(len(CONFIG_ORDER)))
    ax1.set_yticklabels(CONFIG_ORDER)
    ax1.set_ylim(-0.5, len(CONFIG_ORDER) - 0.5)
    ax1.set_title("Cost-Aware Vertical Scaling — Real Hardware Demo")

    patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=c, alpha=0.5)
               for c in CONFIG_ORDER]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=len(CONFIG_ORDER))

    # Panel 2: Demand
    ax2 = axes[1]
    ax2.plot(t, demand, color="#333", linewidth=1.5, zorder=3)
    ax2.fill_between(t, demand, alpha=0.12, color="#4e79a7", zorder=2)
    ax2.set_ylabel("Demand (tok/s)")

    for cid, thr in MEASURED_THROUGHPUT.items():
        if thr < 200:
            ax2.axhline(thr, linestyle=":", color=CONFIG_COLORS[cid],
                        alpha=0.6, linewidth=1)
            ax2.text(t[-1] * 1.01, thr, f"{cid} cap",
                     fontsize=7, va="center", color=CONFIG_COLORS[cid])

    # Panel 3: Hourly cost
    ax3 = axes[2]
    for i in range(len(t) - 1):
        c = configs[i]
        ax3.fill_between([t[i], t[i + 1]], [costs[i], costs[i + 1]],
                         alpha=0.3, color=CONFIG_COLORS.get(c, "gray"),
                         step="post", zorder=2)
    ax3.step(t, costs, where="post", color="#333", linewidth=1.8, zorder=3)

    static_cost = HOURLY_COSTS["gpu_100"]
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
    env["E2E_COOLDOWN"] = "120"
    env["E2E_EMA_WINDOW"] = "60"
    env["E2E_MIN_TPS"] = "10.0"
    env["E2E_SCALE_DOWN_CONCURRENCY"] = "5.0"

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
        "headroom": 0.25,
        "cooldown_s": 120,
        "ema_window_s": 60,
        "min_tps_threshold": 10.0,
        "scale_down_concurrency": 5.0,
        "demand_window_s": 180,
    })

    # Write server output to a separate log file instead of a pipe.
    # Piping stdout causes the server to freeze once the 64KB buffer fills
    # because nobody drains it during the run.
    server_log_path = OUT_DIR.parent / "scaling_demo_logs" / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
