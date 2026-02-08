"""
Dedicated server for the 4-config scaling demo benchmark.

Configs: cpu_4, cpu_12, gpu_25, gpu_100 with asymmetric hysteresis
(headroom=0.25) validated by 320/320 simulation passes.

Scaling runs in a background task (not inline with requests) to avoid
blocking the event loop with subprocess.run() Docker commands.

Usage:
    uv run uvicorn benchmarks.scaling_demo_server:app --port <PORT>
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
import logging
import socket
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from main_cost_aware import (
    HardwareConfig,
    CostAwareAutoscaler,
    DemandTracker,
    Container,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Message,
    DEFAULT_THROUGHPUT,
    get_throughput,
    get_cost_per_token,
)

from fastapi import FastAPI, HTTPException
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIGS: List[HardwareConfig] = [
    HardwareConfig(cpu_cores=4,  memory="8g",  hourly_cost=0.05),
    HardwareConfig(cpu_cores=12, memory="8g",  hourly_cost=0.12),
    HardwareConfig(cpu_cores=2,  memory="8g",  gpu_percentage=25,  hourly_cost=0.50),
    HardwareConfig(cpu_cores=2,  memory="16g", gpu_percentage=100, hourly_cost=4.00),
]

MEASURED_THROUGHPUT: Dict[str, float] = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}
for k, v in MEASURED_THROUGHPUT.items():
    DEFAULT_THROUGHPUT[k] = v

HEADROOM = 0.25
COOLDOWN = 300
DEMAND_WINDOW = 180
SCALING_CHECK_INTERVAL = 10

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
MODEL_NAME = os.environ.get("E2E_MODEL_NAME", "")

autoscaler: Optional[CostAwareAutoscaler] = None
server_start_time: float = 0.0
scaling_in_progress: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _container_info(container: Container) -> Dict:
    """Extract full container metadata for logging."""
    config = container.config
    cid = config.config_id()
    cpt = get_cost_per_token(container.model_name, config) * 1e6
    info = {
        "container_name": container.container_name,
        "config_id": cid,
        "container_type": config.container_type,
        "cpu_cores": config.cpu_cores,
        "memory": config.memory,
        "gpu_percentage": config.gpu_percentage,
        "hourly_cost": config.hourly_cost,
        "image": config.image,
        "port": container.port,
        "parallel": config.cpu_cores or 1,
        "threads": config.cpu_cores or 1,
        "n_gpu_layers": 99 if config.gpu_percentage else 0,
        "docker_flags": container._docker_args(),
        "measured_throughput_tps": MEASURED_THROUGHPUT.get(cid, 0),
        "cost_per_token_micro": round(cpt, 4),
    }
    return info


def _log_json(tag: str, data: Dict) -> None:
    """Print a structured JSON log line."""
    print("[SERVER] [%s] %s" % (tag, json.dumps(data)), flush=True)


# ---------------------------------------------------------------------------
# Async container lifecycle (non-blocking)
# ---------------------------------------------------------------------------


async def _async_container_start(container: Container) -> bool:
    """Non-blocking container start using asyncio subprocess."""
    # Remove existing container
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()

    threads = container.config.cpu_cores or 1
    parallel = 32  # always 32 as per user requirement

    docker_cmd = [
        "docker", "run", "--rm", "-d",
        "--name", container.container_name,
        "-v", "%s:/models:ro" % str(container.model_path.parent),
        "-p", "%d:8080" % container.port,
        *container._docker_args(),
        container.config.image,
        "--server",
        "-m", "/models/%s" % container.model_path.name,
        "--host", "0.0.0.0",
        "--port", "8080",
        "--threads", str(threads),
        "--parallel", "32",
        "--slots",
    ]
    if container.config.gpu_percentage:
        docker_cmd.extend(["--n-gpu-layers", "99"])

    cmd_summary = " ".join(docker_cmd[-10:])
    _log_json("CONTAINER_START_CMD", {
        "container": container.container_name,
        "config_id": container.config.config_id(),
        "cmd_tail": cmd_summary,
        "full_cmd": docker_cmd,
    })

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        _log_json("CONTAINER_START_FAIL", {
            "container": container.container_name,
            "returncode": proc.returncode,
            "stderr": stderr.decode()[:500],
        })
        return False

    _log_json("CONTAINER_STARTED", {
        "container": container.container_name,
        "docker_id": stdout.decode().strip()[:12],
    })

    # Wait for health (non-blocking)
    for attempt in range(90):  # 90 * 2s = 180s
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                url = "http://localhost:%d/health" % container.port
                async with session.get(url) as resp:
                    if resp.status == 200:
                        container.is_ready = True
                        _log_json("CONTAINER_READY", {
                            "container": container.container_name,
                            "config_id": container.config.config_id(),
                            "wait_seconds": attempt * 2,
                        })
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)

    _log_json("CONTAINER_HEALTH_TIMEOUT", {
        "container": container.container_name,
        "waited_seconds": 180,
    })
    return False


async def _async_container_stop(container: Container) -> None:
    """Non-blocking container stop."""
    _log_json("CONTAINER_STOP", {"container": container.container_name})
    proc = await asyncio.create_subprocess_exec(
        "docker", "stop", container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    container.is_ready = False
    _log_json("CONTAINER_STOPPED", {"container": container.container_name})


# ---------------------------------------------------------------------------
# Slots-based demand estimation — saturation detection via derivatives
# ---------------------------------------------------------------------------
#
# Core idea: instead of trying to estimate absolute demand (which is capped
# by current hardware), we detect *saturation* — when the hardware can't
# keep up with incoming requests.
#
# Signals (polled every 1s from /slots endpoint):
#   1. throughput_tps: actual tok/s being produced (from n_decoded deltas)
#   2. concurrency: number of slots actively processing
#
# We maintain moving averages (MA) of both, plus their derivatives:
#   - d(throughput_ma)/dt ≈ 0  AND  d(concurrency_ma)/dt > 0
#     → hardware is saturated → scale UP
#   - concurrency drops, throughput drops proportionally
#     → over-provisioned → scale DOWN
#
# For the autoscaler interface (select_optimal_config needs a demand number),
# we translate saturation into a synthetic demand:
#   - saturated: demand = current_throughput * 1.5 (forces next config up)
#   - normal: demand = throughput_ma (actual production rate)
#   - idle: demand decays toward 0

SLOTS_POLL_INTERVAL = 1  # seconds
MA_WINDOW = 30           # seconds for moving average (short enough to react)
DERIVATIVE_WINDOW = 15   # seconds for derivative estimation
SATURATION_THROUGHPUT_DERIV_THRESHOLD = 0.5  # tok/s² — "almost flat"
SATURATION_CONCURRENCY_DERIV_THRESHOLD = 0.05  # slots/s — "rising"
SATURATION_MIN_CONCURRENCY = 2  # need at least this many active slots

_last_slots_info: Dict[str, Dict] = {}

# Per-slot n_decoded tracking for actual tok/s measurement
_prev_n_decoded: Dict[str, List[int]] = {}  # model -> list of n_decoded per slot
_prev_poll_time: Dict[str, float] = {}

# Ring buffers for MA computation
_throughput_history: Dict[str, List[tuple]] = {}  # model -> [(time, tps), ...]
_concurrency_history: Dict[str, List[tuple]] = {}  # model -> [(time, count), ...]


def _moving_average(history: List[tuple], now: float, window: float) -> float:
    """Compute simple moving average over the last `window` seconds."""
    cutoff = now - window
    vals = [v for t, v in history if t >= cutoff]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _derivative(history: List[tuple], now: float, window: float) -> float:
    """Estimate derivative using linear regression over the last `window` seconds.

    Returns the slope (units per second).
    """
    cutoff = now - window
    points = [(t, v) for t, v in history if t >= cutoff]
    if len(points) < 3:
        return 0.0

    # Simple linear regression: slope = Σ((t-t̄)(v-v̄)) / Σ((t-t̄)²)
    n = len(points)
    t_mean = sum(t for t, _ in points) / n
    v_mean = sum(v for _, v in points) / n

    num = sum((t - t_mean) * (v - v_mean) for t, v in points)
    den = sum((t - t_mean) ** 2 for t, _ in points)

    if den < 1e-9:
        return 0.0
    return num / den


def _trim_history(history: List[tuple], now: float, keep_seconds: float = 120.0) -> None:
    """Remove entries older than keep_seconds to bound memory."""
    cutoff = now - keep_seconds
    while history and history[0][0] < cutoff:
        history.pop(0)


async def _poll_slots(container: Container) -> Dict:
    """Query /slots and compute actual tok/s from n_decoded deltas."""
    port = container.port
    config = container.config
    config_id = config.config_id()

    result = {
        "processing_slots": 0,
        "total_slots": 32,
        "actual_tps": 0.0,
        "slot_details": [],
    }

    try:
        url = "http://localhost:%d/slots" % port
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return result
                slots = await resp.json()

        now = time.time()
        processing = 0
        total = len(slots)
        total_new_tokens = 0
        details = []

        # Get previous n_decoded values for delta computation
        prev_decoded = _prev_n_decoded.get(config_id, [0] * total)
        prev_time = _prev_poll_time.get(config_id, now)
        dt = now - prev_time

        current_decoded = []
        for i, slot in enumerate(slots):
            is_processing = slot.get("is_processing", False)
            n_decoded = slot.get("n_decoded", 0)
            if is_processing:
                processing += 1

            # Compute tokens generated since last poll
            if i < len(prev_decoded) and dt > 0:
                delta = n_decoded - prev_decoded[i]
                # n_decoded resets to 0 when a new request starts on this slot,
                # so negative delta means a new request — count current n_decoded
                if delta < 0:
                    delta = n_decoded
                total_new_tokens += max(0, delta)

            current_decoded.append(n_decoded)
            details.append({
                "id": slot.get("id", -1),
                "is_processing": is_processing,
                "n_decoded": n_decoded,
                "n_remaining": slot.get("n_remaining", -1),
            })

        # Store for next poll
        _prev_n_decoded[config_id] = current_decoded
        _prev_poll_time[config_id] = now

        actual_tps = total_new_tokens / dt if dt > 0.1 else 0.0

        result["processing_slots"] = processing
        result["total_slots"] = max(total, 1)
        result["actual_tps"] = actual_tps
        result["slot_details"] = details

    except Exception as e:
        _log_json("SLOTS_POLL_EXCEPTION", {
            "port": port, "config_id": config_id,
            "error": str(e)[:200],
        })

    return result


def get_saturation_demand(model: str) -> tuple:
    """Compute the synthetic demand signal from saturation detection.

    Returns (demand_tps, is_saturated, debug_info).
    """
    now = time.time()
    thr_hist = _throughput_history.get(model, [])
    con_hist = _concurrency_history.get(model, [])

    throughput_ma = _moving_average(thr_hist, now, MA_WINDOW)
    concurrency_ma = _moving_average(con_hist, now, MA_WINDOW)
    throughput_deriv = _derivative(thr_hist, now, DERIVATIVE_WINDOW)
    concurrency_deriv = _derivative(con_hist, now, DERIVATIVE_WINDOW)

    # Get current config's measured throughput as the ceiling
    current_config = autoscaler.current_config.get(model)
    config_id = current_config.config_id() if current_config else "cpu_4"
    config_throughput = MEASURED_THROUGHPUT.get(config_id, 32.0)

    is_saturated = False
    demand = throughput_ma  # default: actual production rate

    # Saturation detection:
    # Throughput has plateaued (derivative near zero or negative)
    # AND concurrency is still rising (derivative positive)
    # AND we have meaningful concurrency (not just 1 idle slot)
    throughput_flat = abs(throughput_deriv) < SATURATION_THROUGHPUT_DERIV_THRESHOLD
    concurrency_rising = concurrency_deriv > SATURATION_CONCURRENCY_DERIV_THRESHOLD
    enough_concurrency = concurrency_ma >= SATURATION_MIN_CONCURRENCY

    # Also detect saturation when throughput is near the config's measured max
    # and concurrency is non-trivial (even if not actively rising)
    near_capacity = throughput_ma > config_throughput * 0.75
    high_concurrency = concurrency_ma >= 4

    if (throughput_flat and concurrency_rising and enough_concurrency):
        is_saturated = True
        # Signal demand higher than current config can handle
        # Use 1.5x current throughput to push past the headroom threshold
        demand = config_throughput * 1.5
    elif near_capacity and high_concurrency:
        is_saturated = True
        demand = config_throughput * 1.5

    debug = {
        "throughput_ma": round(throughput_ma, 2),
        "concurrency_ma": round(concurrency_ma, 2),
        "throughput_deriv": round(throughput_deriv, 4),
        "concurrency_deriv": round(concurrency_deriv, 4),
        "is_saturated": is_saturated,
        "demand": round(demand, 2),
        "config_throughput": config_throughput,
        "thr_flat": throughput_flat,
        "con_rising": concurrency_rising,
        "near_capacity": near_capacity,
        "high_concurrency": high_concurrency,
        "history_len": len(thr_hist),
    }

    return demand, is_saturated, debug


async def _slots_polling_loop() -> None:
    """Background task: poll /slots every 1s, track throughput + concurrency."""
    while True:
        await asyncio.sleep(SLOTS_POLL_INTERVAL)
        now = time.time()

        for model_name, container in list(autoscaler.containers.items()):
            if not container.is_ready:
                continue
            config = autoscaler.current_config.get(model_name)
            if not config:
                continue

            slots_info = await _poll_slots(container)
            _last_slots_info[model_name] = slots_info

            actual_tps = slots_info.get("actual_tps", 0.0)
            processing = slots_info.get("processing_slots", 0)

            # Append to ring buffers
            if model_name not in _throughput_history:
                _throughput_history[model_name] = []
                _concurrency_history[model_name] = []

            _throughput_history[model_name].append((now, actual_tps))
            _concurrency_history[model_name].append((now, float(processing)))

            # Trim old data
            _trim_history(_throughput_history[model_name], now)
            _trim_history(_concurrency_history[model_name], now)


# ---------------------------------------------------------------------------
# Background scaling loop
# ---------------------------------------------------------------------------


async def _background_scaling_loop() -> None:
    """Background task that checks scaling every SCALING_CHECK_INTERVAL.

    Demand signal: saturation detection via derivatives of throughput MA
    and concurrency MA from /slots polling.

    - When throughput plateaus but concurrency rises → saturated → scale UP
      (synthetic demand = current_throughput * 1.5)
    - When throughput is near capacity with high concurrency → saturated
    - Otherwise demand = throughput_ma (actual production rate)
    """
    global scaling_in_progress
    while True:
        await asyncio.sleep(SCALING_CHECK_INTERVAL)
        if scaling_in_progress:
            continue

        for model_name in list(autoscaler.containers.keys()):
            container = autoscaler.containers.get(model_name)
            current_config = autoscaler.current_config.get(model_name)

            now = time.time()
            slots_info = _last_slots_info.get(model_name, {})

            # Saturation-based demand
            demand, is_saturated, sat_debug = get_saturation_demand(model_name)

            # Also keep completed EMA for logging (still recorded in request handler)
            completed_ema = autoscaler.demand_tracker.get_demand(model_name)

            # Log demand state every check cycle
            _log_json("DEMAND_CHECK", {
                "model": model_name,
                "elapsed": round(now - server_start_time, 3),
                "config_id": current_config.config_id() if current_config else "none",
                "demand_tps": round(demand, 4),
                "is_saturated": is_saturated,
                "completed_ema_tps": round(completed_ema, 4),
                "throughput_ma": sat_debug["throughput_ma"],
                "concurrency_ma": sat_debug["concurrency_ma"],
                "throughput_deriv": sat_debug["throughput_deriv"],
                "concurrency_deriv": sat_debug["concurrency_deriv"],
                "thr_flat": sat_debug["thr_flat"],
                "con_rising": sat_debug["con_rising"],
                "near_capacity": sat_debug["near_capacity"],
                "high_concurrency": sat_debug["high_concurrency"],
                "actual_tps": round(slots_info.get("actual_tps", 0.0), 4),
                "processing_slots": slots_info.get("processing_slots", 0),
                "total_slots": slots_info.get("total_slots", 0),
                "active_requests": container.active_requests if container else 0,
                "history_len": sat_debug["history_len"],
            })

            # Check cooldown
            last_scale = autoscaler.last_scale_time.get(model_name, 0)
            if now - last_scale < autoscaler.cooldown_seconds:
                continue

            # Use saturation-based demand for scaling decision
            optimal = autoscaler.select_optimal_config(
                model_name, demand, current=current_config
            )
            if current_config and optimal.config_id() == current_config.config_id():
                continue

            new_config = optimal
            old_config = current_config
            old_config_id = old_config.config_id() if old_config else "none"
            new_config_id = new_config.config_id()

            scale_start_data = {
                "event": "scaling_start",
                "timestamp": now,
                "elapsed": round(now - server_start_time, 3),
                "model": model_name,
                "from_config": old_config_id,
                "to_config": new_config_id,
                "demand_tps": round(demand, 4),
                "is_saturated": is_saturated,
                "completed_ema_tps": round(completed_ema, 4),
                "throughput_ma": sat_debug["throughput_ma"],
                "concurrency_ma": sat_debug["concurrency_ma"],
                "throughput_deriv": sat_debug["throughput_deriv"],
                "concurrency_deriv": sat_debug["concurrency_deriv"],
                "actual_tps": round(slots_info.get("actual_tps", 0.0), 4),
                "processing_slots": slots_info.get("processing_slots", 0),
                "total_slots": slots_info.get("total_slots", 0),
                "active_requests": container.active_requests if container else 0,
                "from_hourly_cost": old_config.hourly_cost if old_config else 0,
                "to_hourly_cost": new_config.hourly_cost,
                "from_throughput": MEASURED_THROUGHPUT.get(old_config_id, 0),
                "to_throughput": MEASURED_THROUGHPUT.get(new_config_id, 0),
            }
            _log_json("SCALING_START", scale_start_data)

            scaling_in_progress = True
            try:
                old_container = autoscaler.containers.get(model_name)
                if old_container:
                    model_path = old_container.model_path
                else:
                    model_path = autoscaler.get_model_path(model_name)

                port = autoscaler._get_port()
                new_container = Container(model_name, model_path, new_config, port)

                scale_start = time.time()
                success = await _async_container_start(new_container)

                if success:
                    # Swap
                    autoscaler.containers[model_name] = new_container
                    autoscaler.current_config[model_name] = new_config
                    autoscaler.last_scale_time[model_name] = autoscaler.clock()
                    scale_duration = time.time() - scale_start

                    done_data = {
                        "event": "scaling_done",
                        "timestamp": time.time(),
                        "elapsed": round(time.time() - server_start_time, 3),
                        "model": model_name,
                        "from_config": old_config_id,
                        "to_config": new_config_id,
                        "scale_duration_s": round(scale_duration, 1),
                        "new_container": _container_info(new_container),
                    }
                    _log_json("SCALING_DONE", done_data)

                    # Stop old container in background
                    # Wait for in-flight requests to drain first (up to 60s)
                    if old_container:
                        drain_start = time.time()
                        while old_container.active_requests > 0 and (time.time() - drain_start) < 60:
                            _log_json("DRAIN_WAIT", {
                                "container": old_container.container_name,
                                "active_requests": old_container.active_requests,
                                "waited_s": round(time.time() - drain_start, 1),
                            })
                            await asyncio.sleep(2)
                        await _async_container_stop(old_container)
                else:
                    fail_data = {
                        "event": "scaling_fail",
                        "timestamp": time.time(),
                        "elapsed": round(time.time() - server_start_time, 3),
                        "model": model_name,
                        "from_config": old_config_id,
                        "to_config": new_config_id,
                    }
                    _log_json("SCALING_FAIL", fail_data)
            finally:
                scaling_in_progress = False


# ---------------------------------------------------------------------------
# FastAPI app + endpoints
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global autoscaler, server_start_time
    server_start_time = time.time()

    model_name = MODEL_NAME
    models_dir = MODELS_DIR

    autoscaler = CostAwareAutoscaler(
        configs=CONFIGS,
        cooldown_seconds=COOLDOWN,
        models_dir=models_dir,
        headroom=HEADROOM,
    )
    autoscaler.demand_tracker = DemandTracker(window_seconds=DEMAND_WINDOW)

    # Start initial container on cheapest config
    cheapest = min(CONFIGS, key=lambda c: c.hourly_cost)
    model_path = autoscaler.get_model_path(model_name)
    if not model_path:
        # Try all models in directory
        mdir = Path(models_dir).resolve()
        for f in mdir.iterdir():
            if f.suffix.lower() in [".gguf", ".bin"]:
                model_path = f
                model_name = f.stem
                break

    if not model_path:
        logger.error("No model found in %s" % models_dir)
        yield
        return

    port = autoscaler._get_port()
    container = Container(model_name, model_path, cheapest, port)

    _log_json("INIT", {
        "model": model_name,
        "model_path": str(model_path),
        "initial_config": cheapest.config_id(),
        "configs": [c.config_id() for c in CONFIGS],
        "throughputs": MEASURED_THROUGHPUT,
        "headroom": HEADROOM,
        "cooldown_s": COOLDOWN,
        "demand_window_s": DEMAND_WINDOW,
    })

    if await _async_container_start(container):
        autoscaler.containers[model_name] = container
        autoscaler.current_config[model_name] = cheapest
        autoscaler.last_scale_time[model_name] = autoscaler.clock()
        _log_json("INIT_OK", {
            "model": model_name,
            "config": cheapest.config_id(),
            "container": _container_info(container),
        })
    else:
        logger.error("Failed to start initial container")
        yield
        return

    # Start background tasks
    slots_task = asyncio.create_task(_slots_polling_loop())
    scaling_task = asyncio.create_task(_background_scaling_loop())

    yield

    # Shutdown
    slots_task.cancel()
    scaling_task.cancel()
    try:
        await slots_task
    except asyncio.CancelledError:
        pass
    try:
        await scaling_task
    except asyncio.CancelledError:
        pass

    for c in autoscaler.containers.values():
        await _async_container_stop(c)
    _log_json("SHUTDOWN", {"message": "all containers stopped"})


app = FastAPI(title="Scaling Demo Server", lifespan=lifespan)


@app.get("/health")
async def health():
    if autoscaler is None:
        return {"status": "starting"}
    ready = sum(1 for c in autoscaler.containers.values() if c.is_ready)
    return {
        "status": "healthy" if ready > 0 else "down",
        "ready_containers": ready,
        "models": list(autoscaler.containers.keys()),
    }


@app.get("/status")
async def status():
    if autoscaler is None:
        return {}
    base = autoscaler.get_status()
    base["server_uptime_seconds"] = round(time.time() - server_start_time, 1)
    base["scaling_in_progress"] = scaling_in_progress
    # Override demand_tps with saturation-based demand — the actual scaling signal
    for model_name in base.get("models", {}):
        demand, is_saturated, sat_debug = get_saturation_demand(model_name)
        base["models"][model_name]["demand_tps"] = round(demand, 4)
        base["models"][model_name]["is_saturated"] = is_saturated
        base["models"][model_name]["throughput_ma"] = sat_debug["throughput_ma"]
        base["models"][model_name]["concurrency_ma"] = sat_debug["concurrency_ma"]
        base["models"][model_name]["throughput_deriv"] = sat_debug["throughput_deriv"]
        base["models"][model_name]["concurrency_deriv"] = sat_debug["concurrency_deriv"]
    return base


@app.get("/v1/models")
async def list_models():
    if autoscaler is None:
        return {"models": []}
    return {"models": list(autoscaler.containers.keys())}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completions using native /completion endpoint for timings."""
    if autoscaler is None:
        raise HTTPException(503, "Server not ready")

    # Get container directly — do NOT call autoscaler.get_container()
    # because that triggers inline blocking scaling via scale_to()
    container = autoscaler.containers.get(request.model)
    if not container or not container.is_ready:
        raise HTTPException(404, "Model '%s' not found or not ready" % request.model)

    # Track active request
    async with container.lock:
        container.active_requests += 1
        container.total_requests += 1

    req_id = str(uuid.uuid4())[:8]
    req_start = time.time()
    config = autoscaler.current_config.get(request.model)
    config_id = config.config_id() if config else "unknown"

    try:
        # Build prompt from messages
        prompt_parts = []
        for m in request.messages:
            prompt_parts.append("%s: %s" % (m.role, m.content))
        prompt_text = "\n".join(prompt_parts)

        # Use native /completion endpoint to get timings
        payload = {
            "prompt": prompt_text,
            "n_predict": request.max_tokens or 256,
            "temperature": request.temperature or 0.7,
        }

        endpoint = container.get_endpoint()
        url = "%s/completion" % endpoint

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    err_data = {
                        "req_id": req_id,
                        "status": resp.status,
                        "body": body[:300],
                        "config_id": config_id,
                    }
                    _log_json("REQ_UPSTREAM_ERR", err_data)
                    raise HTTPException(resp.status, "Container error: %s" % body[:200])

                result = await resp.json()

        wall_ms = (time.time() - req_start) * 1000

        # Extract timings from native endpoint
        timings = result.get("timings", {})
        content = result.get("content", "")

        prompt_n = timings.get("prompt_n", 0)
        prompt_ms = timings.get("prompt_ms", 0)
        prompt_per_second = timings.get("prompt_per_second", 0)
        prompt_per_token_ms = timings.get("prompt_per_token_ms", 0)
        predicted_n = timings.get("predicted_n", 0)
        predicted_ms = timings.get("predicted_ms", 0)
        predicted_per_second = timings.get("predicted_per_second", 0)
        predicted_per_token_ms = timings.get("predicted_per_token_ms", 0)

        total_tokens = prompt_n + predicted_n

        # Record demand
        if total_tokens > 0:
            autoscaler.demand_tracker.record_tokens(request.model, total_tokens)

        demand_now = autoscaler.demand_tracker.get_demand(request.model)
        cpt = get_cost_per_token(request.model, config) * 1e6 if config else 0

        # Log every detail
        req_log = {
            "req_id": req_id,
            "elapsed_s": round(time.time() - server_start_time, 1),
            "config_id": config_id,
            "wall_ms": round(wall_ms, 1),
            "prompt_tokens": prompt_n,
            "completion_tokens": predicted_n,
            "total_tokens": total_tokens,
            "prompt_eval_ms": round(prompt_ms, 1),
            "generation_ms": round(predicted_ms, 1),
            "prompt_ms_per_token": round(prompt_per_token_ms, 3),
            "generation_ms_per_token": round(predicted_per_token_ms, 3),
            "prompt_tps": round(prompt_per_second, 1),
            "generation_tps": round(predicted_per_second, 1),
            "ttft_ms": round(prompt_ms, 1),
            "demand_tps": round(demand_now, 4),
            "cost_per_token_micro": round(cpt, 4),
            "container": container.container_name,
            "port": container.port,
            "parallel": 32,
            "threads": config.cpu_cores or 1 if config else 1,
            "n_gpu_layers": 99 if (config and config.gpu_percentage) else 0,
            "raw_timings": timings,
        }
        _log_json("REQ_OK", req_log)

        # Build OpenAI-compatible response with timings passthrough
        response = {
            "id": "chatcmpl-%s" % req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_n,
                "completion_tokens": predicted_n,
                "total_tokens": total_tokens,
            },
            "timings": timings,
        }
        return response

    except HTTPException:
        raise
    except Exception as e:
        wall_ms = (time.time() - req_start) * 1000
        err_data = {
            "req_id": req_id,
            "config_id": config_id,
            "wall_ms": round(wall_ms, 1),
            "error": str(e)[:300],
        }
        _log_json("REQ_EXCEPTION", err_data)
        raise HTTPException(500, "Internal error: %s" % str(e)[:200])
    finally:
        async with container.lock:
            container.active_requests = max(0, container.active_requests - 1)
