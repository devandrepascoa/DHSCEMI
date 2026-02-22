"""
Dedicated server for the 4-config scaling demo benchmark.

Scaling logic: /metrics-based throughput EMA.
  Poll /metrics for n_tokens_predicted_total + n_prompt_tokens_processed_total,
  compute delta_tokens / delta_time = instantaneous aggregate tok/s,
  feed to a 4-minute EMA.

  - Scale UP:   throughput_ema >= SCALE_UP_MULT * measured_throughput[current]
  - Scale DOWN: throughput_ema <= SCALE_DOWN_MULT * measured_throughput[cheaper]

Usage:
    uv run uvicorn benchmarks.scaling_demo_server:app --port <PORT>
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from main_cost_aware import (
    HardwareConfig,
    CostAwareAutoscaler,
    DemandTracker,
    Container,
    ChatCompletionRequest,
    Message,
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
CONFIGS_BY_COST = sorted(CONFIGS, key=lambda c: c.hourly_cost)

COOLDOWN = int(os.environ.get("E2E_COOLDOWN", "300"))
DEMAND_WINDOW = 180
SCALING_CHECK_INTERVAL = 10

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
MODEL_NAME = os.environ.get("E2E_MODEL_NAME", "")

autoscaler: Optional[CostAwareAutoscaler] = None
server_start_time: float = 0.0
scaling_in_progress: bool = False

# ---------------------------------------------------------------------------
# Scaling parameters (must match scaling_simulation.py)
# ---------------------------------------------------------------------------
METRICS_POLL_INTERVAL = 1.0   # poll /metrics every 1s
SCALE_UP_MULT = 0.8           # scale up at 80% of current capacity
SCALE_DOWN_MULT = 0.3         # scale down when <= 30% of cheaper capacity
EMA_ALPHA = 2.0 / (240 + 1)  # ~4min EMA window

MEASURED_THROUGHPUT: Dict[str, float] = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}

# EMA state
_throughput_ema: Dict[str, float] = {}
_last_ema_time: Dict[str, float] = {}

# /metrics polling state — previous counter values
_prev_total_tokens: Dict[str, float] = {}
_prev_metrics_time: Dict[str, float] = {}
_last_metrics_tps: Dict[str, float] = {}  # last instantaneous tok/s for logging
_last_predicted_tps_gauge: Dict[str, float] = {}  # predicted_tokens_seconds gauge

# Streaming token counter — incremented in real-time as tokens arrive
_streaming_token_count: Dict[str, int] = {}
_prev_streaming_count: Dict[str, int] = {}
_prev_streaming_time: Dict[str, float] = {}
_last_streaming_tps: Dict[str, float] = {}  # last 1s streaming tok/s

# After a scaling event, wait for the new container to process its first
# tokens before resuming EMA updates, scaling decisions, and cooldown timer.
_waiting_for_first_token: Dict[str, bool] = {}

# Regex patterns for prometheus metrics
_RE_PREDICTED = re.compile(
    r'^llamacpp:tokens_predicted_total\s+(\d+(?:\.\d+)?)', re.MULTILINE
)
_RE_PROMPT = re.compile(
    r'^llamacpp:prompt_tokens_total\s+(\d+(?:\.\d+)?)', re.MULTILINE
)
_RE_PREDICTED_TPS = re.compile(
    r'^llamacpp:predicted_tokens_seconds\s+(\d+(?:\.\d+)?)', re.MULTILINE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _container_info(container: Container) -> Dict:
    config = container.config
    cid = config.config_id()
    cpt = get_cost_per_token(container.model_name, config) * 1e6
    return {
        "container_name": container.container_name,
        "config_id": cid,
        "container_type": config.container_type,
        "cpu_cores": config.cpu_cores,
        "memory": config.memory,
        "gpu_percentage": config.gpu_percentage,
        "hourly_cost": config.hourly_cost,
        "image": config.image,
        "port": container.port,
        "parallel": 32,
        "threads": config.cpu_cores or 1,
        "n_gpu_layers": 99 if config.gpu_percentage else 0,
        "docker_flags": container._docker_args(),
        "measured_throughput_tps": MEASURED_THROUGHPUT.get(cid, 0),
        "cost_per_token_micro": round(cpt, 4),
    }


def _log_json(tag: str, data: Dict) -> None:
    print("[SERVER] [%s] %s" % (tag, json.dumps(data)), flush=True)


# ---------------------------------------------------------------------------
# Async container lifecycle
# ---------------------------------------------------------------------------

async def _async_container_start(container: Container) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()

    threads = container.config.cpu_cores or 1

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
        "--metrics",
    ]
    if container.config.gpu_percentage:
        docker_cmd.extend(["--n-gpu-layers", "99"])

    _log_json("CONTAINER_START_CMD", {
        "container": container.container_name,
        "config_id": container.config.config_id(),
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

    for attempt in range(90):
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
    _log_json("CONTAINER_STOP", {"container": container.container_name})
    proc = await asyncio.create_subprocess_exec(
        "docker", "stop", container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    container.is_ready = False
    _log_json("CONTAINER_STOPPED", {"container": container.container_name})


# ---------------------------------------------------------------------------
# /metrics polling + EMA
# ---------------------------------------------------------------------------

async def _poll_metrics(container: Container) -> Optional[Dict]:
    """Poll /metrics endpoint and parse prometheus counters.

    Returns dict with:
      - total_tokens: n_tokens_predicted_total + n_prompt_tokens_processed_total
      - predicted_total: n_tokens_predicted_total
      - prompt_total: n_prompt_tokens_processed_total
    Or None on failure.
    """
    try:
        url = "http://localhost:%d/metrics" % container.port
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()

        predicted = 0.0
        prompt = 0.0
        predicted_tps_gauge = 0.0
        m = _RE_PREDICTED.search(text)
        if m:
            predicted = float(m.group(1))
        m = _RE_PROMPT.search(text)
        if m:
            prompt = float(m.group(1))
        m = _RE_PREDICTED_TPS.search(text)
        if m:
            predicted_tps_gauge = float(m.group(1))

        return {
            "total_tokens": predicted,
            "predicted_total": predicted,
            "prompt_total": prompt,
            "predicted_tps_gauge": predicted_tps_gauge,
        }
    except Exception as e:
        _log_json("METRICS_POLL_EXCEPTION", {
            "port": container.port,
            "error": str(e)[:200],
        })
        return None


def _update_ema(model: str, tps: float, now: float) -> float:
    """Update throughput EMA and return current value.

    Seeds with first observation, then time-correct exponential smoothing.
    """
    if model not in _throughput_ema:
        _throughput_ema[model] = tps
        _last_ema_time[model] = now
        return tps

    dt = now - _last_ema_time.get(model, now)
    if dt <= 0:
        return _throughput_ema.get(model, 0.0)

    decay = (1.0 - EMA_ALPHA) ** dt
    _throughput_ema[model] = _throughput_ema[model] * decay + (1.0 - decay) * tps
    _last_ema_time[model] = now
    return _throughput_ema[model]


def select_config(
    current_config: HardwareConfig,
    throughput_ema: float,
) -> HardwareConfig:
    """Select config based on throughput EMA vs measured capacity thresholds.

    Scale UP:   throughput_ema >= SCALE_UP_MULT * capacity[current]
    Scale DOWN: throughput_ema <= SCALE_DOWN_MULT * capacity[current]
                AND throughput_ema <= 0.7 * capacity[cheaper]
    """
    current_id = current_config.config_id()
    current_idx = next(
        i for i, c in enumerate(CONFIGS_BY_COST) if c.config_id() == current_id
    )
    current_capacity = MEASURED_THROUGHPUT.get(current_id, 1.0)

    # Check scale UP
    if throughput_ema >= SCALE_UP_MULT * current_capacity:
        if current_idx + 1 < len(CONFIGS_BY_COST):
            return CONFIGS_BY_COST[current_idx + 1]

    # Check scale DOWN: only if underusing current AND cheaper can handle it
    if current_idx > 0:
        cheaper = CONFIGS_BY_COST[current_idx - 1]
        cheaper_capacity = MEASURED_THROUGHPUT.get(cheaper.config_id(), 1.0)
        if (throughput_ema <= SCALE_DOWN_MULT * current_capacity
                and throughput_ema <= 0.75 * cheaper_capacity):
            return cheaper

    return current_config


async def _streaming_throughput_loop() -> None:
    """Compute streaming tok/s every METRICS_POLL_INTERVAL from _streaming_token_count.

    This replaces the /metrics-based throughput signal with a smooth,
    real-time counter that increments as SSE tokens arrive.
    """
    while True:
        await asyncio.sleep(METRICS_POLL_INTERVAL)
        now = time.time()

        for model_name in list(autoscaler.containers.keys()):
            container = autoscaler.containers.get(model_name)
            if not container or not container.is_ready:
                continue

            current_count = _streaming_token_count.get(model_name, 0)

            # First tick after startup or scaling — record baseline
            if model_name not in _prev_streaming_count:
                _prev_streaming_count[model_name] = current_count
                _prev_streaming_time[model_name] = now
                continue

            prev_count = _prev_streaming_count[model_name]
            prev_time = _prev_streaming_time[model_name]
            dt = now - prev_time

            _prev_streaming_count[model_name] = current_count
            _prev_streaming_time[model_name] = now

            if dt <= 0:
                continue

            delta_tokens = current_count - prev_count
            tps = delta_tokens / dt
            _last_streaming_tps[model_name] = tps

            # If waiting for first token after scaling, check streaming counter
            if _waiting_for_first_token.get(model_name, False):
                if delta_tokens > 0:
                    _waiting_for_first_token[model_name] = False
                    autoscaler.last_scale_time[model_name] = autoscaler.clock()
                    _log_json("FIRST_TOKEN_AFTER_SCALE", {
                        "model": model_name,
                        "elapsed": round(now - server_start_time, 3),
                        "config_id": autoscaler.current_config.get(
                            model_name, CONFIGS[0]
                        ).config_id(),
                        "streaming_count": current_count,
                        "delta_tokens": delta_tokens,
                        "tps": round(tps, 2),
                    })
                    # Seed EMA with first observation
                    _update_ema(model_name, tps, now)
                continue

            _update_ema(model_name, tps, now)


async def _metrics_polling_loop() -> None:
    """Poll /metrics every METRICS_POLL_INTERVAL for the gauge value (logging only)."""
    while True:
        await asyncio.sleep(METRICS_POLL_INTERVAL)

        for model_name, container in list(autoscaler.containers.items()):
            if not container.is_ready:
                continue

            metrics = await _poll_metrics(container)
            if metrics is None:
                continue

            _last_predicted_tps_gauge[model_name] = metrics.get(
                "predicted_tps_gauge", 0.0
            )
            _last_metrics_tps[model_name] = 0.0  # no longer used for EMA


# ---------------------------------------------------------------------------
# Background scaling loop
# ---------------------------------------------------------------------------

async def _background_scaling_loop() -> None:
    """Check scaling every SCALING_CHECK_INTERVAL using throughput EMA."""
    global scaling_in_progress
    while True:
        await asyncio.sleep(SCALING_CHECK_INTERVAL)
        if scaling_in_progress:
            continue

        for model_name in list(autoscaler.containers.keys()):
            container = autoscaler.containers.get(model_name)
            current_config = autoscaler.current_config.get(model_name)
            if not container or not current_config:
                continue

            # Don't make scaling decisions until the new container has
            # processed its first tokens (EMA hasn't started yet).
            if _waiting_for_first_token.get(model_name, False):
                continue

            now = time.time()
            ema = _throughput_ema.get(model_name, 0.0)
            current_id = current_config.config_id()
            capacity = MEASURED_THROUGHPUT.get(current_id, 1.0)
            streaming_tps = _last_streaming_tps.get(model_name, 0.0)
            predicted_tps_gauge = _last_predicted_tps_gauge.get(model_name, 0.0)

            _log_json("DEMAND_CHECK", {
                "model": model_name,
                "elapsed": round(now - server_start_time, 3),
                "config_id": current_id,
                "throughput_ema": round(ema, 4),
                "streaming_tps": round(streaming_tps, 4),
                "predicted_tps_gauge": round(predicted_tps_gauge, 4),
                "capacity": capacity,
                "ema_pct_of_capacity": round(
                    ema / capacity * 100, 1
                ) if capacity > 0 else 0,
                "scale_up_threshold": round(SCALE_UP_MULT * capacity, 1),
                "active_requests": container.active_requests,
            })

            # Check cooldown
            last_scale = autoscaler.last_scale_time.get(model_name, 0)
            if now - last_scale < autoscaler.cooldown_seconds:
                continue

            optimal = select_config(current_config, ema)
            if optimal.config_id() == current_id:
                continue

            new_config = optimal
            old_config_id = current_id
            new_config_id = new_config.config_id()

            _log_json("SCALING_START", {
                "event": "scaling_start",
                "timestamp": now,
                "elapsed": round(now - server_start_time, 3),
                "model": model_name,
                "from_config": old_config_id,
                "to_config": new_config_id,
                "throughput_ema": round(ema, 4),
                "from_capacity": MEASURED_THROUGHPUT.get(old_config_id, 0),
                "to_capacity": MEASURED_THROUGHPUT.get(new_config_id, 0),
                "from_hourly_cost": current_config.hourly_cost,
                "to_hourly_cost": new_config.hourly_cost,
                "active_requests": container.active_requests,
            })

            scaling_in_progress = True
            try:
                old_container = autoscaler.containers.get(model_name)
                model_path = (
                    old_container.model_path
                    if old_container
                    else autoscaler.get_model_path(model_name)
                )

                port = autoscaler._get_port()
                new_container = Container(
                    model_name, model_path, new_config, port
                )

                scale_start = time.time()

                # Stop old container first, then start new one
                if old_container:
                    await _async_container_stop(old_container)

                success = await _async_container_start(new_container)
                if success:
                    autoscaler.containers[model_name] = new_container
                    autoscaler.current_config[model_name] = new_config
                    # Don't start cooldown yet — wait for first token.
                    _waiting_for_first_token[model_name] = True
                    # Reset all throughput tracking state
                    _throughput_ema.pop(model_name, None)
                    _last_ema_time.pop(model_name, None)
                    _prev_total_tokens.pop(model_name, None)
                    _prev_metrics_time.pop(model_name, None)
                    _last_metrics_tps.pop(model_name, None)
                    _last_predicted_tps_gauge.pop(model_name, None)
                    # Reset streaming counters
                    _streaming_token_count.pop(model_name, None)
                    _prev_streaming_count.pop(model_name, None)
                    _prev_streaming_time.pop(model_name, None)
                    _last_streaming_tps.pop(model_name, None)

                    _log_json("SCALING_DONE", {
                        "event": "scaling_done",
                        "timestamp": time.time(),
                        "elapsed": round(
                            time.time() - server_start_time, 3
                        ),
                        "model": model_name,
                        "from_config": old_config_id,
                        "to_config": new_config_id,
                        "scale_duration_s": round(
                            time.time() - scale_start, 1
                        ),
                        "new_container": _container_info(new_container),
                    })
                else:
                    _log_json("SCALING_FAIL_ROLLBACK", {
                        "event": "scaling_fail",
                        "timestamp": time.time(),
                        "elapsed": round(
                            time.time() - server_start_time, 3
                        ),
                        "model": model_name,
                        "from_config": old_config_id,
                        "to_config": new_config_id,
                        "action": "restarting_old_container",
                    })
                    old_port = autoscaler._get_port()
                    rollback = Container(
                        model_name, model_path, current_config, old_port
                    )
                    if await _async_container_start(rollback):
                        autoscaler.containers[model_name] = rollback
                    else:
                        _log_json("ROLLBACK_FAIL", {
                            "model": model_name,
                            "config": old_config_id,
                        })
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
        headroom=0.0,
    )
    autoscaler.demand_tracker = DemandTracker(window_seconds=DEMAND_WINDOW)

    initial_config_id = os.environ.get("E2E_INITIAL_CONFIG", "")
    initial_config = None
    if initial_config_id:
        for c in CONFIGS:
            if c.config_id() == initial_config_id:
                initial_config = c
                break
    if initial_config is None:
        initial_config = min(CONFIGS, key=lambda c: c.hourly_cost)
    model_path = autoscaler.get_model_path(model_name)
    if not model_path:
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
    container = Container(model_name, model_path, initial_config, port)

    _log_json("INIT", {
        "model": model_name,
        "model_path": str(model_path),
        "initial_config": initial_config.config_id(),
        "configs": [c.config_id() for c in CONFIGS],
        "scale_up_mult": SCALE_UP_MULT,
        "scale_down_mult": SCALE_DOWN_MULT,
        "ema_alpha": round(EMA_ALPHA, 6),
        "cooldown_s": COOLDOWN,
        "measured_throughput": MEASURED_THROUGHPUT,
    })

    if await _async_container_start(container):
        autoscaler.containers[model_name] = container
        autoscaler.current_config[model_name] = initial_config
        autoscaler.last_scale_time[model_name] = autoscaler.clock()
        _log_json("INIT_OK", {
            "model": model_name,
            "config": initial_config.config_id(),
            "container": _container_info(container),
        })
    else:
        logger.error("Failed to start initial container")
        yield
        return

    metrics_task = asyncio.create_task(_metrics_polling_loop())
    streaming_task = asyncio.create_task(_streaming_throughput_loop())
    scaling_task = asyncio.create_task(_background_scaling_loop())

    yield

    metrics_task.cancel()
    streaming_task.cancel()
    scaling_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    try:
        await streaming_task
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
    for model_name in base.get("models", {}):
        ema = _throughput_ema.get(model_name, 0.0)
        config_id = base["models"][model_name].get("config_id", "")
        capacity = MEASURED_THROUGHPUT.get(config_id, 1.0)
        base["models"][model_name]["throughput_ema"] = round(ema, 4)
        base["models"][model_name]["capacity"] = capacity
        base["models"][model_name]["ema_pct"] = (
            round(ema / capacity * 100, 1) if capacity > 0 else 0
        )
        base["models"][model_name]["demand_tps"] = round(ema, 4)
    return base


@app.get("/v1/models")
async def list_models():
    if autoscaler is None:
        return {"models": []}
    return {"models": list(autoscaler.containers.keys())}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if autoscaler is None:
        raise HTTPException(503, "Server not ready")

    container = autoscaler.containers.get(request.model)
    if not container or not container.is_ready:
        raise HTTPException(
            404, "Model '%s' not found or not ready" % request.model
        )

    async with container.lock:
        container.active_requests += 1
        container.total_requests += 1

    req_id = str(uuid.uuid4())[:8]
    req_start = time.time()
    config = autoscaler.current_config.get(request.model)
    config_id = config.config_id() if config else "unknown"

    try:
        prompt_parts = []
        for m in request.messages:
            prompt_parts.append("%s: %s" % (m.role, m.content))
        prompt_text = "\n".join(prompt_parts)

        payload = {
            "prompt": prompt_text,
            "n_predict": request.max_tokens or 256,
            "temperature": request.temperature or 0.7,
            "stream": True,
        }

        endpoint = container.get_endpoint()
        url = "%s/completion" % endpoint

        content_parts = []
        predicted_n = 0
        prompt_n = 0
        timings = {}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    _log_json("REQ_UPSTREAM_ERR", {
                        "req_id": req_id, "status": resp.status,
                        "body": body[:300], "config_id": config_id,
                    })
                    raise HTTPException(
                        resp.status,
                        "Container error: %s" % body[:200],
                    )

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    json_str = line[6:]
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    token_text = chunk.get("content", "")
                    if token_text:
                        content_parts.append(token_text)

                    # Use the "tokens" field for accurate count
                    chunk_tokens = chunk.get("tokens")
                    if chunk_tokens is None:
                        raise RuntimeError(
                            "SSE chunk missing 'tokens' field: %s" % json_str[:200]
                        )
                    n_tok = len(chunk_tokens)
                    if n_tok > 0:
                        predicted_n += n_tok
                        if request.model not in _streaming_token_count:
                            _streaming_token_count[request.model] = 0
                        _streaming_token_count[request.model] += n_tok

                    if chunk.get("stop", False):
                        timings = chunk.get("timings", {})
                        prompt_n = timings.get("prompt_n", 0)
                        actual_n = timings.get("predicted_n", predicted_n)
                        if actual_n != predicted_n:
                            raise RuntimeError(
                                "Token count mismatch: streamed %d vs timings %d"
                                % (predicted_n, actual_n)
                            )
                        break

        content = "".join(content_parts)
        wall_ms = (time.time() - req_start) * 1000
        total_tokens = prompt_n + predicted_n

        prompt_ms = timings.get("prompt_ms", 0)
        prompt_per_second = timings.get("prompt_per_second", 0)
        prompt_per_token_ms = timings.get("prompt_per_token_ms", 0)
        predicted_ms = timings.get("predicted_ms", 0)
        predicted_per_second = timings.get("predicted_per_second", 0)
        predicted_per_token_ms = timings.get("predicted_per_token_ms", 0)

        if total_tokens > 0:
            autoscaler.demand_tracker.record_tokens(
                request.model, total_tokens
            )

        ema = _throughput_ema.get(request.model, 0.0)
        cpt = (
            get_cost_per_token(request.model, config) * 1e6
            if config else 0
        )

        _log_json("REQ_OK", {
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
            "throughput_ema": round(ema, 4),
            "cost_per_token_micro": round(cpt, 4),
            "container": container.container_name,
            "port": container.port,
            "raw_timings": timings,
        })

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
        _log_json("REQ_EXCEPTION", {
            "req_id": req_id, "config_id": config_id,
            "wall_ms": round(wall_ms, 1), "error": str(e)[:300],
        })
        raise HTTPException(
            500, "Internal error: %s" % str(e)[:200]
        )
    finally:
        async with container.lock:
            container.active_requests = max(
                0, container.active_requests - 1
            )
