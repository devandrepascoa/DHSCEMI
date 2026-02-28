"""
Cost-Aware LLM Inference Autoscaler — Scaling Demo Server.

Scaling logic: per-request tok/s from SSE streaming.
  Each request measures tokens/wall_time from the SSE stream,
  feeds into a per_request_tps_ema.

  - Scale UP:   per_request_tps_ema < MIN_TPS_THRESHOLD
  - Scale DOWN: per_request_tps_ema >= MIN_TPS_THRESHOLD
                AND lower config can serve current concurrency above threshold

Usage:
    uv run uvicorn main_cost_aware:app --port <PORT>
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import time
import uuid
import logging
from typing import Callable, Dict, List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Hardware configuration with pricing for cost-aware autoscaling."""
    cpu_cores: Optional[int] = None
    memory: Optional[str] = None
    gpu_percentage: Optional[int] = None
    hourly_cost: float = 0.0
    parallel_slots: Optional[int] = None

    def config_id(self) -> str:
        if self.gpu_percentage:
            return "gpu_%d" % self.gpu_percentage
        return "cpu_%d" % self.cpu_cores

    @property
    def image(self) -> str:
        if self.container_type == "gpu":
            return "ghcr.io/ggml-org/llama.cpp:full-cuda"
        return "ghcr.io/ggml-org/llama.cpp:full"

    @property
    def container_type(self) -> str:
        return "gpu" if self.gpu_percentage else "cpu"


# Available hardware configurations — loaded from hardware_configs.json
_CONFIG_PATH = Path(__file__).parent / "hardware_configs.json"


def _load_hardware_configs() -> tuple[List[HardwareConfig], Dict[str, float]]:
    """Load hardware configs and measured throughput from JSON file."""
    with open(_CONFIG_PATH) as f:
        data = json.load(f)
    configs = [
        HardwareConfig(
            cpu_cores=c.get("cpu_cores"),
            memory=c.get("memory"),
            gpu_percentage=c.get("gpu_percentage"),
            hourly_cost=c.get("hourly_cost", 0.0),
            parallel_slots=c.get("parallel_slots"),
        )
        for c in data["configs"]
    ]
    throughput = data.get("measured_throughput", {})
    return configs, throughput


HARDWARE_CONFIGS, MEASURED_THROUGHPUT = _load_hardware_configs()

METRICS_POLL_INTERVAL = 1.0
SCALING_CHECK_INTERVAL = 10
_EMA_WINDOW = int(os.environ.get("E2E_EMA_WINDOW", "240"))
EMA_ALPHA = 2.0 / (_EMA_WINDOW + 1)  # default ~4min EMA window

# Per-request scaling thresholds
MIN_TPS_THRESHOLD = float(os.environ.get("E2E_MIN_TPS", "10.0"))
SCALE_DOWN_CONCURRENCY = float(os.environ.get("E2E_SCALE_DOWN_CONCURRENCY", "5.0"))

_RE_PREDICTED = re.compile(
    r'^llamacpp:tokens_predicted_total\s+(\d+(?:\.\d+)?)', re.MULTILINE
)
_RE_PROMPT = re.compile(
    r'^llamacpp:prompt_tokens_total\s+(\d+(?:\.\d+)?)', re.MULTILINE
)
_RE_PREDICTED_TPS = re.compile(
    r'^llamacpp:predicted_tokens_seconds\s+(\d+(?:\.\d+)?)', re.MULTILINE
)


def get_cost_per_token(model: str, config: HardwareConfig) -> float:
    """Cost per token = hourly_cost / (throughput * 3600)."""
    throughput = MEASURED_THROUGHPUT.get(config.config_id(), 1.0)
    if throughput <= 0:
        return float('inf')
    return config.hourly_cost / (throughput * 3600)


# ---------------------------------------------------------------------------
# Pydantic models for API
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=512, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# ---------------------------------------------------------------------------
# ThroughputTracker (used by tests and e2e servers)
# ---------------------------------------------------------------------------

class ThroughputTracker:
    """Tracks throughput using an EMA fed by streaming token counts."""

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._ema: Dict[str, float] = {}
        self._last_time: Dict[str, float] = {}
        self._prev_total_tokens: Dict[str, float] = {}
        self._prev_metrics_time: Dict[str, float] = {}
        self._streaming_count: Dict[str, int] = {}
        self._prev_streaming_count: Dict[str, int] = {}
        self._prev_streaming_time: Dict[str, float] = {}
        self._waiting_for_first_token: Dict[str, bool] = {}

    def update_ema(self, model: str, tps: float, now: float) -> float:
        if model not in self._ema:
            self._ema[model] = tps
            self._last_time[model] = now
            return tps
        dt = now - self._last_time.get(model, now)
        if dt <= 0:
            return self._ema.get(model, 0.0)
        decay = (1.0 - self.alpha) ** dt
        self._ema[model] = self._ema[model] * decay + (1.0 - decay) * tps
        self._last_time[model] = now
        return self._ema[model]

    def get_ema(self, model: str) -> float:
        return self._ema.get(model, 0.0)

    def record_streaming_tokens(self, model: str, count: int) -> None:
        if model not in self._streaming_count:
            self._streaming_count[model] = 0
        self._streaming_count[model] += count

    def reset_model(self, model: str) -> None:
        self._ema.pop(model, None)
        self._last_time.pop(model, None)
        self._prev_total_tokens.pop(model, None)
        self._prev_metrics_time.pop(model, None)
        self._streaming_count.pop(model, None)
        self._prev_streaming_count.pop(model, None)
        self._prev_streaming_time.pop(model, None)
        self._waiting_for_first_token[model] = True


# ---------------------------------------------------------------------------
# DemandTracker (legacy, used by scaling demo lifecycle)
# ---------------------------------------------------------------------------

class DemandTracker:
    """Simple token-rate tracker using a sliding window."""

    def __init__(self, window_seconds: int = 180):
        self.window_seconds = window_seconds
        self._events: Dict[str, deque] = {}

    def record_tokens(self, model: str, count: int) -> None:
        now = time.time()
        if model not in self._events:
            self._events[model] = deque()
        self._events[model].append((now, count))
        self._trim(model, now)

    def get_rate(self, model: str) -> float:
        now = time.time()
        self._trim(model, now)
        events = self._events.get(model, deque())
        if not events:
            return 0.0
        total = sum(c for _, c in events)
        span = now - events[0][0]
        if span <= 0:
            return 0.0
        return total / span

    def _trim(self, model: str, now: float) -> None:
        events = self._events.get(model)
        if not events:
            return
        cutoff = now - self.window_seconds
        while events and events[0][0] < cutoff:
            events.popleft()


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class Container:
    """Manages a single Docker container running llama.cpp server."""

    def __init__(self, model_name: str, model_path: Path,
                 config: HardwareConfig, port: int):
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.port = port
        self.container_name = "llama-%s-%s-%d" % (
            model_name, config.config_id(), port
        )
        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False
        self.lock = asyncio.Lock()

    def _docker_args(self) -> List[str]:
        args: List[str] = []
        if self.config.cpu_cores:
            args.extend(['--cpus', str(self.config.cpu_cores)])
        if self.config.memory:
            args.extend(['--memory', self.config.memory])
        if self.config.gpu_percentage:
            args.extend(['--gpus', 'all', '--privileged'])
        return args

    async def start(self) -> bool:
        subprocess.run(
            ['docker', 'rm', '-f', self.container_name],
            capture_output=True, check=False,
        )
        threads = self.config.cpu_cores or 1
        parallel = self.config.parallel_slots or self.config.cpu_cores or 1
        docker_cmd = [
            'docker', 'run', '--rm', '-d',
            '--name', self.container_name,
            '-v', '%s:/models:ro' % str(self.model_path.parent),
            '-p', '%d:8080' % self.port,
            *self._docker_args(),
            self.config.image,
            '--server',
            '-m', '/models/%s' % self.model_path.name,
            '--host', '0.0.0.0',
            '--port', '8080',
            '--threads', str(threads),
            '--parallel', str(parallel),
            '--metrics',
        ]
        if self.config.gpu_percentage:
            docker_cmd.extend(['--n-gpu-layers', '99'])

        logger.info("Starting container: %s" % self.container_name)
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to start container: %s" % result.stderr)
            return False

        for _ in range(30):
            if await self._health_check():
                self.is_ready = True
                logger.info("Container ready: %s" % self.container_name)
                return True
            await asyncio.sleep(2)

        logger.error("Container failed to become ready: %s" % self.container_name)
        return False

    async def stop(self) -> None:
        logger.info("Stopping container: %s" % self.container_name)
        subprocess.run(
            ['docker', 'stop', self.container_name],
            capture_output=True, check=False,
        )
        self.is_ready = False

    async def _health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(
                    "http://localhost:%d/health" % self.port
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def get_endpoint(self) -> str:
        return "http://localhost:%d" % self.port


# ---------------------------------------------------------------------------
# select_config_per_request (per-request tok/s scaling decision)
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = 300


def select_config_per_request(
    current_config: HardwareConfig,
    per_request_tps_ema: float,
    active_requests_ema: float,
    configs_by_cost: Optional[List[HardwareConfig]] = None,
) -> HardwareConfig:
    """Select config based on per-request tok/s and active concurrency.

    Scale UP:   per_request_tps_ema < MIN_TPS_THRESHOLD
    Scale DOWN: per_request_tps_ema >= MIN_TPS_THRESHOLD
                AND lower config can still serve current concurrency above threshold

    configs_by_throughput is optional; defaults to HARDWARE_CONFIGS sorted by
    measured throughput.
    """
    if configs_by_cost is None:
        configs_by_cost = sorted(HARDWARE_CONFIGS, key=lambda c: MEASURED_THROUGHPUT.get(c.config_id(), 0))

    current_id = current_config.config_id()
    current_idx = next(
        i for i, c in enumerate(configs_by_cost) if c.config_id() == current_id
    )

    # Scale UP: per-request speed below minimum
    if per_request_tps_ema < MIN_TPS_THRESHOLD:
        if current_idx + 1 < len(configs_by_cost):
            return configs_by_cost[current_idx + 1]

    # Scale DOWN: speed acceptable, lower config can handle current concurrency
    #   Use 1.5x safety margin because actual per-request degradation under
    #   concurrency is worse than the linear capacity/concurrency estimate.
    if current_idx > 0:
        if per_request_tps_ema >= MIN_TPS_THRESHOLD:
            lower = configs_by_cost[current_idx - 1]
            lower_capacity = MEASURED_THROUGHPUT.get(lower.config_id(), 1.0)
            concurrency = max(active_requests_ema, 1.0)
            estimated_per_req = lower_capacity / concurrency
            if estimated_per_req >= MIN_TPS_THRESHOLD * 1.5:
                return lower

    return current_config


# ---------------------------------------------------------------------------
# CostAwareAutoscaler
# ---------------------------------------------------------------------------

class CostAwareAutoscaler:
    """Makes scaling decisions based on throughput EMA vs measured capacity."""

    def __init__(
        self,
        configs: List[HardwareConfig],
        cooldown_seconds: float = COOLDOWN_SECONDS,
        cooldown_down_seconds: Optional[float] = None,
        clock: Optional[Callable[[], float]] = None,
        models_dir: str = "./models",
        headroom: float = 0.0,
    ):
        self.configs = configs
        self.configs_by_cost = sorted(configs, key=lambda c: MEASURED_THROUGHPUT.get(c.config_id(), 0))
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_down_seconds = cooldown_down_seconds if cooldown_down_seconds is not None else cooldown_seconds
        self.clock = clock or time.time
        self.models_dir = Path(models_dir).resolve()
        self.headroom = headroom

        self.throughput_tracker = ThroughputTracker()
        self.demand_tracker = DemandTracker()
        self.current_config: Dict[str, HardwareConfig] = {}
        self.last_scale_time: Dict[str, float] = {}
        self.last_scale_direction: Dict[str, str] = {}  # "up" or "down"
        self.containers: Dict[str, Container] = {}
        self.used_ports: set = set()
        self.lock = asyncio.Lock()
        self.scaling_in_progress: bool = False

    def _get_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
            self.used_ports.add(port)
            return port

    def get_model_path(self, model_name: str) -> Optional[Path]:
        for ext in ['', '.gguf', '.bin']:
            path = self.models_dir / ("%s%s" % (model_name, ext))
            if path.exists():
                return path
        for f in self.models_dir.iterdir():
            if f.is_file() and model_name.lower() in f.name.lower():
                return f
        return None

    def select_optimal_config(
        self, model: str, per_request_tps_ema: float,
        active_requests_ema: float,
        current: Optional[HardwareConfig] = None,
    ) -> HardwareConfig:
        if current is None:
            return self.configs_by_cost[0]
        return select_config_per_request(
            current, per_request_tps_ema, active_requests_ema, self.configs_by_cost,
        )

    def check_scaling(self, model: str,
                      per_request_tps_ema: float = 0.0,
                      active_requests_ema: float = 0.0) -> Optional[HardwareConfig]:
        now = self.clock()
        last_scale = self.last_scale_time.get(model, 0)
        if now - last_scale < self.cooldown_seconds:
            return None
        current = self.current_config.get(model)
        optimal = self.select_optimal_config(
            model, per_request_tps_ema, active_requests_ema, current=current,
        )
        if current is None or optimal.config_id() != current.config_id():
            return optimal
        return None

    async def scale_to(self, model: str, new_config: HardwareConfig) -> bool:
        async with self.lock:
            old_container = self.containers.get(model)
            model_path = (
                old_container.model_path if old_container
                else self.get_model_path(model)
            )
            if not model_path:
                logger.error("Cannot scale: model path not found for %s" % model)
                return False

            port = self._get_port()
            new_container = Container(model, model_path, new_config, port)

            if old_container:
                await old_container.stop()

            if not await new_container.start():
                logger.error("Failed to start new container for %s" % model)
                if old_container:
                    rollback_port = self._get_port()
                    rollback = Container(
                        model, model_path,
                        self.current_config.get(model, new_config),
                        rollback_port,
                    )
                    if await rollback.start():
                        self.containers[model] = rollback
                return False

            self.containers[model] = new_container
            self.current_config[model] = new_config
            self.throughput_tracker.reset_model(model)
            logger.info(
                "Scaled %s to %s (cost=$%s/hr)"
                % (model, new_config.config_id(), new_config.hourly_cost)
            )
            return True

    async def initialize(self) -> None:
        logger.info("Scanning models in %s" % self.models_dir)
        cheapest = min(self.configs, key=lambda c: c.hourly_cost)
        for model_file in self.models_dir.iterdir():
            if model_file.suffix.lower() in ['.gguf', '.bin']:
                model_name = model_file.stem
                port = self._get_port()
                container = Container(model_name, model_file, cheapest, port)
                if await container.start():
                    self.containers[model_name] = container
                    self.current_config[model_name] = cheapest
                    self.last_scale_time[model_name] = self.clock()
                    logger.info(
                        "Started %s on %s (cost=$%s/hr)"
                        % (model_name, cheapest.config_id(), cheapest.hourly_cost)
                    )

    async def get_container(self, model_name: str) -> Optional[Container]:
        container = self.containers.get(model_name)
        if not container or not container.is_ready:
            return None
        return container

    async def cleanup(self) -> None:
        for container in self.containers.values():
            await container.stop()
        self.containers.clear()

    def get_status(self) -> Dict:
        models_status = {}
        for name, container in self.containers.items():
            config = self.current_config.get(name)
            config_id = config.config_id() if config else "unknown"
            capacity = MEASURED_THROUGHPUT.get(config_id, 1.0)
            cost_per_tok = get_cost_per_token(name, config) if config else 0.0

            models_status[name] = {
                "config_id": config_id,
                "container_type": config.container_type if config else "unknown",
                "cpu_cores": config.cpu_cores if config else None,
                "gpu_percentage": config.gpu_percentage if config else None,
                "memory": config.memory if config else None,
                "hourly_cost": config.hourly_cost if config else 0.0,
                "image": config.image if config else "unknown",
                "capacity": capacity,
                "min_tps_threshold": MIN_TPS_THRESHOLD,
                "scale_down_concurrency": SCALE_DOWN_CONCURRENCY,
                "cost_per_token": round(cost_per_tok, 10),
                "active_requests": container.active_requests,
                "total_requests": container.total_requests,
                "is_ready": container.is_ready,
                "port": container.port,
            }

        return {
            "models": models_status,
            "cooldown_seconds": self.cooldown_seconds,
            "cooldown_down_seconds": self.cooldown_down_seconds,
            "scaling_in_progress": self.scaling_in_progress,
            "available_configs": [c.config_id() for c in self.configs_by_cost],
            "measured_throughput": MEASURED_THROUGHPUT,
        }


# ===========================================================================
# Scaling Demo Server
# ===========================================================================

CONFIGS: List[HardwareConfig] = HARDWARE_CONFIGS
CONFIGS_BY_COST = sorted(CONFIGS, key=lambda c: MEASURED_THROUGHPUT.get(c.config_id(), 0))

COOLDOWN = int(os.environ.get("E2E_COOLDOWN", "300"))
COOLDOWN_DOWN = int(os.environ.get("E2E_COOLDOWN_DOWN", os.environ.get("E2E_COOLDOWN", "300")))
DEMAND_WINDOW = 180

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
MODEL_NAME = os.environ.get("E2E_MODEL_NAME", "")

autoscaler: Optional[CostAwareAutoscaler] = None
server_start_time: float = 0.0
scaling_in_progress: bool = False

# Per-request tok/s EMA (scaling signal)
_per_request_tps_ema: Dict[str, float] = {}
_per_request_tps_ema_time: Dict[str, float] = {}
_per_request_waiting_first: Dict[str, bool] = {}

# Active requests EMA (for scale-down decisions)
_active_requests_ema: Dict[str, float] = {}
_active_requests_ema_time: Dict[str, float] = {}

# Per-request streaming counters: req_id -> {model, count, start_time, first_token_time}
_active_request_counters: Dict[str, Dict] = {}

# Track when each model last had active requests (for scale-down hysteresis)
RECENT_ACTIVITY_WINDOW = float(os.environ.get("E2E_RECENT_ACTIVITY_WINDOW", "30.0"))
_last_active_time: Dict[str, float] = {}

# /metrics polling state (for logging only)
_prev_total_tokens: Dict[str, float] = {}
_prev_metrics_time: Dict[str, float] = {}
_last_metrics_tps: Dict[str, float] = {}
_last_predicted_tps_gauge: Dict[str, float] = {}

# Streaming token counter (for logging/status display)
_streaming_token_count: Dict[str, int] = {}
_prev_streaming_count: Dict[str, int] = {}
_prev_streaming_time: Dict[str, float] = {}
_last_streaming_tps: Dict[str, float] = {}


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
        "parallel": config.parallel_slots or (config.cpu_cores or 1),
        "threads": config.cpu_cores or 1,
        "n_gpu_layers": 99 if config.gpu_percentage else 0,
        "docker_flags": container._docker_args(),
        "measured_throughput_tps": MEASURED_THROUGHPUT.get(cid, 0),
        "cost_per_token_micro": round(cpt, 4),
    }


def _log_json(tag: str, data: Dict) -> None:
    print("[SERVER] [%s] %s" % (tag, json.dumps(data)), flush=True)


# ---------------------------------------------------------------------------
# Async container lifecycle (used by demo server)
# ---------------------------------------------------------------------------

async def _async_container_start(container: Container) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "docker", "rm", "-f", container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()

    threads = container.config.cpu_cores or 1
    parallel = container.config.parallel_slots or threads

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
        "--parallel", str(parallel),
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


def _update_per_request_ema(model: str, req_tps: float, now: float) -> float:
    """Update per-request tok/s EMA when a request completes."""
    if model not in _per_request_tps_ema:
        _per_request_tps_ema[model] = req_tps
        _per_request_tps_ema_time[model] = now
        return req_tps

    dt = now - _per_request_tps_ema_time.get(model, now)
    if dt <= 0:
        return _per_request_tps_ema.get(model, 0.0)

    decay = (1.0 - EMA_ALPHA) ** dt
    _per_request_tps_ema[model] = (
        _per_request_tps_ema[model] * decay + (1.0 - decay) * req_tps
    )
    _per_request_tps_ema_time[model] = now
    return _per_request_tps_ema[model]


def _update_active_requests_ema(model: str, active: float, now: float) -> float:
    """Update active requests EMA."""
    if model not in _active_requests_ema:
        _active_requests_ema[model] = active
        _active_requests_ema_time[model] = now
        return active

    dt = now - _active_requests_ema_time.get(model, now)
    if dt <= 0:
        return _active_requests_ema.get(model, 0.0)

    decay = (1.0 - EMA_ALPHA) ** dt
    _active_requests_ema[model] = (
        _active_requests_ema[model] * decay + (1.0 - decay) * active
    )
    _active_requests_ema_time[model] = now
    return _active_requests_ema[model]


async def _streaming_counter_loop() -> None:
    """Sample active requests EMA and compute per-request tok/s every tick."""
    while True:
        await asyncio.sleep(METRICS_POLL_INTERVAL)
        now = time.time()

        for model_name in list(autoscaler.containers.keys()):
            container = autoscaler.containers.get(model_name)
            if not container or not container.is_ready:
                continue

            # Sample active requests EMA
            _update_active_requests_ema(
                model_name, float(container.active_requests), now,
            )

            # Track last time this model had active requests
            if container.active_requests > 0:
                _last_active_time[model_name] = now

            # Compute per-request tok/s from active request counters.
            # Use total_tokens / wall_time for ALL active requests (including
            # those queued in llama.cpp waiting for a parallel slot). This
            # ensures queued requests contribute 0 tok/s, properly reflecting
            # that the config is overloaded.
            per_req_tps_values = []
            any_tokens = False
            for req_id, info in list(_active_request_counters.items()):
                if info.get("model") != model_name:
                    continue
                current_count = info.get("count", 0)
                req_start = info.get("start_time", now)
                wall_time = now - req_start
                if wall_time > 0.5:  # skip requests that just started
                    tps = current_count / wall_time
                    per_req_tps_values.append(tps)
                if current_count > 0:
                    any_tokens = True

            if per_req_tps_values:
                avg_per_req_tps = sum(per_req_tps_values) / len(per_req_tps_values)
                _update_per_request_ema(model_name, avg_per_req_tps, now)
                # If waiting for first request after scaling, mark done
                if _per_request_waiting_first.get(model_name, False) and any_tokens:
                    _per_request_waiting_first[model_name] = False
                    _per_request_waiting_first[model_name] = False
                    # Seed EMA at threshold so we start neutral
                    _per_request_tps_ema[model_name] = MIN_TPS_THRESHOLD
                    _per_request_tps_ema_time[model_name] = now
                    autoscaler.last_scale_time[model_name] = autoscaler.clock()
                    _log_json("FIRST_TOKEN_AFTER_SCALE", {
                        "model": model_name,
                        "elapsed": round(now - server_start_time, 3),
                        "config_id": autoscaler.current_config.get(
                            model_name, CONFIGS[0]
                        ).config_id(),
                        "avg_per_req_tps": round(avg_per_req_tps, 2),
                        "active_requests": len(per_req_tps_values),
                        "seeded_ema": MIN_TPS_THRESHOLD,
                    })

            # Streaming tok/s for logging only
            current_count = _streaming_token_count.get(model_name, 0)

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
            _last_metrics_tps[model_name] = 0.0


# ---------------------------------------------------------------------------
# Background scaling loop
# ---------------------------------------------------------------------------

async def _background_scaling_loop() -> None:
    """Check scaling every SCALING_CHECK_INTERVAL using per-request tok/s EMA."""
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

            if _per_request_waiting_first.get(model_name, False):
                continue

            now = time.time()
            pr_ema = _per_request_tps_ema.get(model_name, MIN_TPS_THRESHOLD)
            ar_ema = _active_requests_ema.get(model_name, 0.0)
            current_id = current_config.config_id()
            streaming_tps = _last_streaming_tps.get(model_name, 0.0)
            predicted_tps_gauge = _last_predicted_tps_gauge.get(model_name, 0.0)

            last_active = _last_active_time.get(model_name, 0.0)
            recently_active = (now - last_active) < RECENT_ACTIVITY_WINDOW

            _log_json("DEMAND_CHECK", {
                "model": model_name,
                "elapsed": round(now - server_start_time, 3),
                "config_id": current_id,
                "per_request_tps_ema": round(pr_ema, 4),
                "active_requests_ema": round(ar_ema, 4),
                "active_requests_effective": max(round(ar_ema, 4), container.active_requests),
                "min_tps_threshold": MIN_TPS_THRESHOLD,
                "scale_down_concurrency": SCALE_DOWN_CONCURRENCY,
                "streaming_tps": round(streaming_tps, 4),
                "predicted_tps_gauge": round(predicted_tps_gauge, 4),
                "active_requests": container.active_requests,
                "recently_active": recently_active,
                "seconds_since_active": round(now - last_active, 1),
            })

            # Check cooldown (directional: scale-down uses longer cooldown)
            last_scale = autoscaler.last_scale_time.get(model_name, 0)
            time_since_scale = now - last_scale

            # Determine what direction the next scale would be.
            # We need to tentatively compute the optimal config to know direction.
            # But first, apply the minimum cooldown (scale-up cooldown) as a gate.
            if time_since_scale < autoscaler.cooldown_seconds:
                continue

            # Prevent scale-down if there were active requests recently.
            # Fast configs (gpu_25/gpu_100) finish requests quickly, creating
            # windows where active_requests=0 and ar_ema is low, but load
            # is still ongoing. Only allow scale-up when recently active.
            effective_ar = max(ar_ema, float(container.active_requests))
            if recently_active and container.active_requests == 0:
                # Block scale-down by inflating concurrency so viability check fails
                effective_ar = max(effective_ar, 1e6)

            optimal = select_config_per_request(
                current_config, pr_ema, effective_ar,
            )
            if optimal.config_id() == current_id:
                continue

            new_config = optimal
            old_config_id = current_id
            new_config_id = new_config.config_id()

            # Directional cooldown: scale-down uses longer cooldown
            # Scale-down = moving to a lower-throughput tier (lower index in sorted list)
            new_config_idx = next(
                (i for i, c in enumerate(CONFIGS_BY_COST) if c.config_id() == new_config_id),
                -1,
            )
            current_config_idx = next(
                (i for i, c in enumerate(CONFIGS_BY_COST) if c.config_id() == current_id),
                -1,
            )
            is_scale_down = new_config_idx < current_config_idx
            if is_scale_down and time_since_scale < autoscaler.cooldown_down_seconds:
                continue

            _log_json("SCALING_START", {
                "event": "scaling_start",
                "timestamp": now,
                "elapsed": round(now - server_start_time, 3),
                "model": model_name,
                "from_config": old_config_id,
                "to_config": new_config_id,
                "direction": "down" if is_scale_down else "up",
                "per_request_tps_ema": round(pr_ema, 4),
                "active_requests_ema": round(ar_ema, 4),
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

                if old_container:
                    await _async_container_stop(old_container)

                success = await _async_container_start(new_container)
                if success:
                    autoscaler.containers[model_name] = new_container
                    autoscaler.current_config[model_name] = new_config
                    autoscaler.last_scale_direction[model_name] = (
                        "down" if is_scale_down else "up"
                    )
                    _per_request_waiting_first[model_name] = True
                    # Reset all tracking state
                    _per_request_tps_ema.pop(model_name, None)
                    _per_request_tps_ema_time.pop(model_name, None)
                    _active_requests_ema.pop(model_name, None)
                    _active_requests_ema_time.pop(model_name, None)
                    _last_active_time.pop(model_name, None)
                    _prev_total_tokens.pop(model_name, None)
                    _prev_metrics_time.pop(model_name, None)
                    _last_metrics_tps.pop(model_name, None)
                    _last_predicted_tps_gauge.pop(model_name, None)
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
        cooldown_down_seconds=COOLDOWN_DOWN,
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
        "min_tps_threshold": MIN_TPS_THRESHOLD,
        "scale_down_concurrency": SCALE_DOWN_CONCURRENCY,
        "ema_alpha": round(EMA_ALPHA, 6),
        "cooldown_s": COOLDOWN,
        "measured_throughput": MEASURED_THROUGHPUT,
    })

    if await _async_container_start(container):
        autoscaler.containers[model_name] = container
        autoscaler.current_config[model_name] = initial_config
        autoscaler.last_scale_time[model_name] = autoscaler.clock()
        _per_request_waiting_first[model_name] = True
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
    streaming_task = asyncio.create_task(_streaming_counter_loop())
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


app = FastAPI(title="Cost-Aware LLM Autoscaler", lifespan=lifespan)


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
    base["min_tps_threshold"] = MIN_TPS_THRESHOLD
    base["scale_down_concurrency"] = SCALE_DOWN_CONCURRENCY
    base["recent_activity_window"] = RECENT_ACTIVITY_WINDOW
    for model_name in base.get("models", {}):
        pr_ema = _per_request_tps_ema.get(model_name, 0.0)
        ar_ema = _active_requests_ema.get(model_name, 0.0)
        streaming_tps = _last_streaming_tps.get(model_name, 0.0)
        config_id = base["models"][model_name].get("config_id", "")
        capacity = MEASURED_THROUGHPUT.get(config_id, 1.0)
        base["models"][model_name]["per_request_tps_ema"] = round(pr_ema, 4)
        base["models"][model_name]["active_requests_ema"] = round(ar_ema, 4)
        base["models"][model_name]["throughput_ema"] = round(streaming_tps, 4)
        base["models"][model_name]["capacity"] = capacity
        base["models"][model_name]["demand_tps"] = round(streaming_tps, 4)
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

        # Register per-request counter for background loop
        _active_request_counters[req_id] = {
            "model": request.model, "count": 0, "start_time": time.time(),
        }

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
                        # Increment per-request counter for background loop
                        _active_request_counters[req_id]["count"] += n_tok

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

        # Clean up per-request counter
        _active_request_counters.pop(req_id, None)

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

        pr_ema = _per_request_tps_ema.get(request.model, 0.0)
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
            "per_request_tps_ema": round(pr_ema, 4),
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
        # Clean up per-request counter if still present
        _active_request_counters.pop(req_id, None)
        async with container.lock:
            container.active_requests = max(
                0, container.active_requests - 1
            )
