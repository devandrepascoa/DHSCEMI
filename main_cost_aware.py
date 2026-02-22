"""
Cost-Aware LLM Inference Autoscaler
- Selects hardware configuration (CPU or GPU) with lowest cost-per-token
- Dynamically scales based on workload demand (tokens/second)
- Supports CPU (1, 4, 8 cores) and GPU (50%, 100%) configurations
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
import uuid
import socket
from pathlib import Path
from typing import Callable, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 1: Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Hardware configuration with pricing for cost-aware autoscaling."""
    cpu_cores: Optional[int] = None
    memory: Optional[str] = None
    gpu_percentage: Optional[int] = None
    hourly_cost: float = 0.0

    def config_id(self) -> str:
        """Return unique string identifier for this config."""
        if self.gpu_percentage:
            return f"gpu_{self.gpu_percentage}"
        return f"cpu_{self.cpu_cores}"

    @property
    def image(self) -> str:
        """Return the correct Docker image for this config."""
        if self.container_type == "gpu":
            return "ghcr.io/ggml-org/llama.cpp:full-cuda"
        return "ghcr.io/ggml-org/llama.cpp:full"

    @property
    def container_type(self) -> str:
        """Return 'gpu' if GPU config, else 'cpu'."""
        return "gpu" if self.gpu_percentage else "cpu"


# Available hardware configurations (ordered from cheapest to most expensive)
HARDWARE_CONFIGS: List[HardwareConfig] = [
    HardwareConfig(cpu_cores=4,  memory="8g",  hourly_cost=0.05),
    HardwareConfig(cpu_cores=12, memory="8g",  hourly_cost=0.12),
    HardwareConfig(cpu_cores=2,  memory="8g",  gpu_percentage=25,  hourly_cost=0.50),
    HardwareConfig(cpu_cores=2,  memory="16g", gpu_percentage=100, hourly_cost=4.00),
]


# ---------------------------------------------------------------------------
# Task 2: Measured Throughputs and Cost Functions
# ---------------------------------------------------------------------------

# Measured aggregate throughput per config (from RTX 3060 benchmarks)
MEASURED_THROUGHPUT: Dict[str, float] = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}

# Scaling thresholds
SCALE_UP_MULT = 0.8           # scale up at 80% of current capacity
SCALE_DOWN_MULT = 0.3         # scale down when <= 30% of current capacity
METRICS_POLL_INTERVAL = 1.0   # poll /metrics every 1s
SCALING_CHECK_INTERVAL = 10   # check scaling every 10s
EMA_ALPHA = 2.0 / (240 + 1)  # ~4min EMA window

# Regex patterns for prometheus metrics from llama.cpp /metrics endpoint
_RE_PREDICTED = re.compile(
    r'^llamacpp:tokens_predicted_total\s+(\d+(?:\.\d+)?)', re.MULTILINE
)
_RE_PROMPT = re.compile(
    r'^llamacpp:prompt_tokens_total\s+(\d+(?:\.\d+)?)', re.MULTILINE
)


def get_cost_per_token(model: str, config: HardwareConfig) -> float:
    """Cost per token = hourly_cost / (throughput * 3600). Returns inf if throughput <= 0."""
    throughput = MEASURED_THROUGHPUT.get(config.config_id(), 1.0)
    if throughput <= 0:
        return float('inf')
    return config.hourly_cost / (throughput * 3600)


# ---------------------------------------------------------------------------
# Task 3: Throughput EMA Tracker (from /metrics polling)
# ---------------------------------------------------------------------------


class ThroughputTracker:
    """Tracks throughput (tokens/second) using an exponential moving average
    fed by polling llama.cpp's /metrics endpoint.

    The EMA uses time-corrected decay: value * (1 - alpha)^dt, where
    alpha = 2 / (window + 1) with a ~4min window.
    """

    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self._ema: Dict[str, float] = {}
        self._last_time: Dict[str, float] = {}
        # /metrics polling state
        self._prev_total_tokens: Dict[str, float] = {}
        self._prev_metrics_time: Dict[str, float] = {}
        # Streaming token counter (incremented in real-time as tokens arrive)
        self._streaming_count: Dict[str, int] = {}
        self._prev_streaming_count: Dict[str, int] = {}
        self._prev_streaming_time: Dict[str, float] = {}
        # After scaling, wait for first token before resuming EMA
        self._waiting_for_first_token: Dict[str, bool] = {}

    def update_ema(self, model: str, tps: float, now: float) -> float:
        """Update throughput EMA and return current value."""
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
        """Return the current EMA value."""
        return self._ema.get(model, 0.0)

    def record_streaming_tokens(self, model: str, count: int) -> None:
        """Increment the streaming token counter."""
        if model not in self._streaming_count:
            self._streaming_count[model] = 0
        self._streaming_count[model] += count

    def reset_model(self, model: str) -> None:
        """Reset all tracking state for a model (after scaling)."""
        self._ema.pop(model, None)
        self._last_time.pop(model, None)
        self._prev_total_tokens.pop(model, None)
        self._prev_metrics_time.pop(model, None)
        self._streaming_count.pop(model, None)
        self._prev_streaming_count.pop(model, None)
        self._prev_streaming_time.pop(model, None)
        self._waiting_for_first_token[model] = True


# ---------------------------------------------------------------------------
# Task 4: CostAwareAutoscaler Class
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = 300          # 5 minutes between scaling actions
MAX_DRAIN_TIMEOUT_SECONDS = 60  # Max wait for in-flight requests during scaling


def select_config(
    current_config: HardwareConfig,
    throughput_ema: float,
    configs_by_cost: List[HardwareConfig],
) -> HardwareConfig:
    """Select config based on throughput EMA vs measured capacity thresholds.

    Scale UP:   throughput_ema >= SCALE_UP_MULT * capacity[current]
    Scale DOWN: throughput_ema <= SCALE_DOWN_MULT * capacity[current]
                AND throughput_ema <= 0.75 * capacity[cheaper]
    """
    current_id = current_config.config_id()
    current_idx = next(
        i for i, c in enumerate(configs_by_cost) if c.config_id() == current_id
    )
    current_capacity = MEASURED_THROUGHPUT.get(current_id, 1.0)

    # Check scale UP
    if throughput_ema >= SCALE_UP_MULT * current_capacity:
        if current_idx + 1 < len(configs_by_cost):
            return configs_by_cost[current_idx + 1]

    # Check scale DOWN: only if underusing current AND cheaper can handle it
    if current_idx > 0:
        cheaper = configs_by_cost[current_idx - 1]
        cheaper_capacity = MEASURED_THROUGHPUT.get(cheaper.config_id(), 1.0)
        if (throughput_ema <= SCALE_DOWN_MULT * current_capacity
                and throughput_ema <= 0.75 * cheaper_capacity):
            return cheaper

    return current_config


class CostAwareAutoscaler:
    """Makes scaling decisions based on throughput EMA vs measured capacity."""

    def __init__(
        self,
        configs: List[HardwareConfig],
        cooldown_seconds: float = COOLDOWN_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        models_dir: str = "./models",
        headroom: float = 0.0,
    ):
        self.configs = configs
        self.configs_by_cost = sorted(configs, key=lambda c: c.hourly_cost)
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock or time.time
        self.models_dir = Path(models_dir).resolve()
        self.headroom = headroom

        self.throughput_tracker = ThroughputTracker()
        self.current_config: Dict[str, HardwareConfig] = {}
        self.last_scale_time: Dict[str, float] = {}
        self.containers: Dict[str, Container] = {}
        self.used_ports: set = set()
        self.lock = asyncio.Lock()
        self.scaling_in_progress: bool = False

    def _get_port(self) -> int:
        """Get an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
            self.used_ports.add(port)
            return port

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Find model file by name."""
        for ext in ['', '.gguf', '.bin']:
            path = self.models_dir / f"{model_name}{ext}"
            if path.exists():
                return path
        # Fuzzy match
        for f in self.models_dir.iterdir():
            if f.is_file() and model_name.lower() in f.name.lower():
                return f
        return None

    def select_optimal_config(self, model: str, ema: float,
                              current: Optional[HardwareConfig] = None) -> HardwareConfig:
        """Select config based on throughput EMA vs measured capacity thresholds."""
        if current is None:
            return self.configs_by_cost[0]
        return select_config(current, ema, self.configs_by_cost)

    def check_scaling(self, model: str) -> Optional[HardwareConfig]:
        """Check if scaling is needed, respecting cooldown."""
        now = self.clock()
        last_scale = self.last_scale_time.get(model, 0)
        if now - last_scale < self.cooldown_seconds:
            return None

        # Don't scale while waiting for first token after previous scale
        if self.throughput_tracker._waiting_for_first_token.get(model, False):
            return None

        current = self.current_config.get(model)
        ema = self.throughput_tracker.get_ema(model)
        optimal = self.select_optimal_config(model, ema, current=current)

        if current is None or optimal.config_id() != current.config_id():
            return optimal
        return None

    async def scale_to(self, model: str, new_config: HardwareConfig) -> bool:
        """Scale a model to a new hardware config with graceful transition."""
        async with self.lock:
            old_container = self.containers.get(model)
            model_path = (
                old_container.model_path if old_container
                else self.get_model_path(model)
            )

            if not model_path:
                logger.error(f"Cannot scale: model path not found for {model}")
                return False

            port = self._get_port()
            new_container = Container(model, model_path, new_config, port)

            # Stop old container first, then start new one
            if old_container:
                await old_container.stop()

            if not await new_container.start():
                logger.error(f"Failed to start new container for {model}")
                # Try to rollback
                if old_container:
                    rollback_port = self._get_port()
                    rollback = Container(model, model_path,
                                         self.current_config.get(model, new_config),
                                         rollback_port)
                    if await rollback.start():
                        self.containers[model] = rollback
                return False

            # Swap references
            self.containers[model] = new_container
            self.current_config[model] = new_config
            # Don't start cooldown yet — wait for first token
            self.throughput_tracker.reset_model(model)

            logger.info(
                f"Scaled {model} to {new_config.config_id()} "
                f"(cost=${new_config.hourly_cost}/hr)"
            )
            return True

    async def initialize(self) -> None:
        """Scan models directory and start containers with cheapest config."""
        logger.info(f"Scanning models in {self.models_dir}")
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
                        f"Started {model_name} on {cheapest.config_id()} "
                        f"(cost=${cheapest.hourly_cost}/hr)"
                    )

    async def get_container(self, model_name: str) -> Optional[Container]:
        """Get container for a model."""
        container = self.containers.get(model_name)
        if not container or not container.is_ready:
            return None
        return container

    async def cleanup(self) -> None:
        """Stop all containers."""
        for container in self.containers.values():
            await container.stop()
        self.containers.clear()

    def get_status(self) -> Dict:
        """Return status information for all models."""
        models_status = {}
        for name, container in self.containers.items():
            config = self.current_config.get(name)
            config_id = config.config_id() if config else "unknown"
            ema = self.throughput_tracker.get_ema(name)
            capacity = MEASURED_THROUGHPUT.get(config_id, 1.0)
            cost_per_tok = (
                get_cost_per_token(name, config) if config else 0.0
            )

            models_status[name] = {
                "config_id": config_id,
                "container_type": config.container_type if config else "unknown",
                "cpu_cores": config.cpu_cores if config else None,
                "gpu_percentage": config.gpu_percentage if config else None,
                "memory": config.memory if config else None,
                "hourly_cost": config.hourly_cost if config else 0.0,
                "image": config.image if config else "unknown",
                "throughput_ema": round(ema, 4),
                "capacity": capacity,
                "ema_pct": round(ema / capacity * 100, 1) if capacity > 0 else 0,
                "scale_up_threshold": round(SCALE_UP_MULT * capacity, 1),
                "cost_per_token": round(cost_per_tok, 10),
                "active_requests": container.active_requests,
                "total_requests": container.total_requests,
                "is_ready": container.is_ready,
                "port": container.port,
            }

        return {
            "models": models_status,
            "cooldown_seconds": self.cooldown_seconds,
            "scaling_in_progress": self.scaling_in_progress,
            "available_configs": [c.config_id() for c in self.configs_by_cost],
            "measured_throughput": MEASURED_THROUGHPUT,
        }


# ---------------------------------------------------------------------------
# Task 5: Container and FastAPI Integration
# ---------------------------------------------------------------------------

class Container:
    """Manages a single Docker container running llama.cpp server."""

    def __init__(self, model_name: str, model_path: Path,
                 config: HardwareConfig, port: int):
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.port = port
        self.container_name = (
            f"llama-{model_name}-{config.config_id()}-{port}"
        )

        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False
        self.lock = asyncio.Lock()

    def _docker_args(self) -> List[str]:
        """Generate docker resource-limit arguments for this config."""
        args: List[str] = []
        if self.config.cpu_cores:
            args.extend(['--cpus', str(self.config.cpu_cores)])
        if self.config.memory:
            args.extend(['--memory', self.config.memory])
        if self.config.gpu_percentage:
            args.extend(['--gpus', 'all', '--privileged'])
        return args

    async def start(self) -> bool:
        """Start the Docker container and wait for it to be ready."""
        # Remove any existing container with same name
        subprocess.run(
            ['docker', 'rm', '-f', self.container_name],
            capture_output=True, check=False,
        )

        threads = self.config.cpu_cores or 1
        parallel = self.config.cpu_cores or 1

        docker_cmd = [
            'docker', 'run', '--rm', '-d',
            '--name', self.container_name,
            '-v', f'{self.model_path.parent}:/models:ro',
            '-p', f'{self.port}:8080',
            *self._docker_args(),
            self.config.image,
            '--server',
            '-m', f'/models/{self.model_path.name}',
            '--host', '0.0.0.0',
            '--port', '8080',
            '--threads', str(threads),
            '--parallel', str(parallel),
            '--metrics',
        ]

        # Add GPU layers flag for GPU configs
        if self.config.gpu_percentage:
            docker_cmd.extend(['--n-gpu-layers', '99'])

        logger.info(f"Starting container: {self.container_name}")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Failed to start container: {result.stderr}")
            return False

        # Wait for container to be ready
        for _ in range(30):  # 30 * 2s = 60s timeout
            if await self._health_check():
                self.is_ready = True
                logger.info(f"Container ready: {self.container_name}")
                return True
            await asyncio.sleep(2)

        logger.error(f"Container failed to become ready: {self.container_name}")
        return False

    async def stop(self) -> None:
        """Stop the Docker container."""
        logger.info(f"Stopping container: {self.container_name}")
        subprocess.run(
            ['docker', 'stop', self.container_name],
            capture_output=True, check=False,
        )
        self.is_ready = False

    async def _health_check(self) -> bool:
        """Check if the container is healthy."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(
                    f"http://localhost:{self.port}/health"
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def get_endpoint(self) -> str:
        """Return the HTTP endpoint for this container."""
        return f"http://localhost:{self.port}"


# ---------------------------------------------------------------------------
# Background loops: streaming throughput, metrics polling, scaling
# ---------------------------------------------------------------------------

async def _poll_metrics(container: Container) -> Optional[Dict]:
    """Poll /metrics endpoint and parse prometheus counters."""
    try:
        url = f"http://localhost:{container.port}/metrics"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()

        predicted = 0.0
        prompt = 0.0
        m = _RE_PREDICTED.search(text)
        if m:
            predicted = float(m.group(1))
        m = _RE_PROMPT.search(text)
        if m:
            prompt = float(m.group(1))

        return {"total_tokens": predicted, "predicted_total": predicted, "prompt_total": prompt}
    except Exception:
        return None


async def _streaming_throughput_loop(scaler: CostAwareAutoscaler) -> None:
    """Compute streaming tok/s every METRICS_POLL_INTERVAL from streaming token counter."""
    tracker = scaler.throughput_tracker
    while True:
        await asyncio.sleep(METRICS_POLL_INTERVAL)
        now = time.time()

        for model_name in list(scaler.containers.keys()):
            container = scaler.containers.get(model_name)
            if not container or not container.is_ready:
                continue

            current_count = tracker._streaming_count.get(model_name, 0)

            if model_name not in tracker._prev_streaming_count:
                tracker._prev_streaming_count[model_name] = current_count
                tracker._prev_streaming_time[model_name] = now
                continue

            prev_count = tracker._prev_streaming_count[model_name]
            prev_time = tracker._prev_streaming_time[model_name]
            dt = now - prev_time

            tracker._prev_streaming_count[model_name] = current_count
            tracker._prev_streaming_time[model_name] = now

            if dt <= 0:
                continue

            delta_tokens = current_count - prev_count
            tps = delta_tokens / dt

            # If waiting for first token after scaling, check streaming counter
            if tracker._waiting_for_first_token.get(model_name, False):
                if delta_tokens > 0:
                    tracker._waiting_for_first_token[model_name] = False
                    scaler.last_scale_time[model_name] = scaler.clock()
                    tracker.update_ema(model_name, tps, now)
                    logger.info(
                        f"First token after scale for {model_name}, "
                        f"config={scaler.current_config.get(model_name, HARDWARE_CONFIGS[0]).config_id()}"
                    )
                continue

            tracker.update_ema(model_name, tps, now)


async def _background_scaling_loop(scaler: CostAwareAutoscaler) -> None:
    """Check scaling every SCALING_CHECK_INTERVAL using throughput EMA."""
    while True:
        await asyncio.sleep(SCALING_CHECK_INTERVAL)
        if scaler.scaling_in_progress:
            continue

        for model_name in list(scaler.containers.keys()):
            container = scaler.containers.get(model_name)
            current_config = scaler.current_config.get(model_name)
            if not container or not current_config:
                continue

            if scaler.throughput_tracker._waiting_for_first_token.get(model_name, False):
                continue

            ema = scaler.throughput_tracker.get_ema(model_name)
            current_id = current_config.config_id()
            capacity = MEASURED_THROUGHPUT.get(current_id, 1.0)

            logger.info(
                f"[DEMAND_CHECK] {model_name}: config={current_id} "
                f"ema={ema:.1f} capacity={capacity} "
                f"ema_pct={ema / capacity * 100:.1f}% "
                f"scale_up_at={SCALE_UP_MULT * capacity:.1f}"
            )

            new_config = scaler.check_scaling(model_name)
            if new_config is None:
                continue

            old_config_id = current_id
            new_config_id = new_config.config_id()
            logger.info(
                f"[SCALING] {model_name}: {old_config_id} -> {new_config_id} "
                f"(ema={ema:.1f})"
            )

            scaler.scaling_in_progress = True
            try:
                await scaler.scale_to(model_name, new_config)
            finally:
                scaler.scaling_in_progress = False


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
# FastAPI Application
# ---------------------------------------------------------------------------

autoscaler: Optional[CostAwareAutoscaler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global autoscaler
    autoscaler = CostAwareAutoscaler(configs=HARDWARE_CONFIGS)
    await autoscaler.initialize()

    streaming_task = asyncio.create_task(_streaming_throughput_loop(autoscaler))
    scaling_task = asyncio.create_task(_background_scaling_loop(autoscaler))

    yield

    streaming_task.cancel()
    scaling_task.cancel()
    try:
        await streaming_task
    except asyncio.CancelledError:
        pass
    try:
        await scaling_task
    except asyncio.CancelledError:
        pass
    await autoscaler.cleanup()


app = FastAPI(title="Cost-Aware LLM Autoscaler", lifespan=lifespan)


@app.get("/health")
async def health():
    ready = sum(1 for c in autoscaler.containers.values() if c.is_ready)
    return {
        "status": "healthy" if ready > 0 else "down",
        "ready_containers": ready,
        "models": list(autoscaler.containers.keys()),
    }


@app.get("/status")
async def status():
    return autoscaler.get_status()


@app.get("/v1/models")
async def list_models():
    return {"models": list(autoscaler.containers.keys())}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    container = await autoscaler.get_container(request.model)
    if not container:
        raise HTTPException(404, f"Model '{request.model}' not found or not ready")

    # Track active request
    async with container.lock:
        container.active_requests += 1
        container.total_requests += 1

    try:
        if request.stream:
            return StreamingResponse(
                _stream_completion(request, container),
                media_type="text/event-stream",
            )
        else:
            return await _non_stream_completion(request, container)
    finally:
        async with container.lock:
            container.active_requests = max(0, container.active_requests - 1)


async def _non_stream_completion(
    request: ChatCompletionRequest, container: Container
) -> ChatCompletionResponse:
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{container.get_endpoint()}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                raise HTTPException(resp.status, "Container error")

            result = await resp.json()

            choices = []
            for i, choice in enumerate(result.get('choices', [])):
                msg = choice.get('message', {})
                choices.append(ChatCompletionChoice(
                    index=i,
                    message={
                        "role": msg.get('role', 'assistant'),
                        "content": msg.get('content', ''),
                    },
                    finish_reason=choice.get('finish_reason'),
                ))

            # Record token usage for demand tracking
            usage = result.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            if total_tokens > 0:
                autoscaler.throughput_tracker.record_streaming_tokens(
                    request.model, total_tokens
                )

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=usage,
            )


async def _stream_completion(
    request: ChatCompletionRequest, container: Container
) -> AsyncGenerator[str, None]:
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": True,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{container.get_endpoint()}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                yield f"data: {json.dumps({'error': 'Container error'})}\n\n"
                return

            total_tokens = 0
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    yield f"{line}\n\n"
                    if line != 'data: [DONE]':
                        try:
                            chunk = json.loads(line[6:])
                            # Count tokens from stream chunks
                            usage = chunk.get('usage', {})
                            if usage.get('total_tokens'):
                                total_tokens = usage['total_tokens']
                            # Increment streaming counter for each chunk
                            choices = chunk.get('choices', [])
                            for choice in choices:
                                delta = choice.get('delta', {})
                                if delta.get('content'):
                                    autoscaler.throughput_tracker.record_streaming_tokens(
                                        request.model, 1
                                    )
                        except (json.JSONDecodeError, KeyError):
                            pass
                    if line == 'data: [DONE]':
                        break

            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
