"""
Cost-Aware LLM Inference Autoscaler
- Selects hardware configuration (CPU or GPU) with lowest cost-per-token
- Dynamically scales based on workload demand (tokens/second)
- Supports CPU (1, 4, 8 cores) and GPU (50%, 100%) configurations
"""
from __future__ import annotations

import asyncio
import json
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
    HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40),
    HardwareConfig(cpu_cores=8, memory="16g", hourly_cost=0.80),
    HardwareConfig(cpu_cores=12, memory="24g", hourly_cost=1.20),
    HardwareConfig(cpu_cores=2, memory="8g", gpu_percentage=50, hourly_cost=1.00),
    HardwareConfig(cpu_cores=2, memory="16g", gpu_percentage=100, hourly_cost=2.00),
]

# Default throughput by config_id (tokens/second)
DEFAULT_THROUGHPUT: Dict[str, float] = {
    "cpu_4": 12.0,
    "cpu_8": 18.0,
    "cpu_12": 22.0,
    "gpu_50": 50.0,
    "gpu_100": 100.0,
}


# ---------------------------------------------------------------------------
# Task 2: Throughput and Cost Functions
# ---------------------------------------------------------------------------

# Optional per-model overrides (model_name -> config_id -> throughput)
MODEL_THROUGHPUT_OVERRIDES: Dict[str, Dict[str, float]] = {}


def get_throughput(model: str, config: HardwareConfig) -> float:
    """Get tokens/second. Checks model-specific overrides first, then defaults."""
    config_id = config.config_id()
    if model in MODEL_THROUGHPUT_OVERRIDES:
        if config_id in MODEL_THROUGHPUT_OVERRIDES[model]:
            return MODEL_THROUGHPUT_OVERRIDES[model][config_id]
    return DEFAULT_THROUGHPUT.get(config_id, 1.0)


def get_cost_per_token(model: str, config: HardwareConfig) -> float:
    """Cost per token = hourly_cost / (throughput * 3600). Returns inf if throughput <= 0."""
    throughput = get_throughput(model, config)
    if throughput <= 0:
        return float('inf')
    return config.hourly_cost / (throughput * 3600)


# ---------------------------------------------------------------------------
# Task 3: DemandTracker Class (Exponential Moving Average)
# ---------------------------------------------------------------------------

DEMAND_WINDOW_SECONDS = 300  # 5-minute EMA span


class DemandTracker:
    """Tracks demand (tokens/second) using an exponential moving average.

    Each call to record_tokens first decays the current EMA to the
    present time, then blends in the instantaneous rate implied by the
    new token batch.  get_demand decays to "now" before returning.

    The ``window_seconds`` parameter controls the EMA span: the half-life
    is ``window_seconds * ln(2)`` and ``alpha = 2 / (window_seconds + 1)``.
    A 300 s span gives a smooth 5-minute moving average.
    """

    def __init__(self, window_seconds: int = DEMAND_WINDOW_SECONDS,
                 clock: Optional[Callable[[], float]] = None):
        self.window_seconds = window_seconds
        self.clock = clock or time.time
        # alpha controls how fast the EMA reacts.  Smaller alpha → smoother.
        self.alpha: float = 2.0 / (window_seconds + 1)
        # Per-model EMA state: (ema_value, last_update_time)
        self._ema: Dict[str, float] = {}
        self._last_time: Dict[str, float] = {}

    def _decay(self, model: str, now: float) -> float:
        """Decay the stored EMA to *now* and return the decayed value."""
        if model not in self._ema:
            return 0.0
        dt = now - self._last_time[model]
        if dt <= 0:
            return self._ema[model]
        # Continuous-time EMA decay: value * (1 - alpha)^dt
        decay = (1.0 - self.alpha) ** dt
        val = self._ema[model] * decay
        self._ema[model] = val
        self._last_time[model] = now
        return val

    def record_tokens(self, model: str, token_count: int) -> None:
        """Record a token usage event and update the EMA."""
        now = self.clock()
        # Decay existing EMA to now
        current = self._decay(model, now)
        # Instantaneous rate contribution: tokens arrive as a burst,
        # so we treat them as token_count tok/s for one "tick".
        self._ema[model] = current + self.alpha * token_count
        self._last_time[model] = now

    def get_demand(self, model: str) -> float:
        """Return the current EMA demand estimate (tokens/second)."""
        now = self.clock()
        return self._decay(model, now)


# ---------------------------------------------------------------------------
# Task 4: CostAwareAutoscaler Class
# ---------------------------------------------------------------------------

COOLDOWN_SECONDS = 300          # 5 minutes between scaling actions
MAX_DRAIN_TIMEOUT_SECONDS = 60  # Max wait for in-flight requests during scaling


class CostAwareAutoscaler:
    """Makes scaling decisions based on cost-per-token optimization."""

    def __init__(
        self,
        configs: List[HardwareConfig],
        cooldown_seconds: float = COOLDOWN_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        models_dir: str = "./models",
    ):
        self.configs = configs
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock or time.time
        self.models_dir = Path(models_dir).resolve()

        self.demand_tracker = DemandTracker(clock=self.clock)
        self.current_config: Dict[str, HardwareConfig] = {}
        self.last_scale_time: Dict[str, float] = {}
        self.containers: Dict[str, Container] = {}
        self.used_ports: set = set()
        self.lock = asyncio.Lock()

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

    def select_optimal_config(self, model: str, demand: float) -> HardwareConfig:
        """Select config with lowest cost_per_token that can handle demand."""
        viable = [c for c in self.configs if get_throughput(model, c) >= demand]
        if not viable:
            # Fall back to highest throughput config
            return max(self.configs, key=lambda c: get_throughput(model, c))
        return min(viable, key=lambda c: get_cost_per_token(model, c))

    def check_scaling(self, model: str) -> Optional[HardwareConfig]:
        """Check if scaling is needed, respecting cooldown."""
        now = self.clock()
        last_scale = self.last_scale_time.get(model, 0)
        if now - last_scale < self.cooldown_seconds:
            return None

        current = self.current_config.get(model)
        demand = self.demand_tracker.get_demand(model)
        optimal = self.select_optimal_config(model, demand)

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

            if not await new_container.start():
                logger.error(f"Failed to start new container for {model}, keeping current")
                return False

            # Swap references
            self.containers[model] = new_container
            self.current_config[model] = new_config
            self.last_scale_time[model] = self.clock()

            logger.info(
                f"Scaled {model} to {new_config.config_id()} "
                f"(cost=${new_config.hourly_cost}/hr)"
            )

            # Drain and stop old container
            if old_container:
                deadline = self.clock() + MAX_DRAIN_TIMEOUT_SECONDS
                while old_container.active_requests > 0 and self.clock() < deadline:
                    logger.info(
                        f"Draining {old_container.container_name}: "
                        f"{old_container.active_requests} active"
                    )
                    await asyncio.sleep(1)
                await old_container.stop()

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
        """Get container for a model, checking scaling first."""
        container = self.containers.get(model_name)
        if not container or not container.is_ready:
            return None

        # Check if scaling is needed
        new_config = self.check_scaling(model_name)
        if new_config:
            logger.info(
                f"Scaling {model_name} from "
                f"{self.current_config.get(model_name, HardwareConfig()).config_id()} "
                f"to {new_config.config_id()}"
            )
            await self.scale_to(model_name, new_config)

        return self.containers.get(model_name)

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
            demand = self.demand_tracker.get_demand(name)
            cost_per_tok = (
                get_cost_per_token(name, config) if config else 0.0
            )
            throughput = get_throughput(name, config) if config else 0.0

            models_status[name] = {
                "config_id": config.config_id() if config else "unknown",
                "container_type": config.container_type if config else "unknown",
                "cpu_cores": config.cpu_cores if config else None,
                "gpu_percentage": config.gpu_percentage if config else None,
                "memory": config.memory if config else None,
                "hourly_cost": config.hourly_cost if config else 0.0,
                "image": config.image if config else "unknown",
                "throughput_tps": round(throughput, 2),
                "demand_tps": round(demand, 4),
                "cost_per_token": round(cost_per_tok, 10),
                "active_requests": container.active_requests,
                "total_requests": container.total_requests,
                "is_ready": container.is_ready,
                "port": container.port,
            }

        return {
            "models": models_status,
            "cooldown_seconds": self.cooldown_seconds,
            "demand_window_seconds": self.demand_tracker.window_seconds,
            "available_configs": [c.config_id() for c in self.configs],
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
    yield
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
                autoscaler.demand_tracker.record_tokens(
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
                    # Try to count tokens from stream chunks
                    if line != 'data: [DONE]':
                        try:
                            chunk = json.loads(line[6:])
                            usage = chunk.get('usage', {})
                            if usage.get('total_tokens'):
                                total_tokens = usage['total_tokens']
                        except (json.JSONDecodeError, KeyError):
                            pass
                    if line == 'data: [DONE]':
                        break

            # Record token usage (estimate if not available from stream)
            if total_tokens > 0:
                autoscaler.demand_tracker.record_tokens(
                    request.model, total_tokens
                )
            else:
                # Estimate based on max_tokens as fallback
                estimated = request.max_tokens or 100
                autoscaler.demand_tracker.record_tokens(
                    request.model, estimated
                )

            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
