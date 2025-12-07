from __future__ import annotations

import json
import subprocess
import asyncio
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime
import socket
import statistics

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

model_configs: Dict[str, Dict[str, Any]] = {}
container_manager: 'ContainerManager' = None

# Simple defaults used when a container has no throughput history yet.
DEFAULT_TOKENS_PER_SECOND_BY_TYPE: Dict[str, float] = {
    'cpu': 6.0,
    'gpu': 120.0,
}
MIN_EFFECTIVE_TOKENS_PER_SECOND: float = 1.0
QUEUE_PENALTY_FACTOR: float = 0.5
MIN_PREDICTED_LATENCY_SECONDS: float = 0.5
MAX_PREDICTED_LATENCY_SECONDS: float = 600.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global container_manager
    container_manager = ContainerManager()
    logger.info("Initializing container manager and model clusters...")
    await initialize_all_model_clusters()
    await container_manager.start_background_container_management()
    logger.info("Container initialization completed")

    yield

    logger.info("Shutting down containers...")
    await container_manager.cleanup_all_containers()
    await container_manager.stop_background_container_management()


app = FastAPI(
    title="llama.cpp OpenAI Proxy",
    version="1.0.0",
    debug=True,
    lifespan=lifespan
)


@dataclass
class ContainerConfig:
    cpu_cores: Optional[float] = None
    memory: Optional[str] = None
    gpu_percentage: Optional[int] = None

    def __post_init__(self):

        if self.gpu_percentage and (self.cpu_cores or self.memory):
            raise ValueError("Cannot mix GPU and CPU/memory configs")

    def __str__(self):
        cpu_str = f"cpu{self.cpu_cores}" if self.cpu_cores else "cpu"
        memory_str = f"mem{self.memory}" if self.memory else "mem"
        gpu_str = f"gpu{self.gpu_percentage}" if self.gpu_percentage else "nogpu"
        return f"{cpu_str}_{memory_str}_{gpu_str}"

    @property
    def container_type(self) -> str:
        return "gpu" if self.gpu_percentage else "cpu"

    @property
    def image(self) -> str:
        return "ghcr.io/ggml-org/llama.cpp:full-cuda" if self.container_type == "gpu" else "ghcr.io/ggml-org/llama.cpp:full"

    def to_docker_args(self) -> List[str]:

        args = []

        if self.container_type == "cpu":

            if self.cpu_cores is not None:
                args.extend(['--cpus', str(self.cpu_cores)])
            if self.memory is not None:
                args.extend(['--memory', self.memory])
        elif self.container_type == "gpu":

            if self.gpu_percentage is not None:
                args.extend(["-e", f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={self.gpu_percentage}"])

            args.extend(['--gpus', 'all'])

        return args


AVAILABLE_CONFIGS: List[ContainerConfig] = [
    ContainerConfig(cpu_cores=1.0, memory="4g"),
    ContainerConfig(cpu_cores=2.0, memory="8g"),
    ContainerConfig(cpu_cores=4.0, memory="16g"),
    ContainerConfig(cpu_cores=8.0, memory="32g"),
    ContainerConfig(gpu_percentage=50),
    ContainerConfig(gpu_percentage=100),
]


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=100, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


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


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class BenchmarkLoadRequest(BaseModel):
    model: str
    tokens_per_hour: float
    pulses: int = 10
    reset: bool = False
    override_cooldown: Optional[float] = None


class ContainerInstance:
    def __init__(self, model_name: str, model_path: Path, container_name: str, port: int, config: ContainerConfig):
        self.model_name = model_name
        self.model_path = model_path
        self.container_name = container_name
        self.port = port
        self.config = config
        self.process = None
        self._is_ready = False
        self.last_used = datetime.now()
        self.request_count = 0
        self.lock = asyncio.Lock()

        self.processing_times = deque(maxlen=20)
        self.token_processing_times = deque(maxlen=20)
        self.active_requests = 0
        self.queue_start_times = {}
        self.avg_processing_time = 5.0
        self.tokens_per_second = 10.0
        self.metrics_lock = asyncio.Lock()

        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60
        self.is_circuit_open = False
        self.last_scale_evaluation = time.time()

    async def is_ready(self) -> bool:
        if self.is_circuit_open:

            if self.last_failure_time and (time.time() - self.last_failure_time) > self.circuit_breaker_timeout:
                logger.info(f"Circuit breaker timeout passed for {self.container_name}, attempting recovery")
                self.is_circuit_open = False
                self.failure_count = 0
            else:
                return False

        if self._is_ready and not self.is_circuit_open:
            return True

        async with self.lock:
            if self._is_ready and not self.is_circuit_open:
                return True

            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"http://localhost:{self.port}/health") as response:
                        if response.status == 200:
                            self._is_ready = True
                            self.failure_count = 0
                            self.is_circuit_open = False
                            return True

                self._record_failure()
                return False

            except Exception:
                self._record_failure()
                return False

    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        self._is_ready = False
        if self.failure_count >= self.circuit_breaker_threshold:
            self.is_circuit_open = True

    async def restart(self):
        subprocess.run(['docker', 'stop', self.container_name], capture_output=True, check=False)
        await self.start_container()

    async def start_container(self):
        docker_cmd = [
            'docker', 'run', '--rm', '-d',
            '--name', self.container_name,
            '-v', f'{self.model_path.parent}:/models:ro',
            '-p', f'{self.port}:8080',
        ]

        docker_cmd.extend(self.config.to_docker_args())

        docker_cmd.extend([
            self.config.image,
            '--server',
            '-m', f'/models/{self.model_path.name}',
            '--host', '0.0.0.0',
            '--port', '8080',
        ])

        if self.config.container_type == "cpu" and self.config.cpu_cores:
            threads = max(1, int(self.config.cpu_cores))
            docker_cmd.extend(['--threads', str(threads)])

        try:

            subprocess.run(['docker', 'rm', '-f', self.container_name], capture_output=True)

            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self._record_failure()
                return

            for _ in range(6):
                if await self.is_ready():
                    self._is_ready = True
                    return
                await asyncio.sleep(2)
            self._record_failure()

        except Exception:
            self._record_failure()

    async def stop(self):
        subprocess.run(['docker', 'stop', self.container_name], capture_output=True, check=False)
        self._is_ready = False

    def get_endpoint(self) -> str:
        return f"http://localhost:{self.port}"

    async def record_processing_time(self, duration: float, tokens: int = 0):
        async with self.metrics_lock:
            self.processing_times.append(duration)
            if tokens > 0:
                self.token_processing_times.append((tokens, duration))
            if self.processing_times:
                self.avg_processing_time = statistics.mean(self.processing_times)
            if self.token_processing_times and sum(t[1] for t in self.token_processing_times) > 0:
                self.tokens_per_second = sum(t[0] for t in self.token_processing_times) / sum(
                    t[1] for t in self.token_processing_times)

    async def estimate_processing_time(self, estimated_tokens: int = 100) -> float:
        estimated_tokens = max(estimated_tokens, 1)

        async with self.metrics_lock:
            has_samples = len(self.token_processing_times) > 0
            active_requests = self.active_requests
            tokens_per_second = self.tokens_per_second if has_samples and self.tokens_per_second > 0 else 0.0

        if tokens_per_second <= 0:
            fallback_tps = MIN_EFFECTIVE_TOKENS_PER_SECOND
            if container_manager and container_manager.workload_metrics:
                fallback_tps = container_manager.workload_metrics.get_tokens_per_second_default(
                    self.model_name,
                    self.config
                )
            tokens_per_second = fallback_tps

        tokens_per_second = max(tokens_per_second, MIN_EFFECTIVE_TOKENS_PER_SECOND)

        base_time = estimated_tokens / tokens_per_second
        predicted_time = base_time * (1 + QUEUE_PENALTY_FACTOR * active_requests)
        predicted_time = max(predicted_time, MIN_PREDICTED_LATENCY_SECONDS)
        predicted_time = min(predicted_time, MAX_PREDICTED_LATENCY_SECONDS)
        return predicted_time

    async def get_load_score(self, estimated_tokens: int = 100) -> float:
        if not self._is_ready:
            return float('inf')
        estimated_time = await self.estimate_processing_time(estimated_tokens)
        return estimated_time + (self.request_count * 2.0)


class WorkloadMetrics:
    """Minimal workload tracker used for autoscaling decisions."""

    def __init__(self, token_threshold: int = 10000, request_window: int = 60):
        self.token_threshold = token_threshold
        self.request_window = request_window
        self.token_events = defaultdict(lambda: deque(maxlen=256))
        self.default_tps_by_type = DEFAULT_TOKENS_PER_SECOND_BY_TYPE.copy()
        self.current_container_config: Dict[str, ContainerConfig] = {}
        self.current_hardware_type: Dict[str, str] = {}
        self.available_configs: List[ContainerConfig] = AVAILABLE_CONFIGS.copy()
        self.scale_up_threshold = 0.85
        self.scale_down_threshold = 0.30
        self.scaling_cooldown = 120.0
        self.last_scaling_event: Dict[str, float] = {}

    def record_token_usage(self, model_name: str, token_count: int) -> None:
        now = time.time()
        events = self.token_events[model_name]
        events.append((now, token_count))
        cutoff = now - self.request_window
        while events and events[0][0] <= cutoff:
            events.popleft()

    def get_tokens_per_hour(self, model_name: str) -> float:
        events = self.token_events.get(model_name)
        if not events:
            return 0.0
        total_tokens = sum(tokens for _, tokens in events)
        return (total_tokens / self.request_window) * 3600.0

    def get_tokens_per_second_default(self, model_name: str, config: 'ContainerConfig') -> float:
        if config.container_type == 'cpu':
            per_core = self.default_tps_by_type.get('cpu', MIN_EFFECTIVE_TOKENS_PER_SECOND)
            cores = config.cpu_cores or 1.0
            return max(per_core * cores, MIN_EFFECTIVE_TOKENS_PER_SECOND)
        gpu_base = self.default_tps_by_type.get('gpu', MIN_EFFECTIVE_TOKENS_PER_SECOND)
        percentage = (config.gpu_percentage or 100) / 100.0
        return max(gpu_base * percentage, MIN_EFFECTIVE_TOKENS_PER_SECOND)

    def get_config_capacity_tokens_per_hour(self, config: ContainerConfig) -> float:
        return self.get_tokens_per_second_default("", config) * 3600.0

    def get_current_container_config(self, model_name: str) -> ContainerConfig:
        if model_name not in self.current_container_config:
            self.current_container_config[model_name] = self.available_configs[0]
        return self.current_container_config[model_name]

    def update_container_config(self, model_name: str, config: ContainerConfig) -> None:
        self.current_container_config[model_name] = config
        self.current_hardware_type[model_name] = config.container_type

    def get_config_index(self, config: ContainerConfig) -> int:
        for idx, candidate in enumerate(self.available_configs):
            if str(candidate) == str(config):
                return idx
        return 0

    def get_next_config(self, current: ContainerConfig) -> Optional[ContainerConfig]:
        idx = self.get_config_index(current)
        if idx < len(self.available_configs) - 1:
            return self.available_configs[idx + 1]
        return None

    def get_previous_config(self, current: ContainerConfig) -> Optional[ContainerConfig]:
        idx = self.get_config_index(current)
        if idx > 0:
            return self.available_configs[idx - 1]
        return None

    def select_optimal_config(self, model_name: str) -> ContainerConfig:
        current = self.get_current_container_config(model_name)
        tph = self.get_tokens_per_hour(model_name)
        capacity = self.get_config_capacity_tokens_per_hour(current)
        now = time.time()
        desired = current
        cooldown_ok = now - self.last_scaling_event.get(model_name, 0.0) >= self.scaling_cooldown

        if capacity and tph > capacity * self.scale_up_threshold:
            next_config = self.get_next_config(current)
            if next_config and cooldown_ok:
                desired = next_config
        elif capacity and tph < capacity * self.scale_down_threshold:
            prev_config = self.get_previous_config(current)
            if prev_config and cooldown_ok:
                desired = prev_config

        if str(desired) != str(current):
            self.last_scaling_event[model_name] = now
            self.update_container_config(model_name, desired)

        return self.get_current_container_config(model_name)

    def should_use_gpu(self, model_name: str) -> bool:
        config = self.get_current_container_config(model_name)
        return config.container_type == 'gpu'

    def get_workload_stats(self, model_name: str) -> Dict[str, Any]:
        config = self.get_current_container_config(model_name)
        return {
            "tokens_per_hour": self.get_tokens_per_hour(model_name),
            "current_config": str(config),
            "current_hardware": config.container_type,
            "token_threshold": self.token_threshold,
        }


class DecisionLayer:

    def __init__(self, workload_metrics: WorkloadMetrics):
        self.workload_metrics = workload_metrics
        self.container_manager: Optional['ContainerManager'] = None
        self.available_configs = AVAILABLE_CONFIGS
        self.max_containers_per_model = 3
        self.scale_out_latency_threshold = 30.0
        self.scale_in_latency_threshold = 15.0
        self.scale_cooldown_seconds = 120.0
        self._latency_history: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=3))
        self._last_scale_action: Dict[Tuple[str, str], float] = defaultdict(float)

    def set_container_manager(self, manager: 'ContainerManager') -> None:
        self.container_manager = manager

    def choose_container_config(self, model_name: str) -> ContainerConfig:
        desired = self.workload_metrics.select_optimal_config(model_name)
        return desired

    async def get_best_container(self, model_name: str, config: ContainerConfig, estimated_tokens: int = 100) -> \
    Optional[ContainerInstance]:
        if not self.container_manager or model_name not in self.container_manager.container_pools:
            return None
        containers = self.container_manager.container_pools[model_name]
        ready = [c for c in containers if c._is_ready]
        if not ready:
            return None
        matches = [c for c in ready if str(c.config) == str(config)]
        pool = matches if matches else ready
        load_scores = await asyncio.gather(*[c.get_load_score(estimated_tokens) for c in pool])
        best_idx = load_scores.index(min(load_scores))
        return pool[best_idx]

    async def should_spawn_new_container(self, model_name: str, config: ContainerConfig,
                                         estimated_tokens: int = 100) -> bool:
        if not self.container_manager:
            return False
        if model_name not in self.container_manager.container_pools:
            return True
        containers = self.container_manager.container_pools[model_name]
        if len(containers) >= self.max_containers_per_model:
            return False
        ready = [c for c in containers if c._is_ready]
        if not ready:
            return True
        matches = [c for c in ready if str(c.config) == str(config)]
        if not matches:
            return True
        predictions = await self.container_manager.get_processing_time_predictions(model_name, estimated_tokens, config)
        relevant = [p['prediction'] for p in predictions]
        if not relevant:
            return True
        worst_case = max(relevant)
        key = (model_name, str(config))
        history = self._latency_history[key]
        if worst_case <= self.scale_out_latency_threshold:
            history.clear()
            return False
        history.append(worst_case)
        now = time.time()
        if now - self._last_scale_action.get(key, 0.0) < self.scale_cooldown_seconds:
            return False
        if len(history) == history.maxlen and all(v > self.scale_out_latency_threshold for v in history):
            self._last_scale_action[key] = now
            return True
        return False


class ContainerManager:

    def __init__(self, models_dir: str = "./models", containers_per_model: int = 1,
                 token_threshold: int = 10000, request_window: int = 60):
        self.models_dir = Path(models_dir).resolve()
        self.containers_per_model = containers_per_model
        self.max_containers_per_model = 2
        self.used_ports = set()
        self.lock = asyncio.Lock()
        self.workload_metrics = WorkloadMetrics(token_threshold, request_window)
        self.decision_layer = DecisionLayer(self.workload_metrics)
        self.decision_layer.set_container_manager(self)
        self.metrics_tracker = MetricsTracker()
        self.container_management_running = False
        self.container_management_task = None
        self.container_pools: Dict[str, List['ContainerInstance']] = {}
        self.scale_in_history: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=3))
        self.last_scale_in_action: Dict[Tuple[str, str], float] = defaultdict(float)
        self.scale_in_evaluation_interval = 3600.0

    async def start_background_container_management(self):
        """Start the background container management coroutine"""
        if self.container_management_running:
            return

        self.container_management_running = True
        self.container_management_task = asyncio.create_task(self._background_container_manager())
        logger.info("Started background container management system")

    async def stop_background_container_management(self):
        """Stop the background container management coroutine"""
        self.container_management_running = False
        if self.container_management_task:
            self.container_management_task.cancel()
            try:
                await self.container_management_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped background container management system")

    async def _background_container_manager(self):
        """Background coroutine that continuously evaluates metrics and manages containers"""
        logger.info("Background container manager started")

        while self.container_management_running:
            try:

                for model_name in list(self.container_pools.keys()):
                    await self._evaluate_scale_in(model_name)

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                logger.info("Background container manager cancelled")
                break
            except Exception as e:
                logger.error(f"Error in background container manager: {e}")
                await asyncio.sleep(5)

        logger.info("Background container manager stopped")

    async def _evaluate_scale_in(self, model_name: str) -> None:
        if model_name not in self.container_pools:
            return

        containers = self.container_pools[model_name]
        ready_containers = [c for c in containers if c._is_ready]
        if len(ready_containers) <= 1:
            return

        now = time.time()
        scale_in_threshold = self.decision_layer.scale_in_latency_threshold
        cooldown = self.decision_layer.scale_cooldown_seconds

        for container in ready_containers:
            if container.active_requests > 0 or container.queue_start_times:
                continue
            if now - getattr(container, "last_scale_evaluation", 0.0) < self.scale_in_evaluation_interval:
                continue

            container.last_scale_evaluation = now
            key = (model_name, str(container.config))
            history = self.scale_in_history[key]

            predicted_latency = await container.estimate_processing_time(100)
            if predicted_latency >= scale_in_threshold:
                history.clear()
                continue

            history.append(predicted_latency)
            if len(history) < history.maxlen:
                continue

            last_action = self.last_scale_in_action.get(key, 0.0)
            if now - last_action < cooldown:
                continue

            if len([c for c in ready_containers if c is not container]) == 0:
                continue

            await container.stop()
            if container in self.container_pools.get(model_name, []):
                self.container_pools[model_name].remove(container)
            self.last_scale_in_action[key] = now
            history.clear()
            logger.info(
                "Scaled in %s by stopping %s after sustained low latency (%.2fs)",
                model_name,
                container.container_name,
                predicted_latency,
            )
            break

    def ensure_models_dir(self):

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Optional[Path]:

        self.ensure_models_dir()

        model_path = self.models_dir / model_name
        if model_path.exists():
            return model_path

        for ext in ['.gguf', '.bin']:
            model_path = self.models_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path

        for file in self.models_dir.iterdir():
            if file.is_file() and model_name.lower() in file.name.lower():
                return file

        return None

    def _get_available_port(self, min_port: int = 8081, max_port: int = 65535) -> int:

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                s.bind(('', 0))
                port = s.getsockname()[1]
                s.close()

                self.used_ports.add(port)
                return port

        except OSError:

            raise RuntimeError(f"Error looking for an available port")

    async def initialize_model_cluster(self, model_name: str, model_path: Path):

        if model_name not in self.container_pools:
            self.container_pools[model_name] = []

        cpu_config = ContainerConfig(cpu_cores=1.0, memory="4g")
        cpu_containers = [
            c for c in self.container_pools[model_name]
            if str(c.config) == str(cpu_config)
        ]

        for i in range(max(0, self.containers_per_model - len(cpu_containers))):
            port = self._get_available_port()
            container_name = f"llama-cluster-{model_name}-cpu-{port}"

            instance = ContainerInstance(model_name, model_path, container_name, port, cpu_config)
            await instance.start_container()

            if instance._is_ready:
                self.container_pools[model_name].append(instance)
                logger.info(f""
                            f"Added CPU container {container_name} to pool for {model_name}")
                logger.info(f"Container pools: {self.container_pools} self {self}")
            else:
                logger.error(f"Failed to add CPU container {container_name} to pool")

    async def spawn_container(self, model_name: str, model_path: Path, config: ContainerConfig) -> Optional[
        ContainerInstance]:

        if model_name not in self.container_pools:
            self.container_pools[model_name] = []

        container_type_suffix = "cpu" if config.container_type == "cpu" else "gpu"
        port = self._get_available_port()
        container_name = f"llama-cluster-{model_name}-{container_type_suffix}-{port}"

        instance = ContainerInstance(model_name, model_path, container_name, port, config)
        await instance.start_container()

        if instance._is_ready:
            self.container_pools[model_name].append(instance)
            logger.info(f"Added {config.container_type} container {container_name} to pool for {model_name}")
            return instance
        else:
            logger.error(f"Failed to add {config.container_type} container {container_name} to pool")
            return None

    async def get_available_container(self, model_name: str, estimated_tokens: int = 100) -> Optional[
        ContainerInstance]:
        if model_name not in self.container_pools:
            return None

        self.workload_metrics.record_token_usage(model_name, estimated_tokens)

        config = self.decision_layer.choose_container_config(model_name)

        if await self.decision_layer.should_spawn_new_container(model_name, config, estimated_tokens):
            model_path = self.get_model_path(model_name)
            if model_path:
                logger.info(f"Spawning new container for {model_name} due to high processing time")
                new_container = await self.spawn_container(model_name, model_path, config)
                if new_container:
                    logger.info(f"Successfully spawned new container {new_container.container_name}")
                    return new_container

        return await self.decision_layer.get_best_container(model_name, config, estimated_tokens)

    async def get_processing_time_predictions(
            self,
            model_name: str,
            estimated_tokens: int = 100,
            config: Optional[ContainerConfig] = None,
    ) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        containers = self.container_pools.get(model_name, [])
        for container in containers:
            if not container._is_ready:
                continue
            if config and str(container.config) != str(config):
                continue
            predicted_time = await container.estimate_processing_time(estimated_tokens)
            predictions.append(
                {
                    "container": container,
                    "prediction": predicted_time,
                    "active_requests": container.active_requests,
                    "config": container.config,
                }
            )
        return predictions

    async def cleanup_all_containers(self):
        """Gracefully cleanup all containers with proper error handling"""
        cleanup_tasks = []

        for model_name, containers in self.container_pools.items():
            for container in containers:
                cleanup_tasks.append(self._cleanup_single_container(container))

        if cleanup_tasks:

            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error during container cleanup: {result}")

        self.container_pools.clear()

    async def _cleanup_single_container(self, container: ContainerInstance):
        """Cleanup a single container with timeout"""
        try:

            process = await asyncio.create_subprocess_exec(
                'docker', 'stop', '-t', '10', container.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)

            if process.returncode != 0:
                logger.warning(f"Docker stop failed for {container.container_name}: {stderr.decode()}")

        except asyncio.TimeoutError:
            logger.error(f"Timeout stopping container {container.container_name}, forcing removal")

            try:
                await asyncio.create_subprocess_exec(
                    'docker', 'rm', '-f', container.container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            except Exception as e:
                logger.error(f"Failed to force remove container {container.container_name}: {e}")
        except Exception as e:
            logger.error(f"Error stopping container {container.container_name}: {e}")


async def initialize_all_model_clusters():
    logger.info("Scanning for models and initializing clusters...")

    model_files = []
    for file in container_manager.models_dir.iterdir():
        if file.is_file() and file.suffix.lower() in ['.gguf', '.bin']:
            model_files.append(file)
    model_files = sorted(model_files, key=lambda f: f.name)

    if not model_files:
        logger.warning("No model files found in models directory")
        return

    initialization_tasks = []
    for model_file in model_files:
        model_name = model_file.stem
        logger.info(f"Initializing cluster for model: {model_name}")
        task = container_manager.initialize_model_cluster(model_name, model_file)
        initialization_tasks.append(task)

    results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

    for i, (model_file, result) in enumerate(zip(model_files, results)):
        model_name = model_file.stem
        if isinstance(result, Exception):
            logger.error(f"Failed to initialize cluster for {model_name}: {result}")
        else:
            container_count = len(container_manager.container_pools.get(model_name, []))
            ready_count = len([c for c in container_manager.container_pools.get(model_name, []) if c._is_ready])
            logger.info(f"Model {model_name}: {container_count} containers, {ready_count} ready")


@dataclass
class ModelMetrics:
    model_name: str
    config_type: str

    total_requests: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    time_to_first_token: List[float] = field(default_factory=list)

    prompt_processing_ms: List[float] = field(default_factory=list)
    predicted_processing_ms: List[float] = field(default_factory=list)
    prompt_tokens: List[int] = field(default_factory=list)
    predicted_tokens: List[int] = field(default_factory=list)

    tokens_per_second: List[float] = field(default_factory=list)
    prompt_processing_throughput: List[float] = field(default_factory=list)
    token_generation_throughput: List[float] = field(default_factory=list)
    request_durations: List[float] = field(default_factory=list)

    error_count: int = 0
    last_updated: float = field(default_factory=lambda: time.time())

    def record_request(self, tokens: int, duration_seconds: float, time_to_first: Optional[float] = None,
                       prompt_ms: Optional[float] = None, predicted_ms: Optional[float] = None,
                       prompt_tok: Optional[int] = None, predicted_tok: Optional[int] = None):

        self.total_requests += 1
        self.total_tokens += tokens
        self.total_time_seconds += duration_seconds
        self.last_updated = time.time()

        self.request_durations.append(duration_seconds)

        if time_to_first is not None:
            self.time_to_first_token.append(time_to_first)

        if prompt_ms is not None:
            self.prompt_processing_ms.append(prompt_ms)

        if predicted_ms is not None:
            self.predicted_processing_ms.append(predicted_ms)

        if prompt_tok is not None:
            self.prompt_tokens.append(prompt_tok)

        if predicted_tok is not None:
            self.predicted_tokens.append(predicted_tok)

        if duration_seconds > 0:
            tps = tokens / duration_seconds
            self.tokens_per_second.append(tps)

        if prompt_ms is not None and prompt_tok is not None and prompt_ms > 0:
            prompt_tps = prompt_tok / (prompt_ms / 1000.0)
            self.prompt_processing_throughput.append(prompt_tps)

        if predicted_ms is not None and predicted_tok is not None and predicted_ms > 0:
            token_tps = predicted_tok / (predicted_ms / 1000.0)
            self.token_generation_throughput.append(token_tps)

    def record_error(self):
        self.error_count += 1
        self.last_updated = time.time()

    @property
    def average_throughput(self) -> float:
        if not self.tokens_per_second:
            return 0.0
        return sum(self.tokens_per_second) / len(self.tokens_per_second)

    @property
    def average_time_to_first_token(self) -> float:

        if not self.time_to_first_token:
            return 0.0
        return sum(self.time_to_first_token) / len(self.time_to_first_token)

    @property
    def average_prompt_processing_throughput(self) -> float:

        if not self.prompt_processing_throughput:
            return 0.0
        return sum(self.prompt_processing_throughput) / len(self.prompt_processing_throughput)

    @property
    def average_token_generation_throughput(self) -> float:

        if not self.token_generation_throughput:
            return 0.0
        return sum(self.token_generation_throughput) / len(self.token_generation_throughput)

    @property
    def error_rate(self) -> float:

        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:

        return {
            "model_name": self.model_name,
            "config_type": self.config_type,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "average_throughput": self.average_throughput,
            "average_time_to_first_token": self.average_time_to_first_token,
            "average_prompt_processing_throughput": self.average_prompt_processing_throughput,
            "average_token_generation_throughput": self.average_token_generation_throughput,
            "error_rate": self.error_rate,
            "last_updated": self.last_updated,
            "total_prompt_tokens": sum(self.prompt_tokens),
            "total_predicted_tokens": sum(self.predicted_tokens),
            "total_prompt_ms": sum(self.prompt_processing_ms),
            "total_predicted_ms": sum(self.predicted_processing_ms),
            "prompt_processing_ms_per_token": (sum(self.prompt_processing_ms) / sum(self.prompt_tokens)) if sum(
                self.prompt_tokens) > 0 else 0.0,
            "predicted_processing_ms_per_token": (
                    sum(self.predicted_processing_ms) / sum(self.predicted_tokens)) if sum(
                self.predicted_tokens) > 0 else 0.0,
            "recent_tokens_per_second": self.tokens_per_second[-10:],
            "recent_time_to_first": self.time_to_first_token[-10:],
            "recent_prompt_processing_throughput": self.prompt_processing_throughput[-10:],
            "recent_token_generation_throughput": self.token_generation_throughput[-10:],
        }


class MetricsTracker:

    def __init__(self):
        self.metrics: Dict[str, ModelMetrics] = {}
        self.lock = asyncio.Lock()

    def _get_key(self, model_name: str, config: ContainerConfig) -> str:

        return f"{model_name}_{str(config)}"

    async def record_request(
            self,
            model_name: str,
            config: ContainerConfig,
            tokens: int,
            duration_seconds: float,
            time_to_first: Optional[float] = None,
            prompt_ms: Optional[float] = None,
            predicted_ms: Optional[float] = None,
            prompt_tok: Optional[int] = None,
            predicted_tok: Optional[int] = None
    ):

        async with self.lock:
            key = self._get_key(model_name, config)

            if key not in self.metrics:
                self.metrics[key] = ModelMetrics(
                    model_name=model_name,
                    config_type=str(config)
                )

            self.metrics[key].record_request(
                tokens, duration_seconds, time_to_first,
                prompt_ms, predicted_ms, prompt_tok, predicted_tok
            )

    async def record_error(self, model_name: str, config: ContainerConfig):

        async with self.lock:
            key = self._get_key(model_name, config)

            if key not in self.metrics:
                self.metrics[key] = ModelMetrics(
                    model_name=model_name,
                    config_type=str(config)
                )

            self.metrics[key].record_error()

    async def get_metrics(self, model_name: str, config: ContainerConfig) -> Dict[str, Any]:

        key = self._get_key(model_name, config)

        async with self.lock:
            if key not in self.metrics:
                return {"error": "No metrics found for this model and configuration"}

            return self.metrics[key].to_dict()

    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:

        async with self.lock:
            return {
                key: metrics.to_dict()
                for key, metrics in self.metrics.items()
            }

    async def get_model_metrics(self, model_name: str) -> Dict[str, Dict[str, Any]]:

        async with self.lock:
            return {
                key: metrics.to_dict()
                for key, metrics in self.metrics.items()
                if metrics.model_name == model_name
            }


async def stream_chat_completion_with_error_handling(request: ChatCompletionRequest, container: ContainerInstance,
                                                     request_id: str, request_start_time: float) -> AsyncGenerator[
    str, None]:
    try:
        async for chunk in stream_chat_completion(request, container):
            yield chunk

        processing_time = time.time() - request_start_time
        await container.record_processing_time(processing_time, 100)
        container_manager.workload_metrics.record_token_usage(request.model, 100)
    except Exception as e:
        container._record_failure()
        error_payload = {"error": f"Stream interrupted: {str(e)}"}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"


async def stream_chat_completion(request: ChatCompletionRequest, container: ContainerInstance) -> AsyncGenerator[
    str, None]:
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    payload = {
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": True,
        "stop": request.stop
    }

    total_tokens_streamed = 0

    try:
        endpoint = container.get_endpoint()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    await container_manager.metrics_tracker.record_error(
                        request.model, container.config
                    )
                    raise HTTPException(status_code=response.status, detail="Container error")

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            result = json.loads(data)

                            usage = result.get('usage', {})
                            prompt_tokens = usage.get('prompt_tokens', 0)
                            completion_tokens = usage.get('completion_tokens', 0)
                            total_tokens_streamed = max(total_tokens_streamed, prompt_tokens + completion_tokens)

                            prompt_ms = 0

                            if 'timings' in result:
                                timings = result['timings']
                                prompt_ms = timings.get('prompt_ms', 0)
                                predicted_ms = timings.get('predicted_ms', 0)
                                prompt_tokens = timings.get('prompt_n', prompt_tokens)
                                completion_tokens = timings.get('predicted_n', completion_tokens)
                                total_tokens_streamed = max(total_tokens_streamed, prompt_tokens + completion_tokens)

                            yield f"data: {data}\n\n"
                        except json.JSONDecodeError:
                            continue
                yield "data: [DONE]\n\n"

    except Exception as e:
        await container_manager.metrics_tracker.record_error(
            request.model, container.config
        )
        raise HTTPException(status_code=500, detail=str(e))


async def non_streaming_chat_completion(request: ChatCompletionRequest,
                                        container: ContainerInstance) -> ChatCompletionResponse:
    start_time = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{container.get_endpoint()}/v1/chat/completions",
                    json=request.dict(),
                    timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    await container_manager.metrics_tracker.record_error(request.model, container.config)
                    raise HTTPException(status_code=response.status, detail="Container error")

                result = await response.json()

                end_time = time.time()
                total_duration = end_time - start_time

                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)

                prompt_ms = 0
                predicted_ms = total_duration * 1000

                if 'timings' in result:
                    timings = result['timings']
                    prompt_ms = timings.get('prompt_ms', 0)
                    predicted_ms = timings.get('predicted_ms', total_duration * 1000)
                    prompt_tokens = timings.get('prompt_n', prompt_tokens)
                    completion_tokens = timings.get('predicted_n', completion_tokens)

                time_to_first = total_duration

                await container_manager.metrics_tracker.record_request(
                    request.model,
                    container.config,
                    completion_tokens,
                    total_duration,
                    time_to_first,
                    prompt_ms,
                    predicted_ms,
                    prompt_tokens,
                    completion_tokens
                )

                choices = []
                for i, choice_data in enumerate(result.get('choices', [])):
                    message = choice_data.get('message', {})
                    choices.append(ChatCompletionChoice(
                        index=i,
                        message={
                            "role": message.get('role', 'assistant'),
                            "content": message.get('content', '')
                        },
                        finish_reason=choice_data.get('finish_reason')
                    ))

                return ChatCompletionResponse(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    model=request.model,
                    choices=choices,
                    usage=usage
                )

    except Exception as e:
        await container_manager.metrics_tracker.record_error(
            request.model, container.config
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    model_path = container_manager.get_model_path(request.model)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    estimated_tokens = request.max_tokens or 100

    prompt_text = " ".join([msg.content for msg in request.messages])
    estimated_prompt_tokens = len(prompt_text) // 4
    estimated_tokens += estimated_prompt_tokens

    container = await container_manager.get_available_container(request.model, estimated_tokens)
    if not container:
        raise HTTPException(status_code=503, detail=f"No available containers for model '{request.model}'")

    async with container.metrics_lock:
        container.request_count += 1
        container.active_requests += 1
        container.last_used = datetime.now()
        request_start_time = time.time()
        request_id = str(uuid.uuid4())
        container.queue_start_times[request_id] = request_start_time

    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_completion_with_error_handling(request, container, request_id, request_start_time),
                media_type="text/plain"
            )
        else:
            response = await non_streaming_chat_completion(request, container)

            processing_time = time.time() - request_start_time

            actual_tokens = response.usage.get("total_tokens", estimated_tokens)
            container_manager.workload_metrics.record_token_usage(request.model, actual_tokens)

            await container.record_processing_time(processing_time, actual_tokens)

            await container_manager.metrics_tracker.record_request(
                request.model, container.config, actual_tokens, processing_time
            )

            logger.debug(
                f"Request completed in {processing_time:.2f}s with {actual_tokens} tokens on {container.container_name}")
            return response

    finally:

        async with container.metrics_lock:
            container.request_count = max(0, container.request_count - 1)
            container.active_requests = max(0, container.active_requests - 1)
            if request_id in container.queue_start_times:
                del container.queue_start_times[request_id]


@app.get("/health")
async def health_check():
    total_containers = sum(len(containers) for containers in container_manager.container_pools.values())
    ready_containers = sum(len([c for c in containers if c._is_ready])
                           for containers in container_manager.container_pools.values())

    status = "healthy" if ready_containers > 0 else "down"
    return {
        "status": status,
        "total_containers": total_containers,
        "ready_containers": ready_containers,
        "models": list(container_manager.container_pools.keys())
    }


@app.get("/containers")
async def list_containers():
    containers_info = []

    for model_name, containers in container_manager.container_pools.items():
        for container in containers:
            estimated_processing_time = await container.estimate_processing_time(100)
            load_score = await container.get_load_score(100)

            containers_info.append({
                "model": model_name,
                "container_name": container.container_name,
                "port": container.port,
                "is_ready": container._is_ready,
                "request_count": container.request_count,
                "active_requests": container.active_requests,
                "last_used": container.last_used.isoformat(),
                "container_type": container.config.container_type,
                "avg_processing_time": container.avg_processing_time,
                "tokens_per_second": container.tokens_per_second,
                "estimated_processing_time_100_tokens": estimated_processing_time,
                "load_score_100_tokens": load_score
            })

    return {"containers": containers_info}


@app.get("/v1/metrics")
async def get_metrics():
    return await container_manager.metrics_tracker.get_all_metrics()


@app.get("/v1/models")
async def list_models():
    return {"models": list(container_manager.container_pools.keys())}


@app.get("/v1/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    return await container_manager.metrics_tracker.get_model_metrics(model_name)


@app.get("/v1/metrics-summary")


@app.get("/state")
async def get_state():
    """Expose the full proxy state for diagnostics and benchmarks."""
    cm = container_manager
    capacities = {
        str(cfg): cm.workload_metrics.get_config_capacity_tokens_per_hour(cfg)
        for cfg in AVAILABLE_CONFIGS
    }
    models = set(cm.workload_metrics.current_container_config.keys()) | set(cm.container_pools.keys())
    workload_state = {
        model: {
            "current_config": str(cm.workload_metrics.get_current_container_config(model)),
            "current_hardware": cm.workload_metrics.current_hardware_type.get(model),
            "tokens_per_hour": cm.workload_metrics.get_tokens_per_hour(model),
        }
        for model in models
    }

    state = {
        "workload": workload_state,
        "containers": {},
        "metrics": await cm.metrics_tracker.get_all_metrics(),
        "global": {
            "token_threshold": cm.workload_metrics.token_threshold,
            "request_window": cm.workload_metrics.request_window,
            "available_configs": [str(cfg) for cfg in AVAILABLE_CONFIGS],
            "config_capacity_tokens_per_hour": capacities,
        },
    }

    for model_name, containers in cm.container_pools.items():
        state["containers"][model_name] = [
            {
                "container_name": c.container_name,
                "config": str(c.config),
                "container_type": c.config.container_type,
                "is_ready": c._is_ready,
                "request_count": c.request_count,
                "active_requests": c.active_requests,
                "tokens_per_second": c.tokens_per_second,
            }
            for c in containers
        ]

    return state


@app.post("/benchmark/apply_load")
async def benchmark_apply_load(payload: BenchmarkLoadRequest):
    cm = container_manager
    model = payload.model
    if model not in cm.container_pools:
        cm.container_pools[model] = []
        base_config = AVAILABLE_CONFIGS[0]
        cm.workload_metrics.update_container_config(model, base_config)
    if payload.reset:
        cm.workload_metrics.token_events[model].clear()
        cm.workload_metrics.last_scaling_event.pop(model, None)
    if payload.override_cooldown is not None:
        cm.workload_metrics.scaling_cooldown = payload.override_cooldown
    pulses = max(payload.pulses, 1)
    request_window = cm.workload_metrics.request_window
    total_tokens = int(payload.tokens_per_hour * request_window / 3600.0)
    per_event = max(total_tokens // pulses, 1) if total_tokens > 0 else 0
    if per_event > 0:
        for _ in range(pulses):
            cm.workload_metrics.record_token_usage(model, per_event)
    new_config = cm.workload_metrics.select_optimal_config(model)
    return {
        "current_config": str(new_config),
        "tokens_per_hour": cm.workload_metrics.get_tokens_per_hour(model),
    }
async def get_metrics_summary():
    try:
        all_metrics = await container_manager.metrics_tracker.get_all_metrics()
        if not all_metrics:
            return {"message": "No metrics available"}

        summary = {
            "total_models": len(all_metrics),
            "total_requests": sum(m.get('total_requests', 0) for m in all_metrics.values()),
            "total_tokens": sum(m.get('total_tokens', 0) for m in all_metrics.values()),
            "average_throughput": 0,
            "average_time_to_first_token": 0,
            "error_rate": 0,
            "models": {}
        }

        total_requests = summary['total_requests']
        if total_requests > 0:
            summary['average_throughput'] = sum(
                m.get('average_throughput', 0) * m.get('total_requests', 0)
                for m in all_metrics.values()
            ) / total_requests

            summary['average_time_to_first_token'] = sum(
                m.get('average_time_to_first_token', 0) * m.get('total_requests', 0)
                for m in all_metrics.values()
            ) / total_requests

            summary['error_rate'] = sum(
                m.get('error_count', 0) for m in all_metrics.values()
            ) / total_requests * 100

        for key, metrics in all_metrics.items():
            model_name = metrics['model_name']
            if model_name not in summary['models']:
                summary['models'][model_name] = {
                    'total_requests': 0,
                    'total_tokens': 0,
                    'configurations': []
                }

            summary['models'][model_name]['total_requests'] += metrics['total_requests']
            summary['models'][model_name]['total_tokens'] += metrics['total_tokens']
            summary['models'][model_name]['configurations'].append({
                'config': metrics['config_type'],
                'requests': metrics['total_requests'],
                'tokens': metrics['total_tokens'],
                'throughput': metrics['average_throughput'],
                'time_to_first': metrics['average_time_to_first_token'],
                'error_rate': metrics['error_rate']
            })

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/workload-stats")
async def get_all_workload_stats():
    """Get workload statistics for all models"""
    try:
        stats = {}
        for model_name in container_manager.container_pools.keys():
            stats[model_name] = container_manager.workload_metrics.get_workload_stats(model_name)
        return {"workload_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/workload-stats/{model_name}")
async def get_model_workload_stats(model_name: str):
    """Get workload statistics for a specific model"""
    try:
        if model_name not in container_manager.container_pools:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        stats = container_manager.workload_metrics.get_workload_stats(model_name)
        return {"model": model_name, "workload_stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/cost-analysis/{model_name}")
async def get_cost_analysis(model_name: str):
    """Get actual real-time cost analysis based on currently running containers"""
    try:
        if model_name not in container_manager.container_pools:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        containers = container_manager.container_pools[model_name]
        active_containers = [c for c in containers if c._is_ready]

        if not active_containers:
            return {
                "model": model_name,
                "active_containers": 0,
                "total_hourly_cost": 0.0,
                "containers": [],
                "message": "No active containers running"
            }

        total_cost = 0.0
        container_details = []

        for container in active_containers:

            if container.config.container_type == 'cpu':

                hourly_cost = container.config.cpu_cores * 0.10
            elif container.config.container_type == 'gpu':

                hourly_cost = 2.00
            else:
                hourly_cost = 0.10

            total_cost += hourly_cost

            container_details.append({
                "container_name": container.container_name,
                "config_type": container.config.container_type,
                "cpu_cores": container.config.cpu_cores,
                "gpu_percentage": container.config.gpu_percentage,
                "hourly_cost": hourly_cost,
                "status": "ready" if container._is_ready else "not_ready",
                "request_count": container.request_count
            })

        tokens_per_hour = container_manager.workload_metrics.get_tokens_per_hour(model_name)

        return {
            "model": model_name,
            "active_containers": len(active_containers),
            "total_hourly_cost": round(total_cost, 2),
            "current_tokens_per_hour": tokens_per_hour,
            "cost_per_token": round(total_cost / tokens_per_hour, 6) if tokens_per_hour > 0 else None,
            "containers": container_details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI-compatible llama.cpp proxy with clustering")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", default="./models", help="Directory containing model files")
    parser.add_argument("--containers-per-model", type=int, default=1,
                        help="Number of containers per model")
    parser.add_argument("--token-threshold", type=int, default=10000,
                        help="Tokens per hour threshold for GPU usage")
    parser.add_argument("--request-window", type=int, default=60,
                        help="Seconds to consider for workload analysis")

    args = parser.parse_args()

    container_manager.models_dir = Path(args.models_dir).resolve()
    container_manager.containers_per_model = args.containers_per_model
    container_manager.workload_metrics.token_threshold = args.token_threshold
    container_manager.workload_metrics.request_window = args.request_window

    logger.info(f"Starting proxy server on {args.host}:{args.port}")
    logger.info(f"Models directory: {container_manager.models_dir}")
    logger.info(f"Containers per model: {container_manager.containers_per_model}")
    logger.info(f"GPU token threshold: {container_manager.workload_metrics.token_threshold} tokens/hour")
    logger.info(f"Request window: {container_manager.workload_metrics.request_window} seconds")

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")


if __name__ == "__main__":
    import asyncio


    async def setup_containers():
        global container_manager
        container_manager = ContainerManager()
        await initialize_all_model_clusters()

        await container_manager.start_background_container_management()


    asyncio.run(setup_containers())

    main()
