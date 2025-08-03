#!/usr/bin/env python3

import json
import subprocess
import asyncio
import uuid
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime
from queue import Queue, Empty
import socket
import random

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import pandas as pd
import statistics
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

container_pools: Dict[str, List['ContainerInstance']] = {}
model_configs: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="llama.cpp OpenAI Proxy",
    version="1.0.0",
)


@dataclass
class ContainerConfig:
    cpu_cores: Optional[float] = None
    memory: Optional[str] = None
    gpu_percentage: Optional[int] = None

    def __post_init__(self):
        if self.gpu_percentage is not None and self.gpu_percentage > 0:

            if self.cpu_cores is not None:
                raise ValueError("cpu_cores cannot be set when gpu_percentage is specified")
            if self.memory is not None:
                raise ValueError("memory cannot be set when gpu_percentage is specified")

    def __str__(self):
        cpu_str = f"cpu{self.cpu_cores}" if self.cpu_cores else "cpu"
        memory_str = f"mem{self.memory}" if self.memory else "mem"
        gpu_str = f"gpu{self.gpu_percentage}" if self.gpu_percentage else "nogpu"
        return f"{cpu_str}_{memory_str}_{gpu_str}"

    @property
    def container_type(self) -> str:

        if self.gpu_percentage is not None and self.gpu_percentage > 0:
            return "gpu"
        return "cpu"

    @property
    def image(self) -> str:

        if self.container_type == "gpu":
            return "ghcr.io/ggml-org/llama.cpp:full-cuda"
        return "ghcr.io/ggml-org/llama.cpp:full"

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

    async def is_ready(self) -> bool:

        if self._is_ready:
            return True

        async with self.lock:
            if self._is_ready:
                return True

            try:

                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/health") as response:
                        if response.status == 200:
                            self._is_ready = True
                            return True

                return self._is_ready

            except Exception as e:
                return False

    async def restart(self):

        try:

            subprocess.run(['docker', 'stop', self.container_name],
                           capture_output=True, check=False)

            await self.start_container()

        except Exception as e:
            logger.error(f"Error restarting container {self.container_name}: {e}")

    async def start_container(self):

        docker_cmd = [
            'docker', 'run', '--rm', '-d',

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
            process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=30)

            if process.returncode == 0:

                for _ in range(30):
                    if await self.is_ready():
                        self._is_ready = True
                        logger.info(
                            f"Container {self.container_name} ready on port {self.port} ({self.config.container_type})")
                        return
                    await asyncio.sleep(1)

                logger.error(f"Container {self.container_name} failed to become ready")
            else:
                logger.error(f"Failed to start container: {stderr}")

        except Exception as e:
            logger.error(f"Error starting container: {e}")

    def get_endpoint(self) -> str:

        return f"http://localhost:{self.port}"


class WorkloadMetrics:

    def __init__(self, gpu_threshold: int = 5, request_window: int = 60):
        self.request_counts = {}
        self.last_request_time = {}
        self.gpu_threshold = gpu_threshold
        self.request_window = request_window

    def record_request(self, model_name: str):
        current_time = time.time()

        if model_name not in self.request_counts:
            self.request_counts[model_name] = []

        self.request_counts[model_name].append(current_time)
        self.last_request_time[model_name] = current_time

        cutoff_time = current_time - self.request_window
        self.request_counts[model_name] = [
            req_time for req_time in self.request_counts[model_name]
            if req_time > cutoff_time
        ]

    def get_request_count(self, model_name: str) -> int:
        return len(self.request_counts.get(model_name, []))

    def should_use_gpu(self, model_name: str) -> bool:
        return self.get_request_count(model_name) >= self.gpu_threshold


class DecisionLayer:

    def __init__(self, workload_metrics: WorkloadMetrics):
        self.workload_metrics = workload_metrics

    def choose_container_config(self, model_name: str) -> ContainerConfig:

        if self.workload_metrics.should_use_gpu(model_name):
            return ContainerConfig(
                gpu_percentage=100
            )
        else:

            return ContainerConfig(
                cpu_cores=1.0,
                memory="4g"

            )

    def should_spawn_new_container(self, model_name: str, config: ContainerConfig) -> bool:

        if model_name not in container_pools:
            return True

        containers = container_pools[model_name]
        ready_containers = [
            c for c in containers
            if c._is_ready and str(c.config) == str(config)
        ]

        return len(ready_containers) == 0

    def get_best_container(self, model_name: str, config: ContainerConfig) -> Optional[ContainerInstance]:

        if model_name not in container_pools:
            return None

        containers = container_pools[model_name]
        available_containers = [
            c for c in containers
            if c._is_ready and str(c.config) == str(config)
        ]

        if not available_containers:
            return None

        return min(available_containers, key=lambda c: c.request_count)


class ContainerManager:
    def __init__(self, models_dir: str = "./models", containers_per_model: int = 2,
                 gpu_threshold: int = 5, request_window: int = 60):
        self.models_dir = Path(models_dir).resolve()
        self.containers_per_model = containers_per_model
        self.used_ports = set()
        self.lock = asyncio.Lock()
        self.workload_metrics = WorkloadMetrics(gpu_threshold, request_window)
        self.decision_layer = DecisionLayer(self.workload_metrics)
        self.metrics_tracker = MetricsTracker()

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

        if model_name not in container_pools:
            container_pools[model_name] = []

        cpu_config = ContainerConfig(cpu_cores=1.0, memory="4g")
        cpu_containers = [
            c for c in container_pools[model_name]
            if str(c.config) == str(cpu_config)
        ]

        for i in range(max(0, self.containers_per_model - len(cpu_containers))):
            port = self._get_available_port()
            container_name = f"llama-cluster-{model_name}-cpu-{port}"

            instance = ContainerInstance(model_name, model_path, container_name, port, cpu_config)
            await instance.start_container()

            if instance._is_ready:
                container_pools[model_name].append(instance)
                logger.info(f"Added CPU container {container_name} to pool for {model_name}")
            else:
                logger.error(f"Failed to add CPU container {container_name} to pool")

    async def spawn_container(self, model_name: str, model_path: Path, config: ContainerConfig) -> Optional[
        ContainerInstance]:

        if model_name not in container_pools:
            container_pools[model_name] = []

        container_type_suffix = "cpu" if config.container_type == "cpu" else "gpu"
        port = self._get_available_port()
        container_name = f"llama-cluster-{model_name}-{container_type_suffix}-{port}"

        instance = ContainerInstance(model_name, model_path, container_name, port, config)
        await instance.start_container()

        if instance._is_ready:
            container_pools[model_name].append(instance)
            logger.info(f"Added {config.container_type} container {container_name} to pool for {model_name}")
            return instance
        else:
            logger.error(f"Failed to add {config.container_type} container {container_name} to pool")
            return None

    async def get_available_container(self, model_name: str) -> Optional[ContainerInstance]:

        if model_name not in container_pools:
            return None

        self.workload_metrics.record_request(model_name)

        config = self.decision_layer.choose_container_config(model_name)

        if self.decision_layer.should_spawn_new_container(model_name, config):
            model_path = self.get_model_path(model_name)
            if model_path:
                return await self.spawn_container(model_name, model_path, config)
            return None

        return self.decision_layer.get_best_container(model_name, config)

    async def cleanup_all_containers(self):

        for model_name, containers in container_pools.items():
            for container in containers:
                try:
                    subprocess.run(['docker', 'stop', container.container_name],
                                   capture_output=True, check=False)
                except Exception as e:
                    logger.error(f"Error stopping container {container.container_name}: {e}")

        container_pools.clear()


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
            container_count = len(container_pools.get(model_name, []))
            ready_count = len([c for c in container_pools.get(model_name, []) if c._is_ready])
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

                            prompt_ms = 0

                            if 'timings' in result:
                                timings = result['timings']
                                prompt_ms = timings.get('prompt_ms', 0)
                                predicted_ms = timings.get('predicted_ms', 0)
                                prompt_tokens = timings.get('prompt_n', prompt_tokens)
                                completion_tokens = timings.get('predicted_n', completion_tokens)

                            await container_manager.metrics_tracker.record_request(
                                request.model,
                                container.config,
                                completion_tokens,
                                -1,
                                -1,
                                prompt_ms,
                                predicted_ms,
                                prompt_tokens,
                                completion_tokens
                            )

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

    await container_manager.initialize_model_cluster(request.model, model_path)

    container = await container_manager.get_available_container(request.model)
    if not container:
        raise HTTPException(status_code=503, detail=f"No available containers for model '{request.model}'")

    container.request_count += 1
    container.last_used = datetime.now()

    try:
        if request.stream:
            return StreamingResponse(
                stream_chat_completion(request, container),
                media_type="text/plain"
            )
        else:
            response = await non_streaming_chat_completion(request, container)
            await container_manager.metrics_tracker.record_request(
                request.model, container.config, response.usage["total_tokens"], 0.0
            )
            return response

    finally:
        container.request_count = max(0, container.request_count - 1)


@app.get("/health")
async def health_check():
    total_containers = sum(len(containers) for containers in container_pools.values())
    ready_containers = sum(len([c for c in containers if c._is_ready])
                           for containers in container_pools.values())

    return {
        "status": "healthy",
        "total_containers": total_containers,
        "ready_containers": ready_containers,
        "models": list(container_pools.keys())
    }


@app.get("/containers")
async def list_containers():
    containers_info = []

    for model_name, containers in container_pools.items():
        for container in containers:
            containers_info.append({
                "model": model_name,
                "container_name": container.container_name,
                "port": container.port,
                "is_ready": container._is_ready,
                "request_count": container.request_count,
                "last_used": container.last_used.isoformat(),
                "container_type": container.config.container_type
            })

    return {"containers": containers_info}


@app.get("/metrics")
async def get_metrics():
    return await container_manager.metrics_tracker.get_all_metrics()


@app.get("/v1/models")
async def list_models():
    global container_pools

    return {"models": list(container_pools.keys())}


@app.get("/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    return await container_manager.metrics_tracker.get_model_metrics(model_name)


@app.get("/metrics/{model_name}/{config_type}")
async def get_model_config_metrics(model_name: str, config_type: str):
    config = ContainerConfig(container_type=config_type)
    return await container_manager.metrics_tracker.get_metrics(model_name, config)


@app.post("/v1/save-metrics")
async def save_metrics():
    try:
        result = await container_manager.metrics_tracker.save_metrics()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics-summary")
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI-compatible llama.cpp proxy with clustering")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", default="./models", help="Directory containing model files")
    parser.add_argument("--containers-per-model", type=int, default=2,
                        help="Number of containers per model")
    parser.add_argument("--gpu-threshold", type=int, default=5,
                        help="Number of requests to trigger GPU usage")
    parser.add_argument("--request-window", type=int, default=60,
                        help="Seconds to consider for GPU threshold")

    args = parser.parse_args()

    container_manager.models_dir = Path(args.models_dir).resolve()
    container_manager.containers_per_model = args.containers_per_model
    container_manager.workload_metrics.gpu_threshold = args.gpu_threshold
    container_manager.workload_metrics.request_window = args.request_window

    logger.info(f"Starting clustered proxy server on {args.host}:{args.port}")
    logger.info(f"Models directory: {container_manager.models_dir}")
    logger.info(f"Containers per model: {container_manager.containers_per_model}")
    logger.info(f"GPU threshold: {container_manager.workload_metrics.gpu_threshold}")
    logger.info(f"Request window: {container_manager.workload_metrics.request_window}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    container_manager = ContainerManager()

    main()
