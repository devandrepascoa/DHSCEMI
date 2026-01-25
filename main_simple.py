"""
Simple LLM Inference Autoscaler
- Vertical scaling based on requests per minute (5-min moving average)
- Scales from 1 -> 4 -> 8 CPU cores
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
import socket
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass
from collections import deque
import logging

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    cpu_cores: int
    memory: str
    scale_up_rpm: float    # Scale UP when rpm exceeds this
    scale_down_rpm: float  # Scale DOWN when rpm drops below this
    
    def __str__(self):
        return f"{self.cpu_cores}cpu_{self.memory}"
    
    @property
    def image(self) -> str:
        return "ghcr.io/ggml-org/llama.cpp:full"
    
    def to_docker_args(self) -> List[str]:
        return ['--cpus', str(self.cpu_cores), '--memory', self.memory]


# Available configs for vertical scaling (ordered from smallest to largest)
# Thresholds based on benchmark: 1 core=4.2 tok/s, 4 cores=15.2 tok/s, 8 cores=20.5 tok/s
# At 512 tokens/req: 1 core~0.5 rpm capacity, 4 cores~1.8 rpm, 8 cores~2.4 rpm
# Wide hysteresis band to prevent oscillation
CONTAINER_CONFIGS = [
    ContainerConfig(cpu_cores=1, memory="4g",  scale_up_rpm=0.4,  scale_down_rpm=0.0),   # lowest, no scale down
    ContainerConfig(cpu_cores=4, memory="8g",  scale_up_rpm=1.5,  scale_down_rpm=0.2),   # wide gap: 1.5 vs 0.2
    ContainerConfig(cpu_cores=8, memory="16g", scale_up_rpm=99.0, scale_down_rpm=0.8),   # wide gap: stay at 8 unless rpm < 0.8
]

# Timing constants
RPM_WINDOW_SECONDS = 300      # 5-minute window for rpm calculation
SCALE_COOLDOWN_SECONDS = 300  # 5-minute cooldown between scaling operations


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


class Container:
    def __init__(self, model_name: str, model_path: Path, config: ContainerConfig, port: int):
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.port = port
        self.container_name = f"llama-{model_name}-{config.cpu_cores}cpu-{port}"
        
        self.active_requests = 0
        self.total_requests = 0
        self.is_ready = False
        self.lock = asyncio.Lock()
    
    async def start(self) -> bool:
        # Remove any existing container with same name
        subprocess.run(['docker', 'rm', '-f', self.container_name], 
                      capture_output=True, check=False)
        
        docker_cmd = [
            'docker', 'run', '--rm', '-d',
            '--name', self.container_name,
            '-v', f'{self.model_path.parent}:/models:ro',
            '-p', f'{self.port}:8080',
            *self.config.to_docker_args(),
            self.config.image,
            '--server',
            '-m', f'/models/{self.model_path.name}',
            '--host', '0.0.0.0',
            '--port', '8080',
            '--threads', str(self.config.cpu_cores),
            '--parallel', str(self.config.cpu_cores),
        ]
        
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
    
    async def stop(self):
        logger.info(f"Stopping container: {self.container_name}")
        subprocess.run(['docker', 'stop', self.container_name], 
                      capture_output=True, check=False)
        self.is_ready = False
    
    async def _health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"http://localhost:{self.port}/health") as resp:
                    return resp.status == 200
        except:
            return False
    
    def get_endpoint(self) -> str:
        return f"http://localhost:{self.port}"


class Autoscaler:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir).resolve()
        self.containers: Dict[str, Container] = {}
        self.current_config_idx: Dict[str, int] = {}
        self.last_scale_time: Dict[str, float] = {}
        self.request_timestamps: Dict[str, deque] = {}
        self.used_ports: set = set()
        self.lock = asyncio.Lock()
    
    def _get_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
            self.used_ports.add(port)
            return port
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        for ext in ['', '.gguf', '.bin']:
            path = self.models_dir / f"{model_name}{ext}"
            if path.exists():
                return path
        # Fuzzy match
        for f in self.models_dir.iterdir():
            if f.is_file() and model_name.lower() in f.name.lower():
                return f
        return None
    
    async def initialize(self):
        logger.info(f"Scanning models in {self.models_dir}")
        
        for model_file in self.models_dir.iterdir():
            if model_file.suffix.lower() in ['.gguf', '.bin']:
                model_name = model_file.stem
                self.request_timestamps[model_name] = deque()
                await self._start_container(model_name, model_file, config_idx=0)
    
    async def _start_container(self, model_name: str, model_path: Path, config_idx: int) -> bool:
        config = CONTAINER_CONFIGS[config_idx]
        port = self._get_port()
        
        container = Container(model_name, model_path, config, port)
        if await container.start():
            self.containers[model_name] = container
            self.current_config_idx[model_name] = config_idx
            self.last_scale_time[model_name] = time.time()
            return True
        return False
    
    def _record_request(self, model_name: str):
        self.request_timestamps[model_name].append(time.time())
    
    def _get_rpm(self, model_name: str) -> float:
        if model_name not in self.request_timestamps:
            return 0.0
        
        now = time.time()
        cutoff = now - RPM_WINDOW_SECONDS
        timestamps = self.request_timestamps[model_name]
        
        # Remove old timestamps
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        
        if not timestamps:
            return 0.0
        
        # Calculate rpm: requests in window / window size in minutes
        return len(timestamps) / (RPM_WINDOW_SECONDS / 60)
    
    async def get_container(self, model_name: str) -> Optional[Container]:
        container = self.containers.get(model_name)
        if not container or not container.is_ready:
            return None
        
        # Record request and check scaling
        self._record_request(model_name)
        await self._check_scaling(model_name)
        
        return self.containers.get(model_name)
    
    async def _check_scaling(self, model_name: str):
        now = time.time()
        last_scale = self.last_scale_time.get(model_name, 0)
        
        if now - last_scale < SCALE_COOLDOWN_SECONDS:
            return  # In cooldown
        
        current_idx = self.current_config_idx.get(model_name, 0)
        current_config = CONTAINER_CONFIGS[current_idx]
        rpm = self._get_rpm(model_name)
        
        # Scale UP: rpm exceeds threshold
        if rpm > current_config.scale_up_rpm and current_idx < len(CONTAINER_CONFIGS) - 1:
            new_idx = current_idx + 1
            new_config = CONTAINER_CONFIGS[new_idx]
            logger.info(f"SCALE UP: {model_name} rpm={rpm:.2f} > {current_config.scale_up_rpm} -> {new_config.cpu_cores} cores")
            await self._scale_to(model_name, new_idx)
        
        # Scale DOWN: rpm below threshold
        elif rpm < current_config.scale_down_rpm and current_idx > 0:
            new_idx = current_idx - 1
            new_config = CONTAINER_CONFIGS[new_idx]
            logger.info(f"SCALE DOWN: {model_name} rpm={rpm:.2f} < {current_config.scale_down_rpm} -> {new_config.cpu_cores} cores")
            await self._scale_to(model_name, new_idx)
    
    async def _scale_to(self, model_name: str, new_config_idx: int):
        async with self.lock:
            old_container = self.containers.get(model_name)
            model_path = old_container.model_path if old_container else self.get_model_path(model_name)
            
            if not model_path:
                logger.error(f"Cannot scale: model path not found for {model_name}")
                return
            
            new_config = CONTAINER_CONFIGS[new_config_idx]
            port = self._get_port()
            new_container = Container(model_name, model_path, new_config, port)
            
            if await new_container.start():
                self.containers[model_name] = new_container
                self.current_config_idx[model_name] = new_config_idx
                self.last_scale_time[model_name] = time.time()
                logger.info(f"Scaled {model_name} to {new_config}")
                
                if old_container:
                    while old_container.active_requests > 0:
                        logger.info(f"Draining {old_container.container_name}: {old_container.active_requests} active")
                        await asyncio.sleep(1)
                    await old_container.stop()
            else:
                logger.error(f"Failed to scale {model_name} to {new_config}")
    
    async def cleanup(self):
        for container in self.containers.values():
            await container.stop()
        self.containers.clear()
    
    def get_status(self) -> Dict:
        status = {}
        for name, container in self.containers.items():
            rpm = self._get_rpm(name)
            config = container.config
            status[name] = {
                "config": f"{config.cpu_cores} cores",
                "active_requests": container.active_requests,
                "total_requests": container.total_requests,
                "rpm_5min_avg": round(rpm, 2),
                "scale_up_threshold": config.scale_up_rpm,
                "scale_down_threshold": config.scale_down_rpm,
                "is_ready": container.is_ready,
                "port": container.port,
            }
        return {
            "containers": status,
            "cooldown_seconds": SCALE_COOLDOWN_SECONDS,
            "rpm_window_seconds": RPM_WINDOW_SECONDS,
        }


autoscaler: Autoscaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global autoscaler
    autoscaler = Autoscaler()
    await autoscaler.initialize()
    yield
    await autoscaler.cleanup()


app = FastAPI(title="Simple LLM Autoscaler", lifespan=lifespan)


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
    
    # Track request
    async with container.lock:
        container.active_requests += 1
        container.total_requests += 1
    
    try:
        if request.stream:
            return StreamingResponse(
                _stream_completion(request, container),
                media_type="text/event-stream"
            )
        else:
            return await _non_stream_completion(request, container)
    finally:
        async with container.lock:
            container.active_requests = max(0, container.active_requests - 1)


async def _non_stream_completion(request: ChatCompletionRequest, container: Container) -> ChatCompletionResponse:
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
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                raise HTTPException(resp.status, "Container error")
            
            result = await resp.json()
            
            choices = []
            for i, choice in enumerate(result.get('choices', [])):
                msg = choice.get('message', {})
                choices.append(ChatCompletionChoice(
                    index=i,
                    message={"role": msg.get('role', 'assistant'), "content": msg.get('content', '')},
                    finish_reason=choice.get('finish_reason')
                ))
            
            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=result.get('usage', {})
            )


async def _stream_completion(request: ChatCompletionRequest, container: Container) -> AsyncGenerator[str, None]:
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
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status != 200:
                yield f"data: {json.dumps({'error': 'Container error'})}\n\n"
                return
            
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    yield f"{line}\n\n"
                    if line == 'data: [DONE]':
                        break
            
            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
