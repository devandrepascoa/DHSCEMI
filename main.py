#!/usr/bin/env python3
"""
OpenAI-compatible proxy for llama.cpp Docker containers with container clustering
Maintains a pool of pre-loaded containers per model for efficient request routing
"""

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

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import pandas as pd
import statistics
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global container registry
container_pools: Dict[str, List['ContainerInstance']] = {}
model_configs: Dict[str, Dict[str, Any]] = {}

# Initialize FastAPI app at the top level
app = FastAPI(title="llama.cpp OpenAI Proxy", version="1.0.0")

@dataclass
class ContainerConfig:
    """Detailed container configuration"""
    cpu_cores: Optional[float] = None
    memory: Optional[str] = None
    gpu_percentage: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.gpu_percentage is not None and self.gpu_percentage > 0:
            # GPU containers should not have CPU/memory constraints
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
        """Determine container type based on configuration"""
        if self.gpu_percentage is not None and self.gpu_percentage > 0:
            return "gpu"
        return "cpu"
    
    @property
    def image(self) -> str:
        """Get appropriate Docker image based on configuration"""
        if self.container_type == "gpu":
            return "ghcr.io/ggml-org/llama.cpp:full-cuda"
        return "ghcr.io/ggml-org/llama.cpp:full"
    
    def to_docker_args(self) -> List[str]:
        """Convert configuration to Docker command arguments"""
        args = []
        
        if self.container_type == "cpu":
            # Only add CPU/memory constraints for CPU containers
            if self.cpu_cores is not None:
                args.extend(['--cpus', str(self.cpu_cores)])
            if self.memory is not None:
                args.extend(['--memory', self.memory])
        elif self.container_type == "gpu":
            # GPU containers get GPU access
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
        self.config = config  # Detailed configuration instead of simple type
        self.process = None
        self.is_ready = False
        self.last_used = datetime.now()
        self.request_count = 0
        self.lock = asyncio.Lock()

    async def ensure_ready(self) -> bool:
        """Ensure container is ready for requests"""
        if self.is_ready:
            return True
            
        async with self.lock:
            if self.is_ready:  # Double-check after acquiring lock
                return True
                
            try:
                # Health check the container
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/health") as response:
                        if response.status == 200:
                            self.is_ready = True
                            return True
                            
                # If health check fails, restart container
                logger.warning(f"Container {self.container_name} not ready, attempting restart")
                await self.restart()
                return self.is_ready
                
            except Exception as e:
                logger.error(f"Error checking container readiness: {e}")
                return False

    async def restart(self):
        """Restart the container"""
        try:
            # Stop existing container
            subprocess.run(['docker', 'stop', self.container_name], 
                         capture_output=True, check=False)
            
            # Start new container
            await self.start_container()
            
        except Exception as e:
            logger.error(f"Error restarting container {self.container_name}: {e}")

    async def start_container(self):
        """Start the Docker container with detailed configuration"""
        docker_cmd = [
            'docker', 'run', '--rm', '-d',
            '--name', self.container_name,
            '-v', f'{self.model_path.parent}:/models:ro',
            '-p', f'{self.port}:8080',
        ]
        
        # Add configuration-specific arguments
        docker_cmd.extend(self.config.to_docker_args())
        
        # Add common arguments
        docker_cmd.extend([
            self.config.image,
            '--server',
            '-m', f'/models/{self.model_path.name}',
            '--host', '0.0.0.0',
            '--port', '8080',
        ])
        
        # Add CPU-specific threading for CPU containers
        if self.config.container_type == "cpu" and self.config.cpu_cores:
            threads = max(1, int(self.config.cpu_cores))
            docker_cmd.extend(['--threads', str(threads)])
        
        try:
            process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=30)
            
            if process.returncode == 0:
                # Wait for container to be ready
                for _ in range(30):  # 30 second timeout
                    if await self.health_check():
                        self.is_ready = True
                        logger.info(f"Container {self.container_name} ready on port {self.port} ({self.config.container_type})")
                        return
                    await asyncio.sleep(1)
                        
                logger.error(f"Container {self.container_name} failed to become ready")
            else:
                logger.error(f"Failed to start container: {stderr}")
                
        except Exception as e:
            logger.error(f"Error starting container: {e}")

    def get_endpoint(self) -> str:
        """Get the endpoint URL for this container"""
        return f"http://localhost:{self.port}"

class WorkloadMetrics:
    """Track workload metrics for decision making"""
    def __init__(self, gpu_threshold: int = 5, request_window: int = 60):
        self.request_counts = {}
        self.last_request_time = {}
        self.gpu_threshold = gpu_threshold  # Requests to trigger GPU usage
        self.request_window = request_window  # Seconds to consider for threshold
        
    def record_request(self, model_name: str):
        """Record a request for workload tracking"""
        current_time = time.time()
        
        if model_name not in self.request_counts:
            self.request_counts[model_name] = []
            
        # Add current request
        self.request_counts[model_name].append(current_time)
        self.last_request_time[model_name] = current_time
        
        # Clean old requests outside the window
        cutoff_time = current_time - self.request_window
        self.request_counts[model_name] = [
            req_time for req_time in self.request_counts[model_name]
            if req_time > cutoff_time
        ]
    
    def get_request_count(self, model_name: str) -> int:
        """Get request count for a model in the current window"""
        return len(self.request_counts.get(model_name, []))
    
    def should_use_gpu(self, model_name: str) -> bool:
        """Decide if GPU containers should be used based on workload"""
        return self.get_request_count(model_name) >= self.gpu_threshold

class DecisionLayer:
    """Decision layer for container management and routing"""
    def __init__(self, workload_metrics: WorkloadMetrics):
        self.workload_metrics = workload_metrics
        
    def choose_container_config(self, model_name: str) -> ContainerConfig:
        """Choose container configuration based on workload"""
        if self.workload_metrics.should_use_gpu(model_name):
            # High workload - use GPU configuration
            return ContainerConfig(
                gpu_percentage=100  # Only GPU percentage, no CPU/memory
            )
        else:
            # Low workload - use CPU configuration
            return ContainerConfig(
                cpu_cores=1.0,
                memory="4g"
                # No GPU percentage for CPU containers
            )
    
    def should_spawn_new_container(self, model_name: str, config: ContainerConfig) -> bool:
        """Decide if a new container should be spawned"""
        if model_name not in container_pools:
            return True
            
        containers = container_pools[model_name]
        ready_containers = [
            c for c in containers 
            if c.is_ready and str(c.config) == str(config)
        ]
        
        # Spawn new container if no ready containers of the requested type
        return len(ready_containers) == 0
    
    def get_best_container(self, model_name: str, config: ContainerConfig) -> Optional[ContainerInstance]:
        """Get the best available container for routing"""
        if model_name not in container_pools:
            return None
            
        containers = container_pools[model_name]
        available_containers = [
            c for c in containers 
            if c.is_ready and str(c.config) == str(config)
        ]
        
        if not available_containers:
            return None
            
        # Return container with lowest request count (load balancing)
        return min(available_containers, key=lambda c: c.request_count)

class ContainerManager:
    def __init__(self, models_dir: str = "./models", containers_per_model: int = 2, 
                 gpu_threshold: int = 5, request_window: int = 60):
        self.models_dir = Path(models_dir).resolve()
        self.containers_per_model = containers_per_model
        self.port_counter = 8081
        self.lock = asyncio.Lock()
        self.workload_metrics = WorkloadMetrics(gpu_threshold, request_window)
        self.decision_layer = DecisionLayer(self.workload_metrics)
        self.metrics_tracker = MetricsTracker()
    
    def ensure_models_dir(self):
        """Ensure models directory exists"""
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Find model file by name"""
        self.ensure_models_dir()
        
        # Try exact match first
        model_path = self.models_dir / model_name
        if model_path.exists():
            return model_path
            
        # Try common extensions
        for ext in ['.gguf', '.bin']:
            model_path = self.models_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path
                
        # Try to find any matching file
        for file in self.models_dir.iterdir():
            if file.is_file() and model_name.lower() in file.name.lower():
                return file
                
        return None

    async def initialize_model_cluster(self, model_name: str, model_path: Path):
        """Initialize a cluster of containers for a model"""
        if model_name not in container_pools:
            container_pools[model_name] = []
            
        # Always start with CPU containers
        cpu_config = ContainerConfig(cpu_cores=1.0, memory="4g")
        cpu_containers = [
            c for c in container_pools[model_name] 
            if str(c.config) == str(cpu_config)
        ]
        
        # Add CPU containers if needed
        for i in range(max(0, self.containers_per_model - len(cpu_containers))):
            container_name = f"llama-cluster-{model_name}-cpu-{i}"
            port = self.port_counter + len(container_pools.get(model_name, [])) * 10 + i
            
            instance = ContainerInstance(model_name, model_path, container_name, port, cpu_config)
            await instance.start_container()
            
            if instance.is_ready:
                container_pools[model_name].append(instance)
                logger.info(f"Added CPU container {container_name} to pool for {model_name}")
            else:
                logger.error(f"Failed to add CPU container {container_name} to pool")

    async def spawn_container(self, model_name: str, model_path: Path, config: ContainerConfig) -> Optional[ContainerInstance]:
        """Spawn a container with specific configuration"""
        if model_name not in container_pools:
            container_pools[model_name] = []
            
        # Find next available container index
        existing_containers = [
            c for c in container_pools[model_name] 
            if str(c.config) == str(config)
        ]
        container_index = len(existing_containers)
        
        container_type_suffix = "cpu" if config.container_type == "cpu" else "gpu"
        container_name = f"llama-cluster-{model_name}-{container_type_suffix}-{container_index}"
        
        # Calculate port based on container type and index
        base_port = self.port_counter + len(container_pools.get(model_name, [])) * 10
        if config.container_type == "gpu":
            port = base_port + 100 + container_index
        else:
            port = base_port + container_index
        
        instance = ContainerInstance(model_name, model_path, container_name, port, config)
        await instance.start_container()
        
        if instance.is_ready:
            container_pools[model_name].append(instance)
            logger.info(f"Added {config.container_type} container {container_name} to pool for {model_name}")
            return instance
        else:
            logger.error(f"Failed to add {config.container_type} container {container_name} to pool")
            return None

    async def get_available_container(self, model_name: str) -> Optional[ContainerInstance]:
        """Get an available container for a model based on workload"""
        if model_name not in container_pools:
            return None
            
        # Record request for workload tracking
        self.workload_metrics.record_request(model_name)
        
        # Choose container configuration based on workload
        config = self.decision_layer.choose_container_config(model_name)
        
        # Check if we need to spawn a new container
        if self.decision_layer.should_spawn_new_container(model_name, config):
            model_path = self.get_model_path(model_name)
            if model_path:
                return await self.spawn_container(model_name, model_path, config)
            return None
            
        # Get best available container
        return self.decision_layer.get_best_container(model_name, config)

    async def cleanup_all_containers(self):
        """Cleanup all containers"""
        for model_name, containers in container_pools.items():
            for container in containers:
                try:
                    subprocess.run(['docker', 'stop', container.container_name], 
                                 capture_output=True, check=False)
                except Exception as e:
                    logger.error(f"Error stopping container {container.container_name}: {e}")
        
        container_pools.clear()

async def initialize_all_model_clusters():
    """Initialize container clusters for all available models"""
    logger.info("Scanning for models and initializing clusters...")
    
    model_files = []
    for file in container_manager.models_dir.iterdir():
        if file.is_file() and file.suffix.lower() in ['.gguf', '.bin']:
            model_files.append(file)
    model_files = sorted(model_files, key=lambda f: f.name)

    if not model_files:
        logger.warning("No model files found in models directory")
        return
    
    # Initialize clusters for each model
    initialization_tasks = []
    for model_file in model_files:
        model_name = model_file.stem
        logger.info(f"Initializing cluster for model: {model_name}")
        task = container_manager.initialize_model_cluster(model_name, model_file)
        initialization_tasks.append(task)
    
    # Wait for all clusters to initialize
    results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
    
    # Log results
    for i, (model_file, result) in enumerate(zip(model_files, results)):
        model_name = model_file.stem
        if isinstance(result, Exception):
            logger.error(f"Failed to initialize cluster for {model_name}: {result}")
        else:
            container_count = len(container_pools.get(model_name, []))
            ready_count = len([c for c in container_pools.get(model_name, []) if c.is_ready])
            logger.info(f"Model {model_name}: {container_count} containers, {ready_count} ready")

@dataclass
class ModelMetrics:
    """Metrics for a specific model and configuration"""
    model_name: str
    config_type: str  # String representation of the actual configuration
    
    # Performance metrics
    total_requests: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    time_to_first_token: List[float] = field(default_factory=list)
    
    # Detailed timing metrics from benchmarkv2.py
    prompt_processing_ms: List[float] = field(default_factory=list)
    predicted_processing_ms: List[float] = field(default_factory=list)
    prompt_tokens: List[int] = field(default_factory=list)
    predicted_tokens: List[int] = field(default_factory=list)
    
    # Throughput tracking
    tokens_per_second: List[float] = field(default_factory=list)
    prompt_processing_throughput: List[float] = field(default_factory=list)
    token_generation_throughput: List[float] = field(default_factory=list)
    request_durations: List[float] = field(default_factory=list)
    
    # Error tracking
    error_count: int = 0
    last_updated: float = field(default_factory=lambda: time.time())
    
    def record_request(self, tokens: int, duration_seconds: float, time_to_first: Optional[float] = None,
                      prompt_ms: Optional[float] = None, predicted_ms: Optional[float] = None,
                      prompt_tok: Optional[int] = None, predicted_tok: Optional[int] = None):
        """Record metrics for a completed request"""
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
        
        # Calculate tokens per second for this request
        if duration_seconds > 0:
            tps = tokens / duration_seconds
            self.tokens_per_second.append(tps)
        
        # Calculate prompt processing throughput
        if prompt_ms is not None and prompt_tok is not None and prompt_ms > 0:
            prompt_tps = prompt_tok / (prompt_ms / 1000.0)
            self.prompt_processing_throughput.append(prompt_tps)
        
        # Calculate token generation throughput
        if predicted_ms is not None and predicted_tok is not None and predicted_ms > 0:
            token_tps = predicted_tok / (predicted_ms / 1000.0)
            self.token_generation_throughput.append(token_tps)
    
    def record_error(self):
        """Record an error"""
        self.error_count += 1
        self.last_updated = time.time()
    
    @property
    def average_throughput(self) -> float:
        """Get average throughput in tokens per second"""
        if not self.tokens_per_second:
            return 0.0
        return sum(self.tokens_per_second) / len(self.tokens_per_second)
    
    @property
    def average_time_to_first_token(self) -> float:
        """Get average time to first token in seconds"""
        if not self.time_to_first_token:
            return 0.0
        return sum(self.time_to_first_token) / len(self.time_to_first_token)
    
    @property
    def average_prompt_processing_throughput(self) -> float:
        """Get average prompt processing throughput in tokens per second"""
        if not self.prompt_processing_throughput:
            return 0.0
        return sum(self.prompt_processing_throughput) / len(self.prompt_processing_throughput)
    
    @property
    def average_token_generation_throughput(self) -> float:
        """Get average token generation throughput in tokens per second"""
        if not self.token_generation_throughput:
            return 0.0
        return sum(self.token_generation_throughput) / len(self.token_generation_throughput)
    
    @property
    def error_rate(self) -> float:
        """Get error rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary with benchmarkv2.py format"""
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
            "prompt_processing_ms_per_token": (sum(self.prompt_processing_ms) / sum(self.prompt_tokens)) if sum(self.prompt_tokens) > 0 else 0.0,
            "predicted_processing_ms_per_token": (sum(self.predicted_processing_ms) / sum(self.predicted_tokens)) if sum(self.predicted_tokens) > 0 else 0.0,
            "recent_tokens_per_second": self.tokens_per_second[-10:],  # Last 10 values
            "recent_time_to_first": self.time_to_first_token[-10:],   # Last 10 values
            "recent_prompt_processing_throughput": self.prompt_processing_throughput[-10:],
            "recent_token_generation_throughput": self.token_generation_throughput[-10:],
        }

class MetricsTracker:
    """Centralized metrics tracking for all model and configuration combinations"""
    def __init__(self):
        self.metrics: Dict[str, ModelMetrics] = {}
        self.lock = asyncio.Lock()
    
    def _get_key(self, model_name: str, config: ContainerConfig) -> str:
        """Generate unique key for model and full configuration"""
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
        """Record metrics for a completed request"""
        async with self.lock:
            key = self._get_key(model_name, config)
            
            if key not in self.metrics:
                self.metrics[key] = ModelMetrics(
                    model_name=model_name,
                    config_type=str(config)  # Store string representation of config
                )
            
            self.metrics[key].record_request(
                tokens, duration_seconds, time_to_first, 
                prompt_ms, predicted_ms, prompt_tok, predicted_tok
            )
    
    async def record_error(self, model_name: str, config: ContainerConfig):
        """Record an error for model and configuration"""
        async with self.lock:
            key = self._get_key(model_name, config)
            
            if key not in self.metrics:
                self.metrics[key] = ModelMetrics(
                    model_name=model_name,
                    config_type=str(config)
                )
            
            self.metrics[key].record_error()
    
    async def get_metrics(self, model_name: str, config: ContainerConfig) -> Dict[str, Any]:
        """Get metrics for specific model and configuration"""
        key = self._get_key(model_name, config)
        
        async with self.lock:
            if key not in self.metrics:
                return {"error": "No metrics found for this model and configuration"}
            
            return self.metrics[key].to_dict()
    
    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics"""
        async with self.lock:
            return {
                key: metrics.to_dict() 
                for key, metrics in self.metrics.items()
            }
    
    async def get_model_metrics(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """Get all metrics for a specific model"""
        async with self.lock:
            return {
                key: metrics.to_dict() 
                for key, metrics in self.metrics.items() 
                if metrics.model_name == model_name
            }
    
    async def save_metrics(self, output_dir: str = "./metrics_results"):
        """Save all metrics to CSV and JSON files like benchmarkv2.py"""
        async with self.lock:
            if not self.metrics:
                print("No metrics to save")
                return
            
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Collect all metrics into a list of dictionaries
            all_metrics = []
            for key, metrics in self.metrics.items():
                metrics_dict = metrics.to_dict()
                
                # Add configuration details
                config_parts = metrics_dict['config_type'].split('_')
                if len(config_parts) >= 2:
                    metrics_dict['variant'] = config_parts[0]
                    metrics_dict['cpu_cores'] = config_parts[1] if len(config_parts) > 1 else None
                    metrics_dict['gpu_percentage'] = config_parts[2] if len(config_parts) > 2 else None
                
                all_metrics.append(metrics_dict)
            
            if not all_metrics:
                print("No metrics to save")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as CSV
            try:
                df = pd.DataFrame(all_metrics)
                csv_path = output_dir / f"metrics_results_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Metrics saved to: {csv_path}")
                
                # Save as JSON
                json_path = output_dir / f"metrics_results_{timestamp}.json"
                with open(json_path, 'w') as f:
                    json.dump(all_metrics, f, indent=2, default=str)
                print(f"Metrics saved to: {json_path}")
                
                # Print summary
                print("\n📊 Metrics Summary:")
                print("-" * 50)
                
                # Calculate averages by variant
                variants = set(m.get('variant', 'unknown') for m in all_metrics)
                for variant in sorted(variants):
                    variant_results = [m for m in all_metrics if m.get('variant') == variant]
                    if variant_results:
                        print(f"\n{variant.upper()} Results:")
                        
                        # Calculate averages
                        avg_throughput = statistics.mean([m.get('average_throughput', 0) for m in variant_results])
                        avg_prompt_throughput = statistics.mean([m.get('average_prompt_processing_throughput', 0) for m in variant_results])
                        avg_token_throughput = statistics.mean([m.get('average_token_generation_throughput', 0) for m in variant_results])
                        avg_time_to_first = statistics.mean([m.get('average_time_to_first_token', 0) for m in variant_results])
                        avg_error_rate = statistics.mean([m.get('error_rate', 0) for m in variant_results])
                        
                        print(f"  Average Throughput: {avg_throughput:.2f} tokens/sec")
                        print(f"  Average Prompt Processing: {avg_prompt_throughput:.2f} tokens/sec")
                        print(f"  Average Token Generation: {avg_token_throughput:.2f} tokens/sec")
                        print(f"  Average Time to First Token: {avg_time_to_first:.2f} seconds")
                        print(f"  Average Error Rate: {avg_error_rate:.2f}%")
                        
                        total_requests = sum([m.get('total_requests', 0) for m in variant_results])
                        print(f"  Total Requests: {total_requests}")
                        
                return {
                    "csv_path": str(csv_path),
                    "json_path": str(json_path),
                    "metrics_count": len(all_metrics)
                }
                
            except Exception as e:
                print(f"Error saving metrics: {e}")
                return {"error": str(e)}

async def stream_chat_completion(request: ChatCompletionRequest, container: ContainerInstance) -> AsyncGenerator[str, None]:
    """Stream chat completion responses with metrics tracking"""
    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    
    # Prepare the request payload
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
                
                # Collect response data for metrics
                response_data = []
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            response_data.append(chunk)
                            
                            # Track first token time
                            if first_token_time is None:
                                first_token_time = time.time()
                            
                            # Count tokens from choices
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    # Rough token estimation (1 token ~ 4 chars)
                                    total_tokens += max(1, len(content) // 4)
                            
                            yield f"data: {data}\n\n"
                        except json.JSONDecodeError:
                            continue
                
                # Calculate detailed metrics
                end_time = time.time()
                total_duration = end_time - start_time
                time_to_first = first_token_time - start_time if first_token_time else total_duration
                
                # Extract timing data from response if available
                prompt_ms = 0
                predicted_ms = total_duration * 1000  # Convert to ms
                prompt_tokens = 0
                predicted_tokens = total_tokens
                
                # Try to get actual token counts from the last chunk
                if response_data:
                    last_chunk = response_data[-1]
                    if 'usage' in last_chunk:
                        usage = last_chunk['usage']
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        predicted_tokens = usage.get('completion_tokens', total_tokens)
                        
                        # Estimate prompt processing time based on prompt tokens
                        if prompt_tokens > 0:
                            prompt_ms = time_to_first * 1000
                            predicted_ms = (total_duration - time_to_first) * 1000
                
                # Record detailed metrics
                await container_manager.metrics_tracker.record_request(
                    request.model,
                    container.config,
                    total_tokens,
                    total_duration,
                    time_to_first,
                    prompt_ms,
                    predicted_ms,
                    prompt_tokens,
                    predicted_tokens
                )
                
                yield "data: [DONE]\n\n"
                
    except Exception as e:
        await container_manager.metrics_tracker.record_error(
            request.model, container.config
        )
        raise HTTPException(status_code=500, detail=str(e))

async def non_streaming_chat_completion(request: ChatCompletionRequest, container: ContainerInstance) -> ChatCompletionResponse:
    """Non-streaming chat completion with metrics tracking"""
    start_time = time.time()
    
    # Prepare the request payload
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    payload = {
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": False,
        "stop": request.stop
    }
    
    try:
        endpoint = container.get_endpoint()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{container.get_endpoint()}/v1/chat/completions",
                json=request.dict(),
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                if response.status != 200:
                    await container_manager.metrics_tracker.record_error(request.model, container.config)
                    raise HTTPException(status_code=response.status, detail="Container error")
                
                result = await response.json()
                
                # Calculate detailed metrics
                end_time = time.time()
                total_duration = end_time - start_time
                
                # Extract usage data
                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = prompt_tokens + completion_tokens
                
                # Extract timing data if available in response
                prompt_ms = 0
                predicted_ms = total_duration * 1000
                
                # Try to get timing from response metadata
                if 'timings' in result:
                    timings = result['timings']
                    prompt_ms = timings.get('prompt_ms', 0)
                    predicted_ms = timings.get('predicted_ms', total_duration * 1000)
                    prompt_tokens = timings.get('prompt_n', prompt_tokens)
                    completion_tokens = timings.get('predicted_n', completion_tokens)
                
                # Time to first token is the same as total duration for non-streaming
                time_to_first = total_duration
                
                # Record detailed metrics
                await container_manager.metrics_tracker.record_request(
                    request.model,
                    container.config,
                    completion_tokens,  # Only count completion tokens for throughput
                    total_duration,
                    time_to_first,
                    prompt_ms,
                    predicted_ms,
                    prompt_tokens,
                    completion_tokens
                )
                
                # Build response
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
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion using the container cluster"""
    
    # Find model file
    model_path = container_manager.get_model_path(request.model)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
    
    # Ensure model cluster is initialized
    await container_manager.initialize_model_cluster(request.model, model_path)
    
    # Get available container
    container = await container_manager.get_available_container(request.model)
    if not container:
        raise HTTPException(status_code=503, detail=f"No available containers for model '{request.model}'")
    
    # Mark container as in use
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
    """Health check endpoint"""
    total_containers = sum(len(containers) for containers in container_pools.values())
    ready_containers = sum(len([c for c in containers if c.is_ready]) 
                          for containers in container_pools.values())
    
    return {
        "status": "healthy",
        "total_containers": total_containers,
        "ready_containers": ready_containers,
        "models": list(container_pools.keys())
    }

@app.get("/containers")
async def list_containers():
    """List all containers and their status"""
    containers_info = []
    
    for model_name, containers in container_pools.items():
        for container in containers:
            containers_info.append({
                "model": model_name,
                "container_name": container.container_name,
                "port": container.port,
                "is_ready": container.is_ready,
                "request_count": container.request_count,
                "last_used": container.last_used.isoformat(),
                "container_type": container.config.container_type
            })
    
    return {"containers": containers_info}

@app.get("/metrics")
async def get_metrics():
    """Get all metrics"""
    return await container_manager.metrics_tracker.get_all_metrics()

@app.get("/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get metrics for a specific model"""
    return await container_manager.metrics_tracker.get_model_metrics(model_name)

@app.get("/metrics/{model_name}/{config_type}")
async def get_model_config_metrics(model_name: str, config_type: str):
    """Get metrics for a specific model and configuration"""
    config = ContainerConfig(container_type=config_type)
    return await container_manager.metrics_tracker.get_metrics(model_name, config)

@app.post("/v1/save-metrics")
async def save_metrics():
    """Save all metrics to CSV and JSON files"""
    try:
        result = await container_manager.metrics_tracker.save_metrics()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/metrics-summary")
async def get_metrics_summary():
    """Get a summary of all metrics"""
    try:
        all_metrics = await container_manager.metrics_tracker.get_all_metrics()
        if not all_metrics:
            return {"message": "No metrics available"}
        
        # Calculate summary statistics
        summary = {
            "total_models": len(all_metrics),
            "total_requests": sum(m.get('total_requests', 0) for m in all_metrics.values()),
            "total_tokens": sum(m.get('total_tokens', 0) for m in all_metrics.values()),
            "average_throughput": 0,
            "average_time_to_first_token": 0,
            "error_rate": 0,
            "models": {}
        }
        
        # Calculate averages
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
        
        # Group by model
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
    """Main entry point"""
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
    
    # Update configuration
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
