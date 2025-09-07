#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import asyncio
import uuid
import threading
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Any, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime
from queue import Queue, Empty
import socket
import random
import statistics

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import pandas as pd
from datetime import datetime


logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


model_configs: Dict[str, Dict[str, Any]] = {}
container_manager: 'ContainerManager' = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize containers
    global container_manager
    container_manager = ContainerManager()
    logger.info("Initializing container manager and model clusters...")
    await initialize_all_model_clusters()
    await container_manager.start_background_container_management()
    logger.info("Container initialization completed")

    yield

    # Shutdown: Cleanup containers
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
        # Basic validation only
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
        
        # Processing time estimation fields
        self.processing_times = deque(maxlen=20)  # Store last 20 processing times
        self.token_processing_times = deque(maxlen=20)  # Store (tokens, time) pairs
        self.active_requests = 0  # Current number of active requests
        self.queue_start_times = {}  # Track when requests started queuing
        self.avg_processing_time = 5.0  # Default 5 seconds
        self.tokens_per_second = 10.0  # Default throughput
        self.metrics_lock = asyncio.Lock()  # Protect metrics from race conditions
        
        # Circuit breaker pattern for error handling
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_breaker_threshold = 5  # failures before circuit opens
        self.circuit_breaker_timeout = 60  # seconds before attempting recovery
        self.is_circuit_open = False

    async def is_ready(self) -> bool:
        if self.is_circuit_open:
            # Check if circuit breaker timeout has passed
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
                timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"http://localhost:{self.port}/health") as response:
                        if response.status == 200:
                            self._is_ready = True
                            self.failure_count = 0  # Reset failure count on success
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
            '--name', self.container_name,  # Add explicit container name
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
            # Remove existing container if present
            subprocess.run(['docker', 'rm', '-f', self.container_name], capture_output=True)
            
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self._record_failure()
                return
                
            # Simple readiness check
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
                self.tokens_per_second = sum(t[0] for t in self.token_processing_times) / sum(t[1] for t in self.token_processing_times)
    
    async def estimate_processing_time(self, estimated_tokens: int = 100) -> float:
        async with self.metrics_lock:
            base_time = estimated_tokens / self.tokens_per_second if self.tokens_per_second > 0 else self.avg_processing_time
            return base_time + (self.active_requests * base_time * 0.5)
    
    async def get_load_score(self, estimated_tokens: int = 100) -> float:
        if not self._is_ready:
            return float('inf')
        estimated_time = await self.estimate_processing_time(estimated_tokens)
        return estimated_time + (self.request_count * 2.0)


class WorkloadMetrics:
    def __init__(self, token_threshold: int = 10000, request_window: int = 60):
        self.token_events = {}  # {model_name: [(timestamp, token_count), ...]}
        self.last_request_time = {}
        self.current_hardware_type = {}  # {model_name: 'cpu' or 'gpu'}
        self._cleanup_interval = 600  # Clean up old data every 10 minutes
        self._last_cleanup = time.time()
        self.token_threshold = token_threshold
        self.request_window = request_window
        self.benchmark_data = {}
        self.load_benchmark_metrics()

    def record_token_usage(self, model_name: str, token_count: int):
        """Record token usage with automatic memory cleanup"""
        current_time = time.time()

        if model_name not in self.token_events:
            self.token_events[model_name] = []
            self.current_hardware_type[model_name] = 'cpu'  # Default to CPU

        self.token_events[model_name].append((current_time, token_count))
        self.last_request_time[model_name] = current_time

        # Clean old events outside the window
        cutoff_time = current_time - self.request_window
        self.token_events[model_name] = [
            (timestamp, tokens) for timestamp, tokens in self.token_events[model_name]
            if timestamp > cutoff_time
        ]
        
        # Periodic cleanup of all data structures
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_data(current_time)
            self._last_cleanup = current_time
    
    def _cleanup_old_data(self, current_time: float):
        cutoff_time = current_time - (self.request_window * 2)
        for model_name in list(self.token_events.keys()):
            if model_name in self.token_events:
                self.token_events[model_name] = [
                    (timestamp, tokens) for timestamp, tokens in self.token_events[model_name]
                    if timestamp > cutoff_time
                ]
            # Remove inactive models
            if (model_name in self.last_request_time and 
                current_time - self.last_request_time[model_name] > cutoff_time):
                self.token_events.pop(model_name, None)
                self.last_request_time.pop(model_name, None)
                self.current_hardware_type.pop(model_name, None)
                # Remove if attributes exist
                for attr in ['overload_events', 'last_scaling_time', 'current_container_config']:
                    if hasattr(self, attr):
                        getattr(self, attr).pop(model_name, None)

    def record_request(self, model_name: str):
        """Backward compatibility - record request with default token count"""
        self.record_token_usage(model_name, 100)  # Assume 100 tokens per request

    def get_tokens_per_hour(self, model_name: str) -> float:
        """Calculate current tokens per hour for the model"""
        if model_name not in self.token_events:
            return 0.0

        events = self.token_events[model_name]
        if not events:
            return 0.0

        # Calculate tokens in the current window
        total_tokens = sum(tokens for _, tokens in events)

        # Convert to tokens per hour
        tokens_per_hour = (total_tokens / self.request_window) * 3600
        return tokens_per_hour

    def calculate_cost_per_token(self, model_name: str, hardware_type: str, config: Optional['ContainerConfig'] = None) -> float:
        """Calculate cost per token using benchmark data when available"""
        tokens_per_hour = self.get_tokens_per_hour(model_name)

        # Map config to benchmark hardware config names
        benchmark_config = self.map_config_to_benchmark(hardware_type, config)

        # Try to get cost from benchmark data first
        if benchmark_config:
            benchmark_cost = self.get_benchmark_cost_per_token(benchmark_config, tokens_per_hour)
            if benchmark_cost is not None:
                return benchmark_cost

        # Fallback to calculated costs (simplified without memory)
        if hardware_type == 'cpu':
            if config and config.cpu_cores:
                cpu_cost = self.cpu_cost_per_core_hour * config.cpu_cores
            else:
                cpu_cost = self.cpu_cost_per_core_hour * 1.0  # Default 1 core

            if tokens_per_hour == 0:
                cost_per_hour = cpu_cost * self.cpu_idle_multiplier
            else:
                cost_per_hour = cpu_cost

        else:  # gpu
            if config and config.gpu_percentage:
                # Scale GPU cost by percentage utilization
                gpu_cost = self.gpu_cost_per_hour_full * (config.gpu_percentage / 100.0)
            else:
                gpu_cost = self.gpu_cost_per_hour_full  # Default 100%

            if tokens_per_hour == 0:
                cost_per_hour = gpu_cost * self.gpu_idle_multiplier
            else:
                cost_per_hour = gpu_cost

        if tokens_per_hour == 0:
            return float('inf')  # Infinite cost per token when no tokens

        return cost_per_hour / tokens_per_hour

    def load_benchmark_metrics(self):
        """Load benchmark metrics from analysis results"""
        try:
            benchmark_file = Path(__file__).parent / "benchmarks" / "cost_per_token_analysis_results.csv"

            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['tokens_per_hour']:  # Skip empty rows
                            tokens_per_hour = float(row['tokens_per_hour'])
                            hardware_config = row['hardware_config']
                            cost_per_token = float(row['cost_per_token'])

                            # Store benchmark data by configuration
                            if hardware_config not in self.benchmark_data:
                                self.benchmark_data[hardware_config] = {}

                            self.benchmark_data[hardware_config][tokens_per_hour] = {
                                'cost_per_token': cost_per_token,
                                'total_hourly_cost': float(row['total_hourly_cost']),
                                'utilization_percent': float(row['utilization_percent']),
                                'avg_throughput': float(row['avg_throughput'])
                            }

                logger.info(f"Loaded benchmark data for {len(self.benchmark_data)} hardware configurations")
            else:
                logger.warning("Benchmark data file not found, using default cost assumptions")

        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")

    def get_benchmark_cost_per_token(self, hardware_config: str, tokens_per_hour: float) -> Optional[float]:
        """Get cost per token from benchmark data with interpolation"""
        if hardware_config not in self.benchmark_data:
            return None

        data_points = self.benchmark_data[hardware_config]
        token_rates = sorted(data_points.keys())

        # Exact match
        if tokens_per_hour in data_points:
            return data_points[tokens_per_hour]['cost_per_token']

        # Interpolation between closest points
        lower_rate = None
        upper_rate = None

        for rate in token_rates:
            if rate <= tokens_per_hour:
                lower_rate = rate
            elif rate > tokens_per_hour and upper_rate is None:
                upper_rate = rate
                break

        if lower_rate is None:
            return data_points[token_rates[0]]['cost_per_token']
        elif upper_rate is None:
            return data_points[token_rates[-1]]['cost_per_token']
        else:
            # Linear interpolation
            lower_cost = data_points[lower_rate]['cost_per_token']
            upper_cost = data_points[upper_rate]['cost_per_token']
            ratio = (tokens_per_hour - lower_rate) / (upper_rate - lower_rate)
            return lower_cost + ratio * (upper_cost - lower_cost)

    def map_config_to_benchmark(self, hardware_type: str, config: Optional['ContainerConfig'] = None) -> Optional[str]:
        """Map container configuration to benchmark data configuration names"""
        if hardware_type == 'cpu':
            if config and config.cpu_cores:
                cores = int(config.cpu_cores)
                return f"CPU {cores} cores"
            else:
                return "CPU 1 cores"
        else:  # gpu
            if config and config.gpu_percentage:
                percentage = config.gpu_percentage
                return f"GPU {percentage}%"
            else:
                return "GPU 100%"

    def should_use_gpu(self, model_name: str, cpu_config: Optional['ContainerConfig'] = None, gpu_config: Optional['ContainerConfig'] = None) -> bool:
        """Determine if GPU should be used based on cost per token analysis with specific configs"""
        tokens_per_hour = self.get_tokens_per_hour(model_name)
        current_hw = self.current_hardware_type.get(model_name, 'cpu')

        # Use default configs if not provided
        if cpu_config is None:
            cpu_config = ContainerConfig(cpu_cores=1.0)
        if gpu_config is None:
            gpu_config = ContainerConfig(gpu_percentage=100)

        # Calculate cost per token for both hardware types with actual configs
        cpu_cost_per_token = self.calculate_cost_per_token(model_name, 'cpu', cpu_config)
        gpu_cost_per_token = self.calculate_cost_per_token(model_name, 'gpu', gpu_config)

        # Apply hysteresis to avoid thrashing
        if current_hw == 'cpu':
            # Currently on CPU, need significant benefit to switch to GPU
            threshold = self.token_threshold * (1 + self.hysteresis_factor)
            should_switch = tokens_per_hour > threshold

            # Also check if cost improvement is significant enough
            if should_switch and cpu_cost_per_token != float('inf'):
                cost_improvement = (cpu_cost_per_token - gpu_cost_per_token) / cpu_cost_per_token
                should_switch = cost_improvement > self.switching_cost_threshold

        else:  # current_hw == 'gpu'
            # Currently on GPU, need significant degradation to switch to CPU
            threshold = self.token_threshold * (1 - self.hysteresis_factor)
            should_switch = tokens_per_hour <= threshold

            # Also check cost improvement for switching to CPU
            if should_switch and gpu_cost_per_token != float('inf'):
                cost_improvement = (gpu_cost_per_token - cpu_cost_per_token) / gpu_cost_per_token
                should_switch = cost_improvement > self.switching_cost_threshold

            # Invert the logic since we're checking when to switch away from GPU
            should_switch = not should_switch

        # Update hardware type tracking
        new_hw_type = 'gpu' if should_switch else 'cpu'
        if current_hw == 'cpu' and should_switch:
            new_hw_type = 'gpu'
        elif current_hw == 'gpu' and not should_switch:
            new_hw_type = 'cpu'
        else:
            new_hw_type = current_hw

        self.current_hardware_type[model_name] = new_hw_type

        return new_hw_type == 'gpu'

    def check_container_overload(self, model_name: str, current_config: 'ContainerConfig') -> bool:
        """Check if current container is overloaded and needs scaling"""
        tokens_per_hour = self.get_tokens_per_hour(model_name)

        if tokens_per_hour == 0:
            return False

        # Get benchmark capacity for current configuration
        benchmark_config = self.map_config_to_benchmark(current_config.container_type, current_config)
        if not benchmark_config or benchmark_config not in self.benchmark_data:
            # Fallback to simple threshold-based overload detection
            overload_threshold = self.token_threshold * 1.5  # 15K tokens/hour
            is_overloaded = tokens_per_hour > overload_threshold
        else:
            # Find maximum capacity from benchmark data
            data_points = self.benchmark_data[benchmark_config]
            max_capacity = max(data_points.keys()) if data_points else float('inf')

            # Consider overloaded if current workload exceeds threshold of max benchmark capacity
            overload_threshold = max_capacity * self.overload_threshold_multiplier
            is_overloaded = tokens_per_hour > overload_threshold

        if is_overloaded:
            # Record overload event
            current_time = time.time()
            if model_name not in self.overload_events:
                self.overload_events[model_name] = []
            self.overload_events[model_name].append((current_time, tokens_per_hour))

            # Clean old overload events (keep last 10 minutes)
            cutoff_time = current_time - 600
            self.overload_events[model_name] = [
                (t, tph) for t, tph in self.overload_events[model_name] if t > cutoff_time
            ]

            logger.info(f"Container overload detected for {model_name}: {tokens_per_hour:.0f} tokens/hour > {overload_threshold:.0f} threshold")

        return is_overloaded

    def get_next_config_upgrade(self, current_config: 'ContainerConfig', model_name: str) -> Optional['ContainerConfig']:
        """Determine the next configuration upgrade path based on benchmark data and cost optimization"""
        tokens_per_hour = self.get_tokens_per_hour(model_name)

        # Define upgrade path: 1 core -> 2 cores -> 4 cores -> 8 cores -> GPU 50% -> GPU 100%
        if current_config.container_type == 'cpu':
            current_cores = current_config.cpu_cores or 1.0

            if current_cores < 2.0:
                next_config = ContainerConfig(cpu_cores=2.0)
            elif current_cores < 4.0:
                next_config = ContainerConfig(cpu_cores=4.0)
            elif current_cores < 8.0:
                next_config = ContainerConfig(cpu_cores=8.0)
            else:
                # Upgrade to GPU if CPU maxed out
                next_config = ContainerConfig(gpu_percentage=50)
        else:  # GPU
            current_gpu = current_config.gpu_percentage or 100
            if current_gpu < 100:
                next_config = ContainerConfig(gpu_percentage=100)
            else:
                # No more upgrades available
                return None

        # Verify the upgrade makes sense cost-wise
        current_cost = self.calculate_cost_per_token(model_name, current_config.container_type, current_config)
        next_cost = self.calculate_cost_per_token(model_name, next_config.container_type, next_config)

        # Only upgrade if cost per token decreases or capacity significantly increases
        if current_cost != float('inf') and next_cost != float('inf'):
            cost_ratio = next_cost / current_cost
            # Allow upgrade if cost increase is reasonable (< 50%) or cost decreases
            if cost_ratio > 1.5:
                logger.warning(f"Upgrade for {model_name} would increase cost significantly: {cost_ratio:.2f}x")
                return None

        return next_config

    def should_scale_container(self, model_name: str, current_config: 'ContainerConfig') -> bool:
        """Determine if container should be scaled up due to too many pending requests"""
        current_time = time.time()

        # Check cooldown period
        last_scaling = self.last_scaling_time.get(model_name, 0)
        if current_time - last_scaling < self.scaling_cooldown:
            return False

        # Check if any containers are overloaded with pending requests
        if model_name not in container_manager.container_pools:
            return False

        containers = container_manager.container_pools[model_name]
        active_containers = [c for c in containers if c._is_ready]

        if not active_containers:
            return False

        # Define request threshold for scaling (scale if any container has > 5 pending requests)
        request_threshold = 5
        max_requests = max(c.request_count for c in active_containers)

        if max_requests > request_threshold:
            logger.info(f"Request overload detected for {model_name}: max pending requests = {max_requests} (threshold = {request_threshold})")
            # Update last scaling time to start cooldown
            self.last_scaling_time[model_name] = current_time
            return True

        return False

    def update_container_config(self, model_name: str, config: 'ContainerConfig'):
        """Update the current container configuration for a model"""
        self.current_container_config[model_name] = config
        self.current_hardware_type[model_name] = config.container_type
        logger.info(f"Updated container config for {model_name}: {config}")

    def get_current_container_config(self, model_name: str) -> 'ContainerConfig':
        """Get the current container configuration for a model"""
        if model_name not in self.current_container_config:
            # Default to basic CPU configuration
            default_config = ContainerConfig(cpu_cores=1.0)
            self.current_container_config[model_name] = default_config
            return default_config
        return self.current_container_config[model_name]

    def get_request_count(self, model_name: str) -> int:
        """Backward compatibility - return number of recent requests"""
        return len(self.token_events.get(model_name, []))

    def get_workload_stats(self, model_name: str, cpu_config: Optional['ContainerConfig'] = None, gpu_config: Optional['ContainerConfig'] = None) -> Dict[str, Any]:
        """Get detailed workload statistics for monitoring"""
        tokens_per_hour = self.get_tokens_per_hour(model_name)

        # Use default configs if not provided
        if cpu_config is None:
            cpu_config = ContainerConfig(cpu_cores=1.0)
        if gpu_config is None:
            gpu_config = ContainerConfig(gpu_percentage=100)

        cpu_cost = self.calculate_cost_per_token(model_name, 'cpu', cpu_config)
        gpu_cost = self.calculate_cost_per_token(model_name, 'gpu', gpu_config)
        current_hw = self.current_hardware_type.get(model_name, 'cpu')

        return {
            'tokens_per_hour': tokens_per_hour,
            'cpu_cost_per_token': cpu_cost if cpu_cost != float('inf') else None,
            'gpu_cost_per_token': gpu_cost if gpu_cost != float('inf') else None,
            'cpu_config': str(cpu_config),
            'gpu_config': str(gpu_config),
            'current_hardware': current_hw,
            'crossover_threshold': self.token_threshold,
            'recent_events': len(self.token_events.get(model_name, [])),
            'should_use_gpu': self.should_use_gpu(model_name, cpu_config, gpu_config)
        }


class DecisionLayer:

    def __init__(self, workload_metrics: WorkloadMetrics):
        self.workload_metrics = workload_metrics
        # Define default configurations for cost comparison
        self.default_cpu_config = ContainerConfig(cpu_cores=1.0, memory="4g")
        self.default_gpu_config = ContainerConfig(gpu_percentage=100)

        # Configuration scaling based on workload intensity
        self.high_workload_cpu_config = ContainerConfig(cpu_cores=2.0, memory="8g")
        self.efficient_gpu_config = ContainerConfig(gpu_percentage=50)
        
        # Processing time thresholds
        self.processing_time_threshold = 30.0  # seconds - spawn new container if exceeded
        self.max_containers_per_model = 3  # Maximum containers per model

    def choose_container_config(self, model_name: str) -> ContainerConfig:
        """Choose optimal container configuration based on cost analysis and scaling needs"""
        current_config = self.workload_metrics.get_current_container_config(model_name)

        # First check if current container needs scaling due to overload
        if self.workload_metrics.should_scale_container(model_name, current_config):
            upgrade_config = self.workload_metrics.get_next_config_upgrade(current_config, model_name)
            if upgrade_config:
                logger.info(f"Scaling container for {model_name} from {current_config} to {upgrade_config}")
                self.workload_metrics.update_container_config(model_name, upgrade_config)
                return upgrade_config
            else:
                logger.warning(f"No upgrade path available for {model_name} with config {current_config}")

        # If no scaling needed, perform standard cost-based optimization
        tokens_per_hour = self.workload_metrics.get_tokens_per_hour(model_name)

        # Compare costs for different configurations
        cpu_configs = [self.default_cpu_config, self.high_workload_cpu_config]
        gpu_configs = [self.default_gpu_config, self.efficient_gpu_config]

        best_config = current_config  # Start with current config
        best_cost = self.workload_metrics.calculate_cost_per_token(model_name, current_config.container_type, current_config)

        # Evaluate CPU configurations
        for config in cpu_configs:
            cost = self.workload_metrics.calculate_cost_per_token(model_name, 'cpu', config)
            if cost < best_cost:
                best_cost = cost
                best_config = config

        # Evaluate GPU configurations only if workload justifies it
        if tokens_per_hour > self.workload_metrics.token_threshold * 0.5:  # 50% of threshold
            for config in gpu_configs:
                cost = self.workload_metrics.calculate_cost_per_token(model_name, 'gpu', config)
                if cost < best_cost:
                    best_cost = cost
                    best_config = config

        # Apply final decision with hysteresis
        if best_config.container_type == 'gpu':
            should_use = self.workload_metrics.should_use_gpu(
                model_name,
                self.default_cpu_config,
                best_config
            )
            if not should_use:
                best_config = self.default_cpu_config

        # Update config if it changed
        if str(best_config) != str(current_config):
            self.workload_metrics.update_container_config(model_name, best_config)

        return best_config

    def should_replace_container(self, model_name: str) -> tuple[bool, Optional[ContainerConfig]]:
        """Check if container should be replaced due to scaling needs"""
        current_config = self.workload_metrics.get_current_container_config(model_name)

        # Check for scaling needs
        if self.workload_metrics.should_scale_container(model_name, current_config):
            upgrade_config = self.workload_metrics.get_next_config_upgrade(current_config, model_name)
            if upgrade_config:
                return True, upgrade_config

        # Check for cost optimization opportunities
        optimal_config = self.choose_container_config(model_name)
        if str(optimal_config) != str(current_config):
            # Only replace if cost improvement is significant
            current_cost = self.workload_metrics.calculate_cost_per_token(model_name, current_config.container_type, current_config)
            optimal_cost = self.workload_metrics.calculate_cost_per_token(model_name, optimal_config.container_type, optimal_config)

            if current_cost != float('inf') and optimal_cost != float('inf'):
                cost_improvement = (current_cost - optimal_cost) / current_cost
                if cost_improvement > self.workload_metrics.switching_cost_threshold:
                    return True, optimal_config

        return False, None

    def should_spawn_new_container(self, model_name: str, config: ContainerConfig) -> bool:
        """Check if we should spawn a new container considering limits and existing containers"""
        if model_name not in container_manager.container_pools:
            return True

        containers = container_manager.container_pools[model_name]

        # Enforce maximum container limit
        if len(containers) >= self.max_containers_per_model:
            return False

        ready_containers = [
            c for c in containers
            if c._is_ready and str(c.config) == str(config)
        ]

        return len(ready_containers) == 0

    async def get_best_container(self, model_name: str, config: ContainerConfig, estimated_tokens: int = 100) -> Optional[ContainerInstance]:
        if model_name not in container_manager.container_pools:
            return None

        containers = container_manager.container_pools[model_name]
        ready_containers = [c for c in containers if c._is_ready]

        if not ready_containers:
            logger.warning(f"No ready containers found for {model_name}")
            return None

        # First, try to find containers with matching config (preferred)
        matching_containers = [
            c for c in ready_containers
            if str(c.config) == str(config)
        ]

        if matching_containers:
            logger.debug(f"Found {len(matching_containers)} containers with matching config for {model_name}")
            # Select container with lowest estimated processing time
            load_scores = await asyncio.gather(*[c.get_load_score(estimated_tokens) for c in matching_containers])
            best_idx = load_scores.index(min(load_scores))
            best_container = matching_containers[best_idx]
            logger.debug(f"Selected container {best_container.container_name} with load score {load_scores[best_idx]:.2f}")
            return best_container

        # Fallback: use ANY ready container if no exact config match
        logger.info(f"No exact config match found for {model_name}, using any ready container (config mismatch)")
        load_scores = await asyncio.gather(*[c.get_load_score(estimated_tokens) for c in ready_containers])
        best_idx = load_scores.index(min(load_scores))
        return ready_containers[best_idx]
    
    async def should_spawn_new_container(self, model_name: str, config: ContainerConfig, estimated_tokens: int = 100) -> bool:
        """Determine if we should spawn a new container based on processing time threshold"""
        if model_name not in container_manager.container_pools:
            return True

        containers = container_manager.container_pools[model_name]
        ready_containers = [c for c in containers if c._is_ready]
        
        # Don't exceed maximum containers
        if len(containers) >= self.max_containers_per_model:
            return False
        
        # If no ready containers, we need one
        if not ready_containers:
            return True
            
        # Check if all containers are overloaded
        matching_containers = [
            c for c in ready_containers
            if str(c.config) == str(config)
        ]
        
        if not matching_containers:
            # No containers with this config, should spawn one
            return True
            
        # Check if the best available container exceeds processing time threshold
        load_scores = await asyncio.gather(*[c.get_load_score(estimated_tokens) for c in matching_containers])
        best_idx = load_scores.index(min(load_scores))
        best_container = matching_containers[best_idx]
        estimated_time = await best_container.estimate_processing_time(estimated_tokens)
        
        should_spawn = estimated_time > self.processing_time_threshold
        if should_spawn:
            logger.info(f"Processing time threshold exceeded: {estimated_time:.2f}s > {self.processing_time_threshold}s, spawning new container")
            
        return should_spawn


class ContainerManager:
    def __init__(self, models_dir: str = "./models", containers_per_model: int = 1,
                 token_threshold: int = 10000, request_window: int = 60):
        self.models_dir = Path(models_dir).resolve()
        self.containers_per_model = containers_per_model
        self.max_containers_per_model = 2  # Maximum allowed during replacement
        self.used_ports = set()
        self.lock = asyncio.Lock()
        self.workload_metrics = WorkloadMetrics(token_threshold, request_window)
        self.decision_layer = DecisionLayer(self.workload_metrics)
        self.metrics_tracker = MetricsTracker()
        self.container_management_running = False
        self.container_management_task = None
        self.container_pools: Dict[str, List['ContainerInstance']] = {}

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
                # Evaluate all models for container scaling needs
                for model_name in list(self.container_pools.keys()):
                    await self._evaluate_model_containers(model_name)

                # Wait 30 seconds before next evaluation
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                logger.info("Background container manager cancelled")
                break
            except Exception as e:
                logger.error(f"Error in background container manager: {e}")
                await asyncio.sleep(5)  # Short delay before retry

        logger.info("Background container manager stopped")

    async def _evaluate_model_containers(self, model_name: str):
        """Evaluate if a model needs container scaling based on current metrics"""
        try:
            if model_name not in self.container_pools:
                return

            containers = self.container_pools[model_name]
            active_containers = [c for c in containers if c._is_ready]

            # Enforce maximum container limit
            if len(containers) > self.max_containers_per_model:
                logger.warning(f"Model {model_name} has {len(containers)} containers, max is {self.max_containers_per_model}")
                # Remove excess containers (oldest first)
                excess_containers = containers[self.max_containers_per_model:]
                for container in excess_containers:
                    await container.stop()
                    containers.remove(container)
                logger.info(f"Removed {len(excess_containers)} excess containers for {model_name}")

            # Check if we need to scale based on metrics
            should_replace, new_config = self.decision_layer.should_replace_container(model_name)

            if should_replace and new_config and active_containers:
                logger.info(f"Background evaluation: {model_name} needs scaling to {new_config}")

                # SAFETY CONSTRAINT: Only replace containers with zero active requests
                idle_containers = [c for c in active_containers if c.request_count == 0]

                if idle_containers:
                    # Select an idle container for safe replacement
                    container_to_replace = idle_containers[0]  # Pick first idle container
                    logger.info(f"Found idle container for replacement: {container_to_replace.container_name} (request_count: {container_to_replace.request_count})")

                    # Perform replacement with container limit enforcement
                    new_container = await self._controlled_container_replacement(
                        model_name, container_to_replace, new_config
                    )
                else:
                    logger.info(f"Scaling needed for {model_name} but no idle containers available (all have active requests). Will retry later.")
                    # Log current request counts for debugging
                    request_counts = [f"{c.container_name}:{c.request_count}" for c in active_containers]
                    logger.debug(f"Active container request counts: {', '.join(request_counts)}")
                    new_container = None

                if new_container:
                    self.workload_metrics.update_container_config(model_name, new_config)
                    logger.info(f"Background scaling completed: {model_name} -> {new_config}")

        except Exception as e:
            logger.error(f"Error evaluating containers for {model_name}: {e}")

    async def _controlled_container_replacement(self, model_name: str, old_container: ContainerInstance, new_config: ContainerConfig) -> Optional[ContainerInstance]:
        """Replace container with strict adherence to maximum container limits"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            logger.error(f"Model path not found for {model_name}")
            return None

        try:
            current_containers = len(self.container_pools.get(model_name, []))

            # If we're at max capacity, stop old container first
            if current_containers >= self.max_containers_per_model:
                logger.info(f"At max containers ({self.max_containers_per_model}), stopping old container first")
                await old_container.stop()
                self.container_pools[model_name].remove(old_container)

            # Spawn new container
            new_container = await self.spawn_container(model_name, model_path, new_config)

            if new_container:
                # If old container is still running, stop it now
                if old_container in self.container_pools.get(model_name, []):
                    await old_container.stop()
                    self.container_pools[model_name].remove(old_container)

                logger.info(f"Controlled replacement successful: {old_container.container_name} -> {new_container.container_name}")
                return new_container
            else:
                logger.error(f"Failed to spawn replacement container for {model_name}")
                return None

        except Exception as e:
            logger.error(f"Error in controlled container replacement for {model_name}: {e}")
            return None

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

    async def replace_container(self, model_name: str, old_container: ContainerInstance, new_config: ContainerConfig) -> Optional[ContainerInstance]:
        """Replace an existing container with a new configuration"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            logger.error(f"Model path not found for {model_name}")
            return None

        # Spawn new container with upgraded configuration
        new_container = await self.spawn_container(model_name, model_path, new_config)
        if not new_container:
            logger.error(f"Failed to spawn replacement container for {model_name}")
            return None

        # Stop and remove the old container
        try:
            await old_container.stop()
            if model_name in self.container_pools and old_container in self.container_pools[model_name]:
                self.container_pools[model_name].remove(old_container)
            logger.info(f"Replaced container {old_container.container_name} with {new_container.container_name} for {model_name}")
            return new_container
        except Exception as e:
            logger.error(f"Error stopping old container {old_container.container_name}: {e}")
            # Keep the new container even if old one fails to stop
            return new_container

    async def check_and_scale_containers(self, model_name: str):
        """Check if containers need scaling and perform replacement if necessary"""
        if model_name not in self.container_pools:
            return

        should_replace, new_config = self.decision_layer.should_replace_container(model_name)
        if not should_replace or not new_config:
            return

        # Find a container to replace (prefer the least busy one)
        containers = self.container_pools[model_name]
        active_containers = [c for c in containers if c._is_ready]

        if not active_containers:
            return

        # Select container with lowest request count for replacement
        container_to_replace = min(active_containers, key=lambda c: c.request_count)
        current_config = container_to_replace.config

        logger.info(f"Scaling {model_name}: replacing {current_config} with {new_config}")

        # Replace the container
        new_container = await self.replace_container(model_name, container_to_replace, new_config)
        if new_container:
            # Update workload metrics to track the new configuration
            self.workload_metrics.update_container_config(model_name, new_config)
            logger.info(f"Successfully scaled {model_name} to {new_config}")
        else:
            logger.error(f"Failed to scale {model_name} to {new_config}")

    async def get_available_container(self, model_name: str, estimated_tokens: int = 100) -> Optional[ContainerInstance]:
        if model_name not in self.container_pools:
            return None

        # Use token-based tracking instead of simple request counting
        self.workload_metrics.record_token_usage(model_name, estimated_tokens)

        # Get optimal configuration for current workload
        config = self.decision_layer.choose_container_config(model_name)

        # Check if we need to spawn a new container due to high processing time
        if await self.decision_layer.should_spawn_new_container(model_name, config, estimated_tokens):
            model_path = self.get_model_path(model_name)
            if model_path:
                logger.info(f"Spawning new container for {model_name} due to high processing time")
                new_container = await self.spawn_container(model_name, model_path, config)
                if new_container:
                    logger.info(f"Successfully spawned new container {new_container.container_name}")
                    return new_container

        # Select from existing containers using load balancing
        return await self.decision_layer.get_best_container(model_name, config, estimated_tokens)

    async def cleanup_all_containers(self):
        """Gracefully cleanup all containers with proper error handling"""
        cleanup_tasks = []
        
        for model_name, containers in self.container_pools.items():
            for container in containers:
                cleanup_tasks.append(self._cleanup_single_container(container))
        
        if cleanup_tasks:
            # Run all cleanup tasks concurrently but wait for all to complete
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Log any exceptions that occurred during cleanup
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error during container cleanup: {result}")

        self.container_pools.clear()
        
    async def _cleanup_single_container(self, container: ContainerInstance):
        """Cleanup a single container with timeout"""
        try:
            # Give containers 10 seconds to stop gracefully
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
            # Force remove if graceful stop failed
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
                                                   request_id: str, request_start_time: float) -> AsyncGenerator[str, None]:
    try:
        async for chunk in stream_chat_completion(request, container):
            yield chunk
        # Record completion metrics
        processing_time = time.time() - request_start_time
        await container.record_processing_time(processing_time, 100)
        container_manager.workload_metrics.record_token_usage(request.model, 100)
    except Exception as e:
        container._record_failure()
        yield f"data: {{\"error\": \"Stream interrupted: {str(e)}\"}}}\n\n"
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

    # Estimate tokens for request (prompt + expected completion)
    estimated_tokens = request.max_tokens or 100
    # Add rough estimate for prompt tokens (assume ~1 token per 4 characters)
    prompt_text = " ".join([msg.content for msg in request.messages])
    estimated_prompt_tokens = len(prompt_text) // 4
    estimated_tokens += estimated_prompt_tokens

    container = await container_manager.get_available_container(request.model, estimated_tokens)
    if not container:
        raise HTTPException(status_code=503, detail=f"No available containers for model '{request.model}'")

    # Atomic increment of request counters
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
            # Calculate and record processing time
            processing_time = time.time() - request_start_time
            
            # Update workload metrics with actual token usage
            actual_tokens = response.usage.get("total_tokens", estimated_tokens)
            container_manager.workload_metrics.record_token_usage(request.model, actual_tokens)
            
            # Record processing time in container
            await container.record_processing_time(processing_time, actual_tokens)
            
            await container_manager.metrics_tracker.record_request(
                request.model, container.config, actual_tokens, processing_time
            )
            
            logger.debug(f"Request completed in {processing_time:.2f}s with {actual_tokens} tokens on {container.container_name}")
            return response

    finally:
        # Atomic decrement of request counters
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
                "estimated_processing_time_100_tokens": container.estimate_processing_time(100),
                "load_score_100_tokens": container.get_load_score(100)
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
            # Calculate hourly cost based on container configuration
            if container.config.container_type == 'cpu':
                # CPU cost: $0.10 per core per hour
                hourly_cost = container.config.cpu_cores * 0.10
            elif container.config.container_type == 'gpu':
                # GPU cost: $2.00 per hour (base cost for GPU access)
                hourly_cost = 2.00
            else:
                hourly_cost = 0.10  # Default fallback

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

    logger.info(f"Starting clustered proxy server on {args.host}:{args.port}")
    logger.info(f"Models directory: {container_manager.models_dir}")
    logger.info(f"Containers per model: {container_manager.containers_per_model}")
    logger.info(f"Token threshold for GPU: {container_manager.workload_metrics.token_threshold} tokens/hour")
    logger.info(f"Request window: {container_manager.workload_metrics.request_window} seconds")
    logger.info(f"Cost-based dynamic hardware selection enabled")
    logger.info(f"Benchmark data loaded: {len(container_manager.workload_metrics.benchmark_data)} configurations")

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")


if __name__ == "__main__":
    import asyncio

    async def setup_containers():
        global container_manager
        container_manager = ContainerManager()
        await initialize_all_model_clusters()
        # Start background container management system
        await container_manager.start_background_container_management()

    # Initialize containers first
    asyncio.run(setup_containers())

    # Then start the server normally
    main()
