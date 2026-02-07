"""
Dedicated server entrypoint for the 4-config scaling demo benchmark.

Configs: cpu_4, cpu_12, gpu_25, gpu_100 with asymmetric hysteresis
(headroom=0.25) validated by 320/320 simulation passes.

Scaling runs in a background task (not inline with requests) to avoid
blocking the event loop with subprocess.run() Docker commands.

Usage:
    uv run uvicorn benchmarks.scaling_demo_server:app --port <PORT>
"""
from __future__ import annotations

import asyncio
import os
import json
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

CONFIGS: List[HardwareConfig] = [
    HardwareConfig(cpu_cores=4,  memory="8g",  hourly_cost=0.05),
    HardwareConfig(cpu_cores=12, memory="8g",  hourly_cost=0.12),
    HardwareConfig(cpu_cores=2,  memory="8g",  gpu_percentage=25,  hourly_cost=0.50),
    HardwareConfig(cpu_cores=2,  memory="16g", gpu_percentage=100, hourly_cost=4.00),
]

MEASURED_THROUGHPUT = {
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
SCALING_CHECK_INTERVAL = 10  # Check scaling every 10s

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
MODEL_NAME = os.environ.get("E2E_MODEL_NAME", "")

autoscaler: CostAwareAutoscaler | None = None
server_start_time: float = 0.0
scaling_in_progress: bool = False


def _container_info(container: Container) -> Dict:
    """Extract full container metadata for logging."""
    config = container.config
    return {
        "container_name": container.container_name,
        "config_id": config.config_id(),
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
        "measured_throughput_tps": MEASURED_THROUGHPUT.get(config.config_id(), 0),
        "cost_per_token_micro": round(get_cost_per_token(container.model_name, config) * 1e6, 4),
    }


async def _async_container_start(container: Container) -> bool:
    """Non-blocking container start using asyncio subprocess."""
    # Remove existing container
    proc = await asyncio.create_subprocess_exec(
        'docker', 'rm', '-f', container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()

    threads = container.config.cpu_cores or 1
    parallel = container.config.cpu_cores or 1

    docker_cmd = [
        'docker', 'run', '--rm', '-d',
        '--name', container.container_name,
        '-v', f'{container.model_path.parent}:/models:ro',
        '-p', f'{container.port}:8080',
        *container._docker_args(),
        container.config.image,
        '--server',
        '-m', f'/models/{container.model_path.name}',
        '--host', '0.0.0.0',
        '--port', '8080',
        '--threads', str(threads),
        '--parallel', str(parallel),
    ]
    if container.config.gpu_percentage:
        docker_cmd.extend(['--n-gpu-layers', '99'])

    print(f"[SERVER] Starting container: {container.container_name} cmd={docker_cmd[-8:]}", flush=True)

    proc = await asyncio.create_subprocess_exec(
        *docker_cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        print(f"[SERVER] Container start failed: {stderr.decode()}", flush=True)
        return False

    # Wait for health (non-blocking)
    for _ in range(60):  # 60 * 2s = 120s
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"http://localhost:{container.port}/health") as resp:
                    if resp.status == 200:
                        container.is_ready = True
                        print(f"[SERVER] Container ready: {container.container_name}", flush=True)
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)

    print(f"[SERVER] Container health timeout: {container.container_name}", flush=True)
    return False


async def _async_container_stop(container: Container) -> None:
    """Non-blocking container stop."""
    print(f"[SERVER] Stopping container: {container.container_name}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        'docker', 'stop', container.container_name,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    container.is_ready = False


async def _background_scaling_loop() -> None:
    """Background task that checks scaling periodically without blocking requests."""
    global scaling_in_progress
    while True:
        await asyncio.sleep(SCALING_CHECK_INTERVAL)
        if scaling_in_progress:
            continue

        for model_name in list(autoscaler.containers.keys()):
            new_config = autoscaler.check_scaling(model_name)
            if new_config is None:
                continue

            old_config = autoscaler.current_config.get(model_name)
            old_config_id = old_config.config_id() if old_config else "?"
            new_config_id = new_config.config_id()
            demand = autoscaler.demand_tracker.get_demand(model_name)

            print(f"[SCALING_START] {json.dumps({
                'event': 'scaling_start',
                'timestamp': time.time(),
                'elapsed': round(time.time() - server_start_time, 3),
                'model': model_name,
                'from_config': old_config_id,
                'to_config': new_config_id,
                'demand_tps': round(demand, 4),
                'from_hourly_cost': old_config.hourly_cost if old_config else 0,
                'to_hourly_cost': new_config.hourly_cost,
                'from_throughput': MEASURED_THROUGHPUT.get(old_config_id, 0),
                'to_throughput': MEASURED_THROUGHPUT.get(new_config_id, 0),
            })}", flush=True)

            scaling_in_progress = True
            try:
                old_container = autoscaler.containers.get(model_name)
                model_path = old_container.model_path if old_container else autoscaler.get_model_path(model_name)

                port = autoscaler._get_port()
                new_container = Container(model_name, model_path, new_config, port)

                scale_start = time.time()
                if await _async_container_start(new_container):
                    # Swap
                    autoscaler.containers[model_name] = new_container
                    autoscaler.current_config[model_name] = new_config
                    autoscaler.last_scale_time[model_name] = autoscaler.clock()
                    scale_duration = time.time() - scale_start

                    print(f"[SCALING_DONE] {json.dumps({
                        'event': 'scaling_done',
                        'timestamp': time.time(),
                        'elapsed': round(time.time() - server_start_time, 3),
                        'model': model_name,
                        'from_config': old_config_id,
                        'to_config': new_config_id,
                        'scale_duration_s': round(scale_duration, 1),
                        'new_container': _container_info(new_container),
                    })}", flush=True)

                    # Stop old container in background
                    if old_container:
                        await _async_container_stop(old_container)
                else:
                    print(f"[SCALING_FAIL] Failed to start {new_config_id}, keeping {old_config_id}", flush=True)
            finally:
                scaling_in_progress = False


