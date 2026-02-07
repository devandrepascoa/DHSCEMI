#!/usr/bin/env python3
"""
Benchmark throughput for different hardware configurations.
Tests CPU (1, 4, 8 cores) and GPU (50%, 100%) to get tokens/second values.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
import socket
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import aiohttp

MODEL_PATH = Path("./models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf").resolve()
MODEL_NAME = MODEL_PATH.stem
MAX_TOKENS = 256
NUM_REQUESTS = 3  # Per configuration
WARMUP_REQUESTS = 1  # Warmup before measuring

CPU_IMAGE = "ghcr.io/ggml-org/llama.cpp:full"
GPU_IMAGE = "ghcr.io/ggml-org/llama.cpp:full-cuda"


@dataclass
class HardwareConfig:
    config_id: str
    cpu_cores: Optional[int] = None
    gpu_percentage: Optional[int] = None
    memory: str = "8g"
    
    @property
    def is_gpu(self) -> bool:
        return self.gpu_percentage is not None
    
    @property
    def image(self) -> str:
        return GPU_IMAGE if self.is_gpu else CPU_IMAGE
    
    def docker_args(self) -> List[str]:
        args = ['--memory', self.memory]
        if self.is_gpu:
            # GPU config - use nvidia runtime
            args.extend(['--gpus', 'all'])
            # Limit GPU memory if partial allocation
            if self.gpu_percentage and self.gpu_percentage < 100:
                # Note: actual GPU memory limiting requires CUDA_VISIBLE_DEVICES or MPS
                pass
        else:
            args.extend(['--cpus', str(self.cpu_cores)])
        return args
    
    def server_args(self) -> List[str]:
        if self.is_gpu:
            # GPU: offload layers based on percentage
            # For a small model like 1.5B, ~24 layers total
            # 50% = ~12 layers, 100% = all layers (99)
            if self.gpu_percentage == 100:
                return ['-ngl', '99']  # Offload all layers to GPU
            elif self.gpu_percentage == 50:
                return ['-ngl', '12']  # Offload ~half the layers
            else:
                layers = int(24 * self.gpu_percentage / 100)
                return ['-ngl', str(layers)]
        else:
            return ['--threads', str(self.cpu_cores)]


# Configurations to benchmark
CONFIGS = [
    HardwareConfig(config_id="cpu_1", cpu_cores=1, memory="4g"),
    HardwareConfig(config_id="cpu_4", cpu_cores=4, memory="8g"),
    HardwareConfig(config_id="cpu_8", cpu_cores=8, memory="16g"),
    HardwareConfig(config_id="gpu_50", gpu_percentage=50, memory="8g"),
    HardwareConfig(config_id="gpu_100", gpu_percentage=100, memory="8g"),
]


@dataclass
class BenchmarkResult:
    config_id: str
    successful: int
    failed: int
    avg_latency_ms: float
    tokens_per_second: float
    total_tokens: int


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_container(config: HardwareConfig, port: int) -> Optional[str]:
    """Start a llama.cpp container with specified config."""
    container_name = f"llama-bench-{config.config_id}-{port}"
    
    # Remove if exists
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
    
    cmd = [
        'docker', 'run', '--rm', '-d',
        '--name', container_name,
        '-v', f'{MODEL_PATH.parent}:/models:ro',
        '-p', f'{port}:8080',
        *config.docker_args(),
        config.image,
        '--server',
        '-m', f'/models/{MODEL_PATH.name}',
        '--host', '0.0.0.0',
        '--port', '8080',
        *config.server_args(),
        '-c', '2048',
    ]
    
    print(f"  Starting container: {' '.join(cmd[-10:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    
    return container_name


async def wait_for_container(port: int, timeout: int = 180) -> bool:
    """Wait for container to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"http://localhost:{port}/health") as resp:
                    if resp.status == 200:
                        return True
        except:
            pass
        await asyncio.sleep(2)
    return False


async def send_request(session: aiohttp.ClientSession, port: int) -> tuple[float, bool, int]:
    """Send a request and return (latency_ms, success, tokens_generated)."""
    payload = {
        "messages": [{"role": "user", "content": "Explain what machine learning is."}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": False,
    }
    
    start = time.perf_counter()
    try:
        async with session.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                latency = (time.perf_counter() - start) * 1000
                tokens = result.get('usage', {}).get('completion_tokens', MAX_TOKENS)
                return latency, True, tokens
            else:
                return 0, False, 0
    except Exception as e:
        print(f"    Request failed: {e}")
        return 0, False, 0


async def benchmark_container(config: HardwareConfig, port: int) -> BenchmarkResult:
    """Run benchmark on a container."""
    latencies = []
    failed = 0
    total_tokens = 0
    total_time_s = 0
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print(f"  Warmup ({WARMUP_REQUESTS} requests)...")
        for _ in range(WARMUP_REQUESTS):
            await send_request(session, port)
        
        # Actual benchmark
        print(f"  Benchmarking ({NUM_REQUESTS} requests)...")
        for i in range(NUM_REQUESTS):
            latency, success, tokens = await send_request(session, port)
            if success:
                latencies.append(latency)
                total_tokens += tokens
                total_time_s += latency / 1000
                print(f"    Request {i+1}: {latency:.0f}ms, {tokens} tokens, {tokens/(latency/1000):.1f} tok/s")
            else:
                failed += 1
    
    if not latencies:
        return BenchmarkResult(config.config_id, 0, failed, 0, 0, 0)
    
    return BenchmarkResult(
        config_id=config.config_id,
        successful=len(latencies),
        failed=failed,
        avg_latency_ms=sum(latencies) / len(latencies),
        tokens_per_second=total_tokens / total_time_s if total_time_s > 0 else 0,
        total_tokens=total_tokens,
    )


def stop_container(container_name: str):
    """Stop and remove a container."""
    subprocess.run(['docker', 'stop', container_name], capture_output=True)


async def run_benchmark_for_config(config: HardwareConfig) -> Optional[BenchmarkResult]:
    """Run full benchmark for a hardware configuration."""
    port = get_free_port()
    container_name = start_container(config, port)
    
    if not container_name:
        return None
    
    try:
        print(f"  Waiting for container to be ready...")
        if not await wait_for_container(port):
            print(f"  ERROR: Container failed to start")
            return None
        
        result = await benchmark_container(config, port)
        return result
    finally:
        print(f"  Stopping container...")
        stop_container(container_name)


async def main():
    print("=" * 70)
    print("Hardware Configuration Throughput Benchmark")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Requests per config: {NUM_REQUESTS} (+ {WARMUP_REQUESTS} warmup)")
    print()
    
    results = {}
    
    for config in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing: {config.config_id}")
        print('='*70)
        
        result = await run_benchmark_for_config(config)
        if result:
            results[config.config_id] = result
        
        await asyncio.sleep(2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Tokens per Second by Configuration")
    print("=" * 70)
    print(f"{'Config':<12} {'Avg Latency':<15} {'Tokens/sec':<12} {'Status'}")
    print("-" * 70)
    
    for config in CONFIGS:
        if config.config_id in results:
            r = results[config.config_id]
            print(f"{r.config_id:<12} {r.avg_latency_ms:>10.0f} ms   {r.tokens_per_second:>10.1f}    OK")
        else:
            print(f"{config.config_id:<12} {'N/A':>10}      {'N/A':>10}    FAILED")
    
    # Output as Python dict for copy-paste
    print("\n" + "=" * 70)
    print("Copy this to DEFAULT_THROUGHPUT in your code:")
    print("=" * 70)
    print("DEFAULT_THROUGHPUT = {")
    for config in CONFIGS:
        if config.config_id in results:
            r = results[config.config_id]
            print(f'    "{r.config_id}": {r.tokens_per_second:.1f},')
    print("}")
    
    # Save to JSON
    output = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "results": {k: {"tokens_per_second": v.tokens_per_second, "avg_latency_ms": v.avg_latency_ms} 
                   for k, v in results.items()}
    }
    with open("throughput_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to throughput_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
