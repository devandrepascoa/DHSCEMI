#!/usr/bin/env python3
"""Benchmark comparing 4-core vs 12-core CPU containers."""
import asyncio
import subprocess
import time
import aiohttp
import statistics
import socket
from dataclasses import dataclass
from pathlib import Path

MODEL_PATH = Path("./models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf").resolve()
MODEL_NAME = MODEL_PATH.name
MAX_TOKENS = 512
NUM_REQUESTS = 5  # Per configuration
DOCKER_IMAGE = "ghcr.io/ggml-org/llama.cpp:full"


@dataclass
class BenchmarkResult:
    cpu_cores: int
    successful: int
    failed: int
    latencies_ms: list
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    tokens_per_second: float


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_container(cpu_cores: int, port: int) -> str:
    """Start a llama.cpp container with specified CPU cores."""
    container_name = f"llama-bench-{cpu_cores}cpu-{port}"
    
    # Remove if exists
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
    
    cmd = [
        'docker', 'run', '--rm', '-d',
        '--name', container_name,
        '-v', f'{MODEL_PATH.parent}:/models:ro',
        '-p', f'{port}:8080',
        '--cpus', str(cpu_cores),
        '--memory', '16g',
        DOCKER_IMAGE,
        '--server',
        '-m', f'/models/{MODEL_NAME}',
        '--host', '0.0.0.0',
        '--port', '8080',
        '--threads', str(cpu_cores),
        '--parallel', str(cpu_cores),
        '-c', '2048',
    ]
    
    print(f"  Starting container with {cpu_cores} CPUs...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    
    return container_name


async def wait_for_container(port: int, timeout: int = 120) -> bool:
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


async def send_request(session: aiohttp.ClientSession, port: int, request_id: int) -> tuple[float, bool, int]:
    """Send a request and return (latency_ms, success, tokens_generated)."""
    payload = {
        "messages": [{"role": "user", "content": "Explain what machine learning is in detail."}],
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
        print(f"    Request {request_id} failed: {e}")
        return 0, False, 0


async def benchmark_container(cpu_cores: int, port: int, num_requests: int) -> BenchmarkResult:
    """Run benchmark on a container."""
    latencies = []
    failed = 0
    total_tokens = 0
    total_time_s = 0
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            latency, success, tokens = await send_request(session, port, i)
            if success:
                latencies.append(latency)
                total_tokens += tokens
                total_time_s += latency / 1000
                print(f"    Request {i}: {latency:.0f}ms ({tokens} tokens)")
            else:
                failed += 1
    
    if not latencies:
        return BenchmarkResult(cpu_cores, 0, failed, [], 0, 0, 0, 0)
    
    return BenchmarkResult(
        cpu_cores=cpu_cores,
        successful=len(latencies),
        failed=failed,
        latencies_ms=latencies,
        avg_latency_ms=statistics.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        tokens_per_second=total_tokens / total_time_s if total_time_s > 0 else 0,
    )


def stop_container(container_name: str):
    """Stop and remove a container."""
    subprocess.run(['docker', 'stop', container_name], capture_output=True)


async def run_benchmark_for_config(cpu_cores: int) -> BenchmarkResult:
    """Run full benchmark for a CPU configuration."""
    port = get_free_port()
    container_name = start_container(cpu_cores, port)
    
    if not container_name:
        return None
    
    try:
        print(f"  Waiting for container to be ready...")
        if not await wait_for_container(port):
            print(f"  ERROR: Container failed to start")
            return None
        
        print(f"  Container ready, running {NUM_REQUESTS} requests...")
        result = await benchmark_container(cpu_cores, port, NUM_REQUESTS)
        return result
    finally:
        print(f"  Stopping container...")
        stop_container(container_name)


async def main():
    print("=" * 60)
    print("CPU Core Comparison Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Requests per config: {NUM_REQUESTS}")
    print()
    
    configs = [1, 4, 8]
    results = {}
    
    for cpu_cores in configs:
        print(f"\n{'='*60}")
        print(f"Testing {cpu_cores} CPU cores")
        print('='*60)
        
        result = await run_benchmark_for_config(cpu_cores)
        if result:
            results[cpu_cores] = result
        
        # Small delay between configs
        await asyncio.sleep(2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Tok/s':<10}")
    print("-" * 60)
    
    for cpu_cores in configs:
        if cpu_cores in results:
            r = results[cpu_cores]
            print(f"{cpu_cores} cores     {r.avg_latency_ms:<12.0f} {r.min_latency_ms:<12.0f} {r.max_latency_ms:<12.0f} {r.tokens_per_second:<10.1f}")
    
    # Calculate speedups relative to 4 cores
    if 4 in results:
        print()
        for cores in [8, 12]:
            if cores in results:
                speedup = results[4].avg_latency_ms / results[cores].avg_latency_ms
                print(f"Speedup ({cores} cores vs 4 cores): {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
