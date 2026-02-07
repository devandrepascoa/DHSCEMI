#!/usr/bin/env python3
"""
Benchmark throughput for different hardware configurations.
Tests CPU and GPU configs with varying batch sizes to measure aggregate tokens/second.
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
NUM_REQUESTS = 3  # Rounds per batch size
WARMUP_REQUESTS = 1
BATCH_TIMEOUT = 120  # 2 minute timeout per batch size (covers all rounds)

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
            args.extend(['--gpus', 'all', '--privileged'])
        else:
            args.extend(['--cpus', str(self.cpu_cores)])
        return args

    def server_args(self) -> List[str]:
        if self.is_gpu:
            if self.gpu_percentage == 100:
                return ['-ngl', '99']
            elif self.gpu_percentage == 25:
                return ['-ngl', '6']
            else:
                layers = int(24 * self.gpu_percentage / 100)
                return ['-ngl', str(layers)]
        else:
            return ['--threads', str(self.cpu_cores)]


# Configurations to benchmark
CONFIGS = [
   # HardwareConfig(config_id="cpu_4", cpu_cores=4),
   # HardwareConfig(config_id="cpu_8", cpu_cores=8),
    HardwareConfig(config_id="cpu_12", cpu_cores=12),
    #HardwareConfig(config_id="gpu_25", gpu_percentage=25, memory="8g"),
   # HardwareConfig(config_id="gpu_100", gpu_percentage=100, memory="8g"),
]

BATCH_SIZES = [1, 4, 16, 32]

PROMPT = "Explain what machine learning is."
# Rough estimate of input tokens for the prompt + system overhead
INPUT_TOKENS_ESTIMATE = 15


@dataclass
class BenchmarkResult:
    config_id: str
    batch_size: int
    successful: int
    failed: int
    timed_out: int
    tokens_per_second: float
    total_output_tokens: int
    total_input_tokens: int
    wall_time_seconds: float


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_container(config: HardwareConfig, port: int) -> Optional[str]:
    """Start a llama.cpp container with specified config."""
    container_name = f"llama-bench-{config.config_id}-{port}"
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
        '--parallel', '32',
    ]

    print(f"  Starting container: {' '.join(cmd[-10:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    return container_name


async def wait_for_container(port: int, timeout: int = 180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"http://localhost:{port}/health") as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)
    return False


async def send_request(session: aiohttp.ClientSession, port: int) -> tuple[float, bool, int, int]:
    """Send a request. Returns (latency_ms, success, output_tokens, input_tokens)."""
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        async with session.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)  # generous per-request timeout
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                latency = (time.perf_counter() - start) * 1000
                usage = result.get('usage', {})
                output_tokens = usage.get('completion_tokens', 0)
                input_tokens = usage.get('prompt_tokens', INPUT_TOKENS_ESTIMATE)
                return latency, True, output_tokens, input_tokens
            else:
                return 0, False, 0, 0
    except Exception as e:
        print(f"    Request failed: {e}")
        return 0, False, 0, 0


async def benchmark_batch(port: int, batch_size: int) -> BenchmarkResult:
    """Run benchmark for a batch size with BATCH_TIMEOUT covering all rounds.
    Partial results from completed rounds are preserved even if a later round times out."""
    async with aiohttp.ClientSession() as session:
        # Warmup (outside timeout)
        print(f"  Warmup ({WARMUP_REQUESTS} requests)...")
        for _ in range(WARMUP_REQUESTS):
            await send_request(session, port)

        print(f"  Benchmarking (batch_size={batch_size}, {NUM_REQUESTS} rounds, "
              f"timeout={BATCH_TIMEOUT}s for all rounds)...")

        total_output_tokens = 0
        total_input_tokens = 0
        failed = 0
        round_tps_list = []
        total_wall_time = 0.0
        timed_out = 0
        deadline = time.perf_counter() + BATCH_TIMEOUT

        for i in range(NUM_REQUESTS):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                print(f"    TIMEOUT — no time left for round {i+1}")
                timed_out = 1
                break

            tasks = [send_request(session, port) for _ in range(batch_size)]
            start = time.perf_counter()
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                wall_time = time.perf_counter() - start
                total_wall_time += wall_time
                print(f"    TIMEOUT during round {i+1} after {wall_time:.1f}s")
                timed_out = 1
                break

            wall_time = time.perf_counter() - start
            total_wall_time += wall_time

            round_out_tokens = 0
            round_in_tokens = 0
            round_failed = 0
            for r in results:
                if isinstance(r, Exception):
                    round_failed += 1
                    continue
                latency, success, out_tok, in_tok = r
                if success:
                    round_out_tokens += out_tok
                    round_in_tokens += in_tok
                else:
                    round_failed += 1

            total_output_tokens += round_out_tokens
            total_input_tokens += round_in_tokens
            failed += round_failed
            round_tps = round_out_tokens / wall_time if wall_time > 0 else 0
            round_tps_list.append(round_tps)
            print(f"    Round {i+1}: {round_out_tokens} out + {round_in_tokens} in tokens "
                  f"in {wall_time:.1f}s = {round_tps:.1f} tok/s "
                  f"(batch={batch_size}, failed={round_failed})")

        avg_tps = sum(round_tps_list) / len(round_tps_list) if round_tps_list else 0
        completed_rounds = len(round_tps_list)
        successful = completed_rounds * batch_size - failed

        if timed_out and completed_rounds > 0:
            print(f"    Using {completed_rounds}/{NUM_REQUESTS} completed rounds "
                  f"(avg {avg_tps:.1f} tok/s)")

        return BenchmarkResult(
            config_id="",
            batch_size=batch_size,
            successful=successful,
            failed=failed,
            timed_out=timed_out,
            tokens_per_second=avg_tps,
            total_output_tokens=total_output_tokens,
            total_input_tokens=total_input_tokens,
            wall_time_seconds=total_wall_time,
        )


def stop_container(container_name: str):
    subprocess.run(['docker', 'stop', container_name], capture_output=True)


async def main():
    print("=" * 70)
    print("Hardware Configuration Throughput Benchmark")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Rounds per batch: {NUM_REQUESTS} (+ {WARMUP_REQUESTS} warmup)")
    print(f"Timeout per batch round: {BATCH_TIMEOUT}s")
    print()

    # results[config_id][batch_size] = BenchmarkResult
    results = {}

    for config in CONFIGS:
        print(f"\n{'='*50}")
        print(f"Config: {config.config_id}")
        print(f"{'='*50}")
        results[config.config_id] = {}
        port = get_free_port()
        container_name = start_container(config, port)

        if not container_name:
            print(f"  FAILED to start {config.config_id}")
            continue

        try:
            print(f"  Waiting for container to be ready...")
            if not await wait_for_container(port):
                print(f"  ERROR: Container failed to start")
                continue

            config_start = time.perf_counter()
            for batch_size in BATCH_SIZES:
                print(f"\n  --- {config.config_id} | batch_size={batch_size} ---")
                result = await benchmark_batch(port, batch_size)
                result.config_id = config.config_id
                results[config.config_id][batch_size] = result
                await asyncio.sleep(1)
            config_wall = time.perf_counter() - config_start
            print(f"\n  Total time for {config.config_id}: {config_wall:.1f}s")
        finally:
            print(f"  Stopping container...")
            stop_container(container_name)
            await asyncio.sleep(2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Aggregate Tokens/sec by Configuration and Batch Size")
    print("=" * 70)
    header = f"{'Config':<12}"
    for bs in BATCH_SIZES:
        header += f"{'batch=' + str(bs):<18}"
    print(header)
    print("-" * 70)

    for config in CONFIGS:
        line = f"{config.config_id:<12}"
        for bs in BATCH_SIZES:
            r = results.get(config.config_id, {}).get(bs)
            if r:
                line += f"{r.tokens_per_second:>8.1f} t/s      "
            else:
                line += f"{'N/A':>8}          "
        print(line)

    # Save to JSON
    output = {
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "batch_sizes": BATCH_SIZES,
        "batch_timeout_seconds": BATCH_TIMEOUT,
        "rounds_per_batch": NUM_REQUESTS,
        "warmup_requests": WARMUP_REQUESTS,
        "prompt": PROMPT,
        "results": {}
    }
    for cid, batch_results in results.items():
        output["results"][cid] = {}
        for bs, r in batch_results.items():
            output["results"][cid][str(bs)] = {
                "tokens_per_second": r.tokens_per_second,
                "total_output_tokens": r.total_output_tokens,
                "total_input_tokens": r.total_input_tokens,
                "wall_time_seconds": round(r.wall_time_seconds, 2),
                "successful": r.successful,
                "failed": r.failed,
                "timed_out": r.timed_out,
            }

    with open("throughput_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to throughput_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
