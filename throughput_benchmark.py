#!/usr/bin/env python3
"""
Benchmark throughput for different hardware configurations.
Tests CPU and GPU configs with varying batch sizes to measure aggregate tokens/second.
"""
from __future__ import annotations

import argparse
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

CONFIG_PATH = Path(__file__).parent / "hardware_configs.json"


@dataclass
class HardwareConfig:
    config_id: str
    cpu_cores: Optional[int] = None
    gpu_percentage: Optional[int] = None
    memory: str = "8g"
    parallel_slots: int = 32

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
            if self.gpu_percentage and self.gpu_percentage < 100:
                args.extend(['-e', f'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={self.gpu_percentage}'])
        else:
            args.extend(['--cpus', str(self.cpu_cores)])
        return args

    def server_args(self) -> List[str]:
        if self.is_gpu:
            return ['-ngl', '99']
        else:
            return ['--threads', str(self.cpu_cores)]


def load_configs_from_json() -> List[HardwareConfig]:
    """Load benchmark configs from hardware_configs.json."""
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    return [
        HardwareConfig(
            config_id=c["config_id"],
            cpu_cores=c.get("cpu_cores"),
            gpu_percentage=c.get("gpu_percentage"),
            memory=c.get("memory", "8g"),
            parallel_slots=c.get("parallel_slots", 32),
        )
        for c in data["configs"]
    ]


# Configurations to benchmark — loaded from hardware_configs.json
CONFIGS = load_configs_from_json()

BATCH_SIZES = [1, 4, 16, 32]

# Adaptive batch sizing: stop when throughput improvement drops below this ratio
PLATEAU_THRESHOLD = 0.05  # 5% improvement minimum to keep going
MAX_BATCH_SIZE = 512

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


def start_container(config: HardwareConfig, port: int, parallel: int = 32) -> Optional[str]:
    """Start a llama.cpp container with specified config and parallel slots."""
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
        '--parallel', str(parallel),
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
    parser = argparse.ArgumentParser(description="Benchmark throughput for hardware configs")
    parser.add_argument("--config", help="Run only this config_id (e.g. gpu_25)")
    parser.add_argument("--no-plateau", action="store_true", help="Disable plateau detection, test all batch sizes up to max")
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH_SIZE, help="Maximum batch size to test")
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.config:
        configs_to_run = [c for c in CONFIGS if c.config_id == args.config]
        if not configs_to_run:
            print(f"Unknown config: {args.config}. Available: {[c.config_id for c in CONFIGS]}")
            return

    print("=" * 70)
    print("Hardware Configuration Throughput Benchmark (Adaptive Batch Sizing)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Initial batch sizes: {BATCH_SIZES}")
    print(f"Plateau threshold: {PLATEAU_THRESHOLD*100:.0f}% improvement minimum")
    print(f"Max batch size: {MAX_BATCH_SIZE}")
    print(f"Rounds per batch: {NUM_REQUESTS} (+ {WARMUP_REQUESTS} warmup)")
    print(f"Timeout per batch: {BATCH_TIMEOUT}s")
    if args.config:
        print(f"Running only: {args.config}")
    print()

    # results[config_id][batch_size] = BenchmarkResult
    results = {}

    for config in configs_to_run:
        print(f"\n{'='*50}")
        print(f"Config: {config.config_id} (parallel_slots={config.parallel_slots})")
        print(f"{'='*50}")
        results[config.config_id] = {}
        port = get_free_port()
        container_name = None

        try:
            config_start = time.perf_counter()
            prev_tps = 0.0
            batch_size = 1
            current_parallel = 0  # track current --parallel setting

            while batch_size <= args.max_batch:
                # Restart container if we need more parallel slots
                if batch_size > current_parallel:
                    if current_parallel > 0:
                        print(f"  Restarting container with --parallel {batch_size}...")
                        stop_container(container_name)
                        await asyncio.sleep(2)
                        port = get_free_port()
                    container_name = start_container(config, port, parallel=batch_size)
                    if not container_name:
                        print(f"  FAILED to restart container for batch_size={batch_size}")
                        break
                    print(f"  Waiting for container (--parallel {batch_size})...")
                    if not await wait_for_container(port):
                        print(f"  ERROR: Container failed to start")
                        break
                    current_parallel = batch_size

                print(f"\n  --- {config.config_id} | batch_size={batch_size} (parallel={current_parallel}) ---")
                result = await benchmark_batch(port, batch_size)
                result.config_id = config.config_id
                results[config.config_id][batch_size] = result

                curr_tps = result.tokens_per_second

                # Check for plateau
                if prev_tps > 0 and curr_tps > 0:
                    improvement = (curr_tps - prev_tps) / prev_tps
                    print(f"  Throughput: {curr_tps:.1f} tok/s "
                          f"(+{improvement*100:.1f}% vs batch={batch_size // 2 if batch_size > 1 else 1})")
                    if not args.no_plateau and improvement < PLATEAU_THRESHOLD:
                        print(f"  PLATEAU reached at batch_size={batch_size} "
                              f"({improvement*100:.1f}% < {PLATEAU_THRESHOLD*100:.0f}% threshold)")
                        break
                elif result.timed_out or curr_tps == 0:
                    print(f"  STOPPED: timeout or zero throughput at batch_size={batch_size}")
                    break
                else:
                    print(f"  Throughput: {curr_tps:.1f} tok/s (baseline)")

                prev_tps = curr_tps
                # Double the batch size
                batch_size = batch_size * 2 if batch_size >= 4 else batch_size * 4
                await asyncio.sleep(1)

            config_wall = time.perf_counter() - config_start
            print(f"\n  Total time for {config.config_id}: {config_wall:.1f}s")
        finally:
            print(f"  Stopping container...")
            if container_name:
                stop_container(container_name)
            await asyncio.sleep(2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Aggregate Tokens/sec by Configuration and Batch Size")
    print("=" * 70)

    # Collect all batch sizes seen
    all_batch_sizes = sorted({bs for cr in results.values() for bs in cr})

    header = f"{'Config':<12}"
    for bs in all_batch_sizes:
        header += f"{'batch=' + str(bs):<18}"
    header += f"{'peak':>10}"
    print(header)
    print("-" * (12 + 18 * len(all_batch_sizes) + 10))

    for config in configs_to_run:
        line = f"{config.config_id:<12}"
        peak_tps = 0.0
        for bs in all_batch_sizes:
            r = results.get(config.config_id, {}).get(bs)
            if r:
                line += f"{r.tokens_per_second:>8.1f} t/s      "
                peak_tps = max(peak_tps, r.tokens_per_second)
            else:
                line += f"{'—':>8}          "
        line += f"{peak_tps:>8.1f} t/s"
        print(line)

    # Save to JSON — merge with existing results if running a single config
    output_path = Path("throughput_benchmark_results.json")
    if args.config and output_path.exists():
        with open(output_path) as f:
            output = json.load(f)
    else:
        output = {
            "model": MODEL_NAME,
            "max_tokens": MAX_TOKENS,
            "adaptive_batch_sizing": True,
            "plateau_threshold": PLATEAU_THRESHOLD,
            "max_batch_size": MAX_BATCH_SIZE,
            "rounds_per_batch": NUM_REQUESTS,
            "warmup_requests": WARMUP_REQUESTS,
            "prompt": PROMPT,
            "results": {}
        }
    for cid, batch_results in results.items():
        output["results"][cid] = {}
        peak = 0.0
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
            peak = max(peak, r.tokens_per_second)
        output["results"][cid]["peak_tokens_per_second"] = round(peak, 2)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
