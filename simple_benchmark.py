#!/usr/bin/env python3
"""Simple benchmark to measure inference time per request."""
import asyncio
import time
import aiohttp
import statistics
from dataclasses import dataclass

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
PROXY_URL = "http://localhost:8000"
NUM_REQUESTS = 10
MAX_TOKENS = 50


@dataclass
class BenchmarkResult:
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


async def send_request(session: aiohttp.ClientSession, request_id: int) -> tuple[int, float, bool]:
    """Send a single chat completion request, return (request_id, latency_ms, success)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status == 200:
                await resp.json()
                latency = (time.perf_counter() - start) * 1000
                return request_id, latency, True
            else:
                text = await resp.text()
                print(f"Request {request_id} failed with status {resp.status}: {text}")
                return request_id, 0, False
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return request_id, 0, False


async def run_benchmark(num_requests: int, concurrency: int = 1) -> BenchmarkResult:
    """Run benchmark with specified concurrency."""
    latencies = []
    failed = 0

    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, num_requests, concurrency):
            batch_end = min(batch_start + concurrency, num_requests)
            tasks = [send_request(session, i) for i in range(batch_start, batch_end)]
            results = await asyncio.gather(*tasks)

            for req_id, latency, success in results:
                if success:
                    latencies.append(latency)
                    print(f"  Request {req_id}: {latency:.1f}ms")
                else:
                    failed += 1

    if not latencies:
        return BenchmarkResult(num_requests, 0, failed, 0, 0, 0, 0, 0)

    sorted_lat = sorted(latencies)
    p50_idx = int(len(sorted_lat) * 0.5)
    p95_idx = int(len(sorted_lat) * 0.95)

    return BenchmarkResult(
        total_requests=num_requests,
        successful=len(latencies),
        failed=failed,
        avg_latency_ms=statistics.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        p50_latency_ms=sorted_lat[p50_idx],
        p95_latency_ms=sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
    )


async def main():
    print("=== Inference Time Benchmark ===")
    print(f"Model: {MODEL}")
    print(f"Max tokens: {MAX_TOKENS}")
    print()

    # Check health first
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{PROXY_URL}/health") as resp:
                health = await resp.json()
                print(f"Proxy status: {health.get('status')}")
                print(f"Ready containers: {health.get('ready_containers')}")
        except Exception as e:
            print(f"ERROR: Cannot connect to proxy at {PROXY_URL}: {e}")
            return

    print()

    # Sequential benchmark (concurrency=1)
    print(f"--- Sequential Requests (concurrency=1) ---")
    result = await run_benchmark(NUM_REQUESTS, concurrency=1)
    print(f"\nResults:")
    print(f"  Successful: {result.successful}/{result.total_requests}")
    print(f"  Avg latency: {result.avg_latency_ms:.1f}ms")
    print(f"  Min latency: {result.min_latency_ms:.1f}ms")
    print(f"  Max latency: {result.max_latency_ms:.1f}ms")
    print(f"  P50 latency: {result.p50_latency_ms:.1f}ms")
    print(f"  P95 latency: {result.p95_latency_ms:.1f}ms")

    # Concurrent benchmark
    print()
    print(f"--- Concurrent Requests (concurrency=3) ---")
    result = await run_benchmark(NUM_REQUESTS, concurrency=3)
    print(f"\nResults:")
    print(f"  Successful: {result.successful}/{result.total_requests}")
    print(f"  Avg latency: {result.avg_latency_ms:.1f}ms")
    print(f"  Min latency: {result.min_latency_ms:.1f}ms")
    print(f"  Max latency: {result.max_latency_ms:.1f}ms")
    print(f"  P50 latency: {result.p50_latency_ms:.1f}ms")
    print(f"  P95 latency: {result.p95_latency_ms:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
