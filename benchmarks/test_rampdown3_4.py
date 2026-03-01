#!/usr/bin/env python3
"""
Quick test: start on gpu_25, scale down to cpu_16 (with 8 back-to-back workers),
then scale down to cpu_4 (with rate-limited 1 worker at 3 rpm).

Phase 1 (5 min): 8 workers back-to-back on gpu_25
  - gpu_25 with 8: 335.8/8 = 42.0 > 10 → stable
  - Viability cpu_16: 291.2/8 = 36.4 >= 15 → scale down ✓
  - Viability cpu_4:  92.2/8  = 11.5 < 15  → blocked ✗
  → Should land on cpu_16 and stay there.

Phase 2 (5 min): 1 worker at 3 rpm on cpu_16
  - Viability cpu_4: 92.2/1 = 92.2 >= 15 → scale down ✓
  - cpu_4 with 1 req: ~28 tok/s > 10 → stable
  → Should scale to cpu_4 and stay.

Expected sequence: gpu_25 → cpu_16 → cpu_4 (exactly 2 transitions, no re-scale-up)
Total runtime: ~10 min + container startup.
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MAX_TOKENS = 256
SERVER_STARTUP_TIMEOUT = 180


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


async def _poll_status(base_url: str):
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(f"{base_url}/status") as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None


async def _send_request(session: aiohttp.ClientSession, base_url: str, label: str):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Explain vertical scaling in one paragraph."}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            raw = await resp.json()
            if resp.status == 200:
                tps = raw.get("timings", {}).get("predicted_per_second", 0)
                log(f"  [{label}] OK  gen_tps={tps:.1f}")
                return True
            else:
                log(f"  [{label}] FAIL status={resp.status}")
                return False
    except Exception as e:
        log(f"  [{label}] ERR {e}")
        return False


async def _run_phase_backtoback(
    base_url: str, n_workers: int, duration: int,
    scaling_events: list, prev_config: list, start_time: float,
):
    """Run back-to-back workers for a fixed duration."""
    deadline = time.time() + duration
    stop = asyncio.Event()

    async def worker(session: aiohttp.ClientSession, wid: int):
        while not stop.is_set() and time.time() < deadline:
            await _send_request(session, base_url, f"w{wid}")
            await asyncio.sleep(0.1)

    async def monitor():
        while not stop.is_set() and time.time() < deadline:
            status = await _poll_status(base_url)
            if status:
                model_info = status.get("models", {}).get(MODEL, {})
                config_id = model_info.get("config_id", "?")
                per_req_ema = model_info.get("per_request_tps_ema", 0)
                active_ema = model_info.get("active_requests_ema", 0)
                elapsed = time.time() - start_time

                if config_id != prev_config[0]:
                    event = f"{prev_config[0]} → {config_id}"
                    scaling_events.append((elapsed, event))
                    log(f"*** SCALE EVENT: {event} at {elapsed:.0f}s ***")
                    prev_config[0] = config_id

                log(f"[{elapsed:.0f}s] config={config_id} per_req_ema={per_req_ema:.2f} active_ema={active_ema:.2f}")
            await asyncio.sleep(5)

    session = aiohttp.ClientSession()
    tasks = [asyncio.create_task(monitor())]
    for i in range(n_workers):
        tasks.append(asyncio.create_task(worker(session, i)))

    remaining = deadline - time.time()
    if remaining > 0:
        await asyncio.sleep(remaining)

    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await session.close()


async def _run_phase_ratelimited(
    base_url: str, rpm: float, duration: int,
    scaling_events: list, prev_config: list, start_time: float,
):
    """Run rate-limited requests (1 worker) for a fixed duration."""
    deadline = time.time() + duration
    interval = 60.0 / rpm

    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            req_start = time.time()

            status = await _poll_status(base_url)
            if status:
                model_info = status.get("models", {}).get(MODEL, {})
                config_id = model_info.get("config_id", "?")
                per_req_ema = model_info.get("per_request_tps_ema", 0)
                active_ema = model_info.get("active_requests_ema", 0)
                elapsed = time.time() - start_time

                if config_id != prev_config[0]:
                    event = f"{prev_config[0]} → {config_id}"
                    scaling_events.append((elapsed, event))
                    log(f"*** SCALE EVENT: {event} at {elapsed:.0f}s ***")
                    prev_config[0] = config_id

                log(f"[{elapsed:.0f}s] config={config_id} per_req_ema={per_req_ema:.2f} active_ema={active_ema:.2f}")

            await _send_request(session, base_url, "rl")

            elapsed_req = time.time() - req_start
            wait = max(0, interval - elapsed_req)
            if wait > 0:
                await asyncio.sleep(wait)


async def main():
    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())
    env["E2E_INITIAL_CONFIG"] = "gpu_25"
    env["E2E_COOLDOWN"] = "60"
    env["E2E_COOLDOWN_DOWN"] = "240"
    env["E2E_EMA_WINDOW"] = "30"
    env["E2E_MIN_TPS"] = "10.0"
    env["E2E_SCALE_DOWN_CONCURRENCY"] = "5.0"
    env["E2E_RECENT_ACTIVITY_WINDOW"] = "15.0"

    log(f"Starting server on port {port} with INITIAL_CONFIG=gpu_25")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "main_cost_aware:app",
         "--host", "0.0.0.0", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Wait for healthy
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    healthy = False
    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                async with session.get(f"{base_url}/health") as resp:
                    data = await resp.json()
                    if data.get("status") == "healthy":
                        healthy = True
                        break
        except Exception:
            pass
        if proc.poll() is not None:
            log("Server died!")
            return
        await asyncio.sleep(3)

    if not healthy:
        proc.kill()
        proc.wait()
        log("Server not healthy in time")
        return

    log("Server healthy")

    scaling_events = []
    prev_config = ["gpu_25"]  # mutable for closures
    start_time = time.time()

    # Phase 1: 8 back-to-back workers for 5 min
    # Should scale gpu_25 → cpu_16, but NOT cpu_16 → cpu_4
    log("=" * 60)
    log("Phase 1: 8 back-to-back workers (5 min)")
    log("  Expect: gpu_25 → cpu_16, cpu_4 blocked (92.2/8=11.5 < 15)")
    log("=" * 60)
    await _run_phase_backtoback(
        base_url, n_workers=8, duration=300,
        scaling_events=scaling_events, prev_config=prev_config,
        start_time=start_time,
    )

    # Phase 2: 1 worker at 3 rpm for 5 min
    # Should scale cpu_16 → cpu_4 and stay stable
    log("=" * 60)
    log("Phase 2: 1 worker at 3 rpm (5 min)")
    log("  Expect: cpu_16 → cpu_4, stable")
    log("=" * 60)
    await _run_phase_ratelimited(
        base_url, rpm=3, duration=300,
        scaling_events=scaling_events, prev_config=prev_config,
        start_time=start_time,
    )

    # Summary
    log("=" * 60)
    log(f"Scaling events: {len(scaling_events)}")
    for t, ev in scaling_events:
        log(f"  {t:.0f}s: {ev}")

    expected = ["gpu_25 → cpu_16", "cpu_16 → cpu_4"]
    actual = [ev for _, ev in scaling_events]
    if actual == expected:
        log("PASS — clean gpu_25 → cpu_16 → cpu_4, no re-scale-up")
    else:
        log(f"FAIL — expected {expected}, got {actual}")

    # Cleanup
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    for name in [n.strip() for n in result.stdout.splitlines() if n.strip()]:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    log("Done")


if __name__ == "__main__":
    asyncio.run(main())
