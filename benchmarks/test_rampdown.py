#!/usr/bin/env python3
"""
Quick ramp-down-only test.
Starts server on gpu_100, then runs ramp-down phases to reach cpu_4.

Expected: gpu_100 → cpu_48 → cpu_16 → cpu_4
"""
from __future__ import annotations

import asyncio
import json
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

PHASES = [
    # Warm-up on gpu_100: 6 workers for 60s to seed EMA
    ("warmup",       60,   6,   0),
    # Ramp-down 1: 6 workers, expect gpu_100→cpu_48 (maybe →cpu_16 too)
    ("ramp-down 1", 360,   6,   0),
    # Ramp-down 2: 1 worker at 2 rpm, drop remaining to cpu_4
    ("ramp-down 2", 360,   1,   2),
]

EXPECTED = ["gpu_100", "cpu_48", "cpu_16", "cpu_4"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def log(msg: str) -> None:
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


async def _send_request(session, base_url, phase):
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
            if resp.status != 200:
                log(f"  [{phase}] FAIL status={resp.status}")
                return False
            tps = raw.get("timings", {}).get("predicted_per_second", 0)
            tok = raw.get("usage", {}).get("completion_tokens", 0)
            log(f"  [{phase}] OK tok={tok} tps={tps:.1f}")
            return True
    except Exception as e:
        log(f"  [{phase}] ERR {e}")
        return False


async def _run_phase(base_url, phase_name, duration, concurrency, rpm):
    if concurrency > 0 and rpm > 0:
        worker_interval = 60.0 * concurrency / rpm
    else:
        worker_interval = 0.0

    log(f"=== PHASE: {phase_name} | {duration}s | workers={concurrency} rpm={rpm} ===")

    deadline = time.time() + duration
    stop = asyncio.Event()
    configs_seen = []

    async def worker(session):
        while not stop.is_set() and time.time() < deadline:
            t0 = time.time()
            await _send_request(session, base_url, phase_name)
            if worker_interval > 0:
                wait = max(0, worker_interval - (time.time() - t0))
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                await asyncio.sleep(0.1)

    async def monitor():
        while not stop.is_set() and time.time() < deadline:
            status = await _poll_status(base_url)
            if status:
                mi = status.get("models", {}).get(MODEL, {})
                cid = mi.get("config_id", "?")
                pr_ema = mi.get("per_request_tps_ema", 0)
                ar_ema = mi.get("active_requests_ema", 0)
                active = mi.get("active_requests", 0)
                elapsed = time.time() - start_time
                log(f"  STATUS config={cid} per_req_ema={pr_ema:.1f} active={active} ar_ema={ar_ema:.1f} elapsed={elapsed:.0f}s")
                if not configs_seen or configs_seen[-1] != cid:
                    configs_seen.append(cid)
                    if len(configs_seen) > 1:
                        log(f"  >>> SCALE: {configs_seen[-2]} → {cid}")
            await asyncio.sleep(5)

    tasks = [asyncio.create_task(monitor())]
    session = None
    if concurrency > 0:
        session = aiohttp.ClientSession()
        for _ in range(concurrency):
            tasks.append(asyncio.create_task(worker(session)))

    remaining = deadline - time.time()
    if remaining > 0:
        await asyncio.sleep(remaining)

    stop.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if session:
        await session.close()

    return configs_seen


start_time = 0


async def main():
    global start_time

    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())
    env["E2E_INITIAL_CONFIG"] = "gpu_100"  # Start at top
    env["E2E_COOLDOWN"] = "60"
    env["E2E_COOLDOWN_DOWN"] = "240"
    env["E2E_EMA_WINDOW"] = "30"
    env["E2E_MIN_TPS"] = "10.0"
    env["E2E_SCALE_DOWN_CONCURRENCY"] = "5.0"
    env["E2E_RECENT_ACTIVITY_WINDOW"] = "15.0"

    log(f"Starting server on port {port} with initial_config=gpu_100")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "main_cost_aware:app",
         "--host", "0.0.0.0", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Wait for healthy
    deadline = time.time() + 180
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
        log("Server not healthy in 180s")
        proc.kill()
        proc.wait()
        return

    log("Server healthy")
    start_time = time.time()

    all_configs = []
    try:
        for phase_name, duration, concurrency, rpm in PHASES:
            seen = await _run_phase(base_url, phase_name, duration, concurrency, rpm)
            for c in seen:
                if not all_configs or all_configs[-1] != c:
                    all_configs.append(c)
    finally:
        log("Stopping server...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Cleanup containers
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        for name in [n.strip() for n in result.stdout.splitlines() if n.strip()]:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    log(f"Observed: {' → '.join(all_configs)}")
    log(f"Expected: {' → '.join(EXPECTED)}")
    if all_configs == EXPECTED:
        log("PASS")
    else:
        log("FAIL")


if __name__ == "__main__":
    asyncio.run(main())
