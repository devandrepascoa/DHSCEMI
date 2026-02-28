#!/usr/bin/env python3
"""
Quick test: start on cpu_16, scale down to cpu_4 with rate-limited load,
verify cpu_4 stays stable (no re-scale-up).

Expected: cpu_16 → cpu_4, then stable on cpu_4 for 5+ minutes.
Total runtime: ~10 minutes.
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


async def _send_request(session: aiohttp.ClientSession, base_url: str):
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
                log(f"  OK  gen_tps={tps:.1f}")
                return True
            else:
                log(f"  FAIL status={resp.status}")
                return False
    except Exception as e:
        log(f"  ERR {e}")
        return False


async def main():
    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())
    env["E2E_INITIAL_CONFIG"] = "cpu_16"  # Start on cpu_16
    env["E2E_COOLDOWN"] = "60"
    env["E2E_COOLDOWN_DOWN"] = "240"
    env["E2E_EMA_WINDOW"] = "30"
    env["E2E_MIN_TPS"] = "10.0"
    env["E2E_SCALE_DOWN_CONCURRENCY"] = "5.0"
    env["E2E_RECENT_ACTIVITY_WINDOW"] = "15.0"

    log(f"Starting server on port {port} with INITIAL_CONFIG=cpu_16")

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

    # Phase 1: Rate-limited load on cpu_16 (1 worker, ~3 rpm) for 5 min
    # This should trigger scale-down to cpu_4 after cooldown
    # Phase 2: Continue same load on cpu_4 for 5 min — must stay stable
    total_duration = 600  # 10 minutes total
    phase_deadline = time.time() + total_duration
    interval = 20.0  # ~3 rpm

    scaling_events = []
    prev_config = "cpu_16"

    async with aiohttp.ClientSession() as session:
        while time.time() < phase_deadline:
            req_start = time.time()

            # Check status
            status = await _poll_status(base_url)
            if status:
                model_info = status.get("models", {}).get(MODEL, {})
                config_id = model_info.get("config_id", "?")
                per_req_ema = model_info.get("per_request_tps_ema", 0)
                active_ema = model_info.get("active_requests_ema", 0)
                elapsed = time.time() - (phase_deadline - total_duration)

                if config_id != prev_config:
                    event = f"{prev_config} → {config_id}"
                    scaling_events.append((elapsed, event))
                    log(f"*** SCALE EVENT: {event} at {elapsed:.0f}s ***")
                    prev_config = config_id

                log(f"[{elapsed:.0f}s] config={config_id} per_req_ema={per_req_ema:.2f} active_ema={active_ema:.2f}")

            # Send request
            await _send_request(session, base_url)

            # Rate limit
            elapsed_req = time.time() - req_start
            wait = max(0, interval - elapsed_req)
            if wait > 0:
                await asyncio.sleep(wait)

    # Summary
    log("=" * 60)
    log(f"Scaling events: {len(scaling_events)}")
    for t, ev in scaling_events:
        log(f"  {t:.0f}s: {ev}")

    if len(scaling_events) == 1 and "cpu_16 → cpu_4" in scaling_events[0][1]:
        log("PASS — clean cpu_16 → cpu_4 transition, no re-scale-up")
    elif len(scaling_events) == 0:
        log("FAIL — no scale-down happened (need more time or different params)")
    else:
        log("FAIL — unexpected scaling events")

    # Cleanup
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    for name in [n.strip() for n in result.stdout.splitlines() if n.strip()]:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    log("Done")


if __name__ == "__main__":
    asyncio.run(main())
