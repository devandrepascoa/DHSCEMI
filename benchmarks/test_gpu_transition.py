#!/usr/bin/env python3
"""
Quick test: start on gpu_25, send saturating load, verify gpu_25 → gpu_100 transition.

Usage:
    uv run python benchmarks/test_gpu_transition.py
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
from pathlib import Path

import aiohttp

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MAX_TOKENS = 256
TIMEOUT = 600  # 10 min max


def get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


async def wait_healthy(base_url: str, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def get_status(base_url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/status", timeout=aiohttp.ClientTimeout(total=5)) as r:
            return await r.json()


async def send_request(base_url: str, session: aiohttp.ClientSession) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Write a poem"}],
        "max_tokens": MAX_TOKENS,
    }
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as r:
            await r.json()
    except Exception as e:
        log(f"  req error: {e}")


async def main() -> None:
    port = get_free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())
    env["E2E_MODEL_NAME"] = MODEL
    env["E2E_INITIAL_CONFIG"] = "gpu_25"
    env["E2E_COOLDOWN"] = "0"

    log(f"Starting server on port {port} with initial config gpu_25...")

    # Start the server — let output go to a log file
    server_log = Path("benchmarks/scaling_demo_logs") / "gpu_transition_test.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_log_fh = open(server_log, "w")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "benchmarks.scaling_demo_server:app",
         "--host", "0.0.0.0", "--port", str(port)],
        env=env,
        stdout=server_log_fh,
        stderr=subprocess.STDOUT,
    )

    try:
        log("Waiting for server health...")
        if not await wait_healthy(base_url):
            log("FAIL: Server didn't become healthy")
            return

        # Check initial config
        status = await get_status(base_url)
        model_info = list(status["models"].values())[0]
        initial_config = model_info["config_id"]
        log(f"Initial config: {initial_config}")

        if initial_config != "gpu_25":
            log(f"FAIL: Expected gpu_25, got {initial_config}")
            return

        log("Sending saturating load (8 concurrent workers) to trigger gpu_25 → gpu_100...")
        start = time.time()
        found_gpu_100 = False

        async with aiohttp.ClientSession() as session:
            workers = set()

            while time.time() - start < TIMEOUT:
                # Keep 8 concurrent requests going
                while len(workers) < 8:
                    workers.add(asyncio.create_task(send_request(base_url, session)))

                # Wait for any to complete
                done, workers = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)

                # Check status
                try:
                    status = await get_status(base_url)
                    model_info = list(status["models"].values())[0]
                    config = model_info["config_id"]
                    ema = model_info.get("throughput_ema", 0)
                    cap = model_info.get("capacity", 0)
                    pct = (ema / cap * 100) if cap > 0 else 0
                    elapsed = time.time() - start
                    log(f"  [{elapsed:.0f}s] config={config} ema={ema:.1f} cap={cap} ({pct:.1f}%)")

                    if config == "gpu_100":
                        log(f"SUCCESS: Scaled to gpu_100 after {elapsed:.0f}s")
                        found_gpu_100 = True
                        # Cancel remaining workers
                        for w in workers:
                            w.cancel()
                        break
                except Exception:
                    pass

        if not found_gpu_100:
            log(f"FAIL: Did not scale to gpu_100 within {TIMEOUT}s")

    finally:
        log("Stopping server...")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        server_log_fh.close()

        # Print last lines of server log for debugging
        log(f"Server log: {server_log}")
        with open(server_log) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(f"  [SERVER] {line.rstrip()}", flush=True)

        # Clean up any leftover containers
        log("Cleaning up containers...")
        subprocess.run(
            "docker ps -q --filter name=llama- | xargs -r docker stop",
            shell=True, capture_output=True, timeout=60,
        )


if __name__ == "__main__":
    asyncio.run(main())
