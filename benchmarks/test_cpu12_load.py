#!/usr/bin/env python3
"""
Quick test: start a cpu_12 container directly, blast it with 8 saturated
workers for 3 min, poll /slots every 1s, and generate a plot showing
live delta-based throughput from n_decoded.
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
from typing import Dict, List

import aiohttp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_FILE = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
MODEL = MODEL_FILE.replace(".gguf", "")
MAX_TOKENS = 256
DURATION = 180
CONCURRENCY = 8
CONTAINER_NAME = "llama-cpu12-test"
CONTAINER_PORT = 8090
MODELS_DIR = str(Path("models").resolve())
OUT_DIR = Path("benchmarks/thesis_figures")


def log(msg: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


async def start_container() -> bool:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   capture_output=True)
    cmd = [
        "docker", "run", "--rm", "-d",
        "--name", CONTAINER_NAME,
        "-v", "%s:/models:ro" % MODELS_DIR,
        "-p", "%d:8080" % CONTAINER_PORT,
        "--cpus", "12", "--memory", "8g",
        "ghcr.io/ggml-org/llama.cpp:full",
        "--server",
        "-m", "/models/%s" % MODEL_FILE,
        "--host", "0.0.0.0", "--port", "8080",
        "--threads", "12", "--parallel", "32",
        "--metrics", "--slots",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log("Failed to start container: %s" % proc.stderr[:300])
        return False

    log("Container started, waiting for health...")
    for _ in range(90):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as s:
                async with s.get("http://localhost:%d/health" % CONTAINER_PORT) as r:
                    if r.status == 200:
                        log("Container healthy")
                        return True
        except Exception:
            pass
        await asyncio.sleep(2)
    log("Container not healthy after 180s")
    return False


async def worker(session, url, stats, stop):
    payload = {
        "prompt": "Explain vertical scaling in one paragraph.",
        "n_predict": MAX_TOKENS,
        "temperature": 0.7,
    }
    while not stop.is_set():
        try:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                await resp.json()
                if resp.status == 200:
                    stats["ok"] += 1
                else:
                    stats["fail"] += 1
        except Exception:
            stats["fail"] += 1
        await asyncio.sleep(0.05)


async def poll_slots(samples: List[Dict], start_time: float, stop: asyncio.Event):
    """Poll /slots every 1s and record n_decoded per slot."""
    url = "http://localhost:%d/slots" % CONTAINER_PORT
    while not stop.is_set():
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as s:
                async with s.get(url) as r:
                    slots_data = await r.json()
            n_decoded_total = 0
            n_processing = 0
            for slot in slots_data:
                is_proc = slot.get("is_processing", False)
                if is_proc:
                    n_processing += 1
                next_token = slot.get("next_token", {})
                if isinstance(next_token, list) and len(next_token) > 0:
                    next_token = next_token[0]
                if isinstance(next_token, dict):
                    n_decoded_total += next_token.get("n_decoded", 0)
            samples.append({
                "_elapsed": time.time() - start_time,
                "n_decoded_total": n_decoded_total,
                "n_processing": n_processing,
            })
        except Exception as e:
            log("slots poll error: %s" % str(e)[:100])
        await asyncio.sleep(1)


async def poll_metrics(samples: List[Dict], start_time: float, stop: asyncio.Event):
    """Also poll /metrics for comparison."""
    url = "http://localhost:%d/metrics" % CONTAINER_PORT
    while not stop.is_set():
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as s:
                async with s.get(url) as r:
                    text = await r.text()
            result = {}
            for line in text.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        result[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
            result["_elapsed"] = time.time() - start_time
            samples.append(result)
        except Exception:
            pass
        await asyncio.sleep(1)


EMA_ALPHA = 2.0 / (300 + 1)  # ~5min EMA window


def generate_plot(slot_samples: List[Dict], metric_samples: List[Dict]) -> None:
    if len(slot_samples) < 3:
        log("Not enough slot samples for plot")
        return

    t = np.array([s["_elapsed"] for s in slot_samples])
    n_decoded = np.array([s["n_decoded_total"] for s in slot_samples])
    n_processing = np.array([s["n_processing"] for s in slot_samples])

    # Delta-based throughput from /slots
    delta_tps = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            delta_tps[i] = (n_decoded[i] - n_decoded[i - 1]) / dt

    # EMA of delta throughput
    ema_tps = np.zeros_like(t)
    ema_val = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt > 0:
            decay = (1.0 - EMA_ALPHA) ** dt
            ema_val = ema_val * decay + EMA_ALPHA * delta_tps[i]
        ema_tps[i] = ema_val

    # /metrics data for comparison
    mt = np.array([s["_elapsed"] for s in metric_samples]) if metric_samples else np.array([])
    metrics_gauge = np.array([
        s.get("llamacpp:predicted_tokens_seconds", 0) for s in metric_samples
    ]) if metric_samples else np.array([])
    metrics_total = np.array([
        s.get("llamacpp:tokens_predicted_total", 0) for s in metric_samples
    ]) if metric_samples else np.array([])

    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150,
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    # Panel 1: n_decoded_total from /slots
    axes[0].plot(t, n_decoded, color="#333", linewidth=1.5)
    axes[0].set_ylabel("n_decoded_total (/slots)")
    axes[0].set_title(
        "cpu_12 load test — /slots vs /metrics (8 saturated workers, 3 min)"
    )

    # Panel 2: delta throughput + EMA + metrics gauge
    axes[1].plot(t, delta_tps, color="#e15759", linewidth=1, alpha=0.5,
                 label="delta(n_decoded)/dt per poll (/slots)")
    axes[1].plot(t, ema_tps, color="#4e79a7", linewidth=2.5,
                 label="EMA of /slots delta (alpha=%.4f)" % EMA_ALPHA)
    if len(mt) > 0:
        axes[1].plot(mt, metrics_gauge, color="#59a14f", linewidth=1.5,
                     linestyle="--",
                     label="predicted_tokens_seconds (/metrics gauge)")
    axes[1].axhline(47, color="orange", linestyle=":", alpha=0.7,
                    label="cpu_12 capacity (47)")
    axes[1].axhline(47 * 0.8, color="red", linestyle=":", alpha=0.5,
                    label="scale-up threshold (80%%=%.1f)" % (47 * 0.8))
    axes[1].set_ylabel("Throughput (tok/s)")
    axes[1].legend(fontsize=8, loc="upper right")

    # Panel 3: n_processing (busy slots)
    axes[2].plot(t, n_processing, color="#f28e2b", linewidth=1.5)
    axes[2].set_ylabel("n_processing (/slots)")

    # Panel 4: /metrics tokens_predicted_total for comparison
    if len(mt) > 0:
        axes[3].plot(mt, metrics_total, color="#333", linewidth=1.5)
    axes[3].set_ylabel("tokens_predicted_total (/metrics)")

    # Panel 5: comparison — /slots delta vs /metrics delta
    if len(mt) > 2:
        metrics_delta = np.zeros_like(mt)
        for i in range(1, len(mt)):
            dt = mt[i] - mt[i - 1]
            if dt > 0:
                metrics_delta[i] = (metrics_total[i] - metrics_total[i - 1]) / dt
        axes[4].plot(mt, metrics_delta, color="#59a14f", linewidth=1,
                     alpha=0.6, label="/metrics delta(total)/dt")
    axes[4].plot(t, delta_tps, color="#e15759", linewidth=1, alpha=0.6,
                 label="/slots delta(n_decoded)/dt")
    axes[4].axhline(47, color="orange", linestyle=":", alpha=0.7)
    axes[4].set_ylabel("Throughput comparison")
    axes[4].set_xlabel("Time (seconds)")
    axes[4].legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "cpu12_slots_test.png"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    log("Plot saved to %s" % out)


async def main():
    if not await start_container():
        return

    log("Starting %d saturated workers for %ds" % (CONCURRENCY, DURATION))

    stats = {"ok": 0, "fail": 0}
    slot_samples: List[Dict] = []
    metric_samples: List[Dict] = []
    stop = asyncio.Event()
    start_time = time.time()

    url = "http://localhost:%d/completion" % CONTAINER_PORT
    session = aiohttp.ClientSession()

    tasks = [
        asyncio.create_task(poll_slots(slot_samples, start_time, stop)),
        asyncio.create_task(poll_metrics(metric_samples, start_time, stop)),
    ]
    for _ in range(CONCURRENCY):
        tasks.append(asyncio.create_task(worker(session, url, stats, stop)))

    await asyncio.sleep(DURATION)
    stop.set()

    for t_task in tasks:
        t_task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await session.close()

    elapsed = time.time() - start_time
    log("Done. %d ok, %d fail in %.0fs" % (stats["ok"], stats["fail"], elapsed))
    log("Slot samples: %d, Metric samples: %d" % (
        len(slot_samples), len(metric_samples)))

    generate_plot(slot_samples, metric_samples)

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    log("Container removed")


if __name__ == "__main__":
    asyncio.run(main())
