"""
Dedicated server entrypoint for the scaling demo benchmark.

CPU-only configs: cpu_4, cpu_8, cpu_12 with short cooldown and demand
window so scaling transitions happen within minutes.

Thresholds (demand in tok/s to trigger scale-up):
  - cpu_4 throughput: 12 tok/s → demand > 12 triggers scale to cpu_8
  - cpu_8 throughput: 18 tok/s → demand > 18 triggers scale to cpu_12
  - cpu_12 throughput: 22 tok/s (max CPU tier)

Usage:
    uv run uvicorn benchmarks.scaling_demo_server:app --port <PORT>
"""
from __future__ import annotations

import os
import logging
from typing import List
from contextlib import asynccontextmanager

from main_cost_aware import (
    HardwareConfig,
    CostAwareAutoscaler,
    DemandTracker,
    Container,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Message,
    DEFAULT_THROUGHPUT,
)

from fastapi import FastAPI, HTTPException
import aiohttp
import uuid
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CPU-only configs for clean scaling transitions
CONFIGS: List[HardwareConfig] = [
    HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40),
    HardwareConfig(cpu_cores=8, memory="16g", hourly_cost=0.80),
    HardwareConfig(cpu_cores=12, memory="24g", hourly_cost=1.20),
]

DEFAULT_THROUGHPUT["cpu_12"] = 22.0

COOLDOWN = 300
DEMAND_WINDOW = 60

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
MODEL_NAME = os.environ.get("E2E_MODEL_NAME", "")

autoscaler: CostAwareAutoscaler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global autoscaler
    autoscaler = CostAwareAutoscaler(
        configs=CONFIGS,
        cooldown_seconds=COOLDOWN,
        models_dir=MODELS_DIR,
    )
    autoscaler.demand_tracker = DemandTracker(window_seconds=DEMAND_WINDOW)

    if MODEL_NAME:
        model_path = autoscaler.get_model_path(MODEL_NAME)
        if model_path:
            cheapest = min(CONFIGS, key=lambda c: c.hourly_cost)
            port = autoscaler._get_port()
            container = Container(MODEL_NAME, model_path, cheapest, port)
            if await container.start():
                autoscaler.containers[MODEL_NAME] = container
                autoscaler.current_config[MODEL_NAME] = cheapest
                autoscaler.last_scale_time[MODEL_NAME] = autoscaler.clock()
                logger.info(f"Started {MODEL_NAME} on {cheapest.config_id()}")
    else:
        await autoscaler.initialize()

    yield
    await autoscaler.cleanup()


app = FastAPI(title="Scaling Demo Server", lifespan=lifespan)


@app.get("/health")
async def health():
    ready = sum(1 for c in autoscaler.containers.values() if c.is_ready)
    return {
        "status": "healthy" if ready > 0 else "down",
        "ready_containers": ready,
        "models": list(autoscaler.containers.keys()),
    }


@app.get("/status")
async def status():
    return autoscaler.get_status()


@app.get("/v1/models")
async def list_models():
    return {"models": list(autoscaler.containers.keys())}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    container = await autoscaler.get_container(request.model)
    if not container:
        raise HTTPException(404, f"Model '{request.model}' not found or not ready")

    async with container.lock:
        container.active_requests += 1
        container.total_requests += 1

    try:
        return await _non_stream_completion(request, container)
    finally:
        async with container.lock:
            container.active_requests = max(0, container.active_requests - 1)


async def _non_stream_completion(request, container):
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{container.get_endpoint()}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                raise HTTPException(resp.status, "Container error")

            result = await resp.json()

            choices = []
            for i, choice in enumerate(result.get("choices", [])):
                msg = choice.get("message", {})
                choices.append(ChatCompletionChoice(
                    index=i,
                    message={
                        "role": msg.get("role", "assistant"),
                        "content": msg.get("content", ""),
                    },
                    finish_reason=choice.get("finish_reason"),
                ))

            usage = result.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            if total_tokens > 0:
                autoscaler.demand_tracker.record_tokens(request.model, total_tokens)

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=usage,
            )
