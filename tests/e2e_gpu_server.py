"""
Test entrypoint for cost-aware autoscaler GPU E2E tests.

Overrides HARDWARE_CONFIGS to CPU (4, 8, 14 cores) + GPU configs with short
cooldown / demand window so that scaling is observable in seconds.

Usage:
    uv run uvicorn tests.e2e_gpu_server:app --host 0.0.0.0 --port <PORT>

Environment variables (all optional):
    E2E_MODELS_DIR   – path to models directory (default: ./models)
    E2E_MODEL_NAME   – single model file stem to load (default: load all)
"""
from __future__ import annotations

import os
import logging
from typing import List
from contextlib import asynccontextmanager

from main_cost_aware import (  # noqa: F401
    HardwareConfig,
    CostAwareAutoscaler,
    ThroughputTracker,
    Container,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Message,
    MEASURED_THROUGHPUT,
    get_cost_per_token,
)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as _BM
import aiohttp
import uuid
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Override configs for GPU E2E testing: CPU (4, 8, 14) + GPU
# ---------------------------------------------------------------------------
E2E_HARDWARE_CONFIGS: List[HardwareConfig] = [
    HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40),                        # cpu_4
    HardwareConfig(cpu_cores=8, memory="16g", hourly_cost=0.80),                       # cpu_8
    HardwareConfig(cpu_cores=14, memory="24g", hourly_cost=1.40),                      # cpu_14
    HardwareConfig(cpu_cores=2, memory="8g", gpu_percentage=50, hourly_cost=5.00),     # gpu_50
]

# Register throughput for cpu_14 (not in MEASURED_THROUGHPUT)
MEASURED_THROUGHPUT["cpu_14"] = 24.0

E2E_COOLDOWN_SECONDS = 10
E2E_DEMAND_WINDOW_SECONDS = 30

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
SINGLE_MODEL = os.environ.get("E2E_MODEL_NAME", "")

autoscaler: CostAwareAutoscaler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global autoscaler
    autoscaler = CostAwareAutoscaler(
        configs=E2E_HARDWARE_CONFIGS,
        cooldown_seconds=E2E_COOLDOWN_SECONDS,
        models_dir=MODELS_DIR,
    )
    autoscaler.throughput_tracker = ThroughputTracker()

    if SINGLE_MODEL:
        model_path = autoscaler.get_model_path(SINGLE_MODEL)
        if model_path:
            cheapest = min(E2E_HARDWARE_CONFIGS, key=lambda c: c.hourly_cost)
            port = autoscaler._get_port()
            container = Container(SINGLE_MODEL, model_path, cheapest, port)
            if await container.start():
                autoscaler.containers[SINGLE_MODEL] = container
                autoscaler.current_config[SINGLE_MODEL] = cheapest
                autoscaler.last_scale_time[SINGLE_MODEL] = autoscaler.clock()
                logger.info(f"Started {SINGLE_MODEL} on {cheapest.config_id()}")
            else:
                logger.error(f"Failed to start container for {SINGLE_MODEL}")
        else:
            logger.error(f"Model not found: {SINGLE_MODEL}")
    else:
        await autoscaler.initialize()

    yield
    await autoscaler.cleanup()


app = FastAPI(title="Cost-Aware GPU E2E Test Server", lifespan=lifespan)


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
                autoscaler.throughput_tracker.record_streaming_tokens(request.model, total_tokens)

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=usage,
            )


# ---------------------------------------------------------------------------
# Test-only endpoints: inject synthetic demand / reset cooldown
# ---------------------------------------------------------------------------

class _InjectDemandRequest(_BM):
    model: str
    tokens: int


@app.post("/test/inject_demand")
async def inject_demand(req: _InjectDemandRequest):
    """Inject synthetic token demand for testing."""
    autoscaler.throughput_tracker.record_streaming_tokens(req.model, req.tokens)
    ema = autoscaler.throughput_tracker.get_ema(req.model)
    return {"model": req.model, "injected_tokens": req.tokens, "current_demand_tps": ema}


@app.post("/test/reset_cooldown")
async def reset_cooldown():
    """Reset all cooldown timers."""
    autoscaler.last_scale_time.clear()
    return {"status": "cooldowns_reset"}
