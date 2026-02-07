"""
Test entrypoint for cost-aware autoscaler E2E tests.

Overrides HARDWARE_CONFIGS to CPU-only (2 configs) and uses short
cooldown / demand window so that scaling is observable in seconds.

Usage:
    uv run uvicorn tests.e2e_server:app --host 0.0.0.0 --port <PORT>

Environment variables (all optional):
    E2E_MODELS_DIR   – path to models directory (default: ./models)
    E2E_MODEL_NAME   – single model file stem to load (default: load all)
"""
from __future__ import annotations

import asyncio
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

# Re-export everything from main_cost_aware so the FastAPI app works
from main_cost_aware import (  # noqa: F401
    HardwareConfig,
    CostAwareAutoscaler,
    DemandTracker,
    Container,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    Message,
    get_throughput,
    get_cost_per_token,
    DEFAULT_THROUGHPUT,
    MAX_DRAIN_TIMEOUT_SECONDS,
)
import main_cost_aware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Override configs for E2E testing: CPU-only, 2 tiers
# ---------------------------------------------------------------------------
E2E_HARDWARE_CONFIGS: List[HardwareConfig] = [
    HardwareConfig(cpu_cores=1, memory="4g", hourly_cost=0.10),   # cpu_1
    HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40),   # cpu_4
]

E2E_COOLDOWN_SECONDS = 10        # short cooldown for fast tests
E2E_DEMAND_WINDOW_SECONDS = 30   # short window so demand decays quickly

MODELS_DIR = os.environ.get("E2E_MODELS_DIR", "./models")
SINGLE_MODEL = os.environ.get("E2E_MODEL_NAME", "")

# ---------------------------------------------------------------------------
# Build a new FastAPI app with overridden autoscaler
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

autoscaler: CostAwareAutoscaler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global autoscaler
    autoscaler = CostAwareAutoscaler(
        configs=E2E_HARDWARE_CONFIGS,
        cooldown_seconds=E2E_COOLDOWN_SECONDS,
        models_dir=MODELS_DIR,
    )
    # Override the demand tracker window
    autoscaler.demand_tracker = DemandTracker(
        window_seconds=E2E_DEMAND_WINDOW_SECONDS,
    )

    if SINGLE_MODEL:
        # Only load the specified model
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


app = FastAPI(title="Cost-Aware E2E Test Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints (delegate to the shared autoscaler instance)
# ---------------------------------------------------------------------------

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
    """Forward a non-streaming completion to the container."""
    import aiohttp
    import uuid
    import time

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
                autoscaler.demand_tracker.record_tokens(
                    request.model, total_tokens
                )

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=usage,
            )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
