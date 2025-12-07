import asyncio
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import main
from main import AVAILABLE_CONFIGS, ChatCompletionChoice, ChatCompletionResponse, ContainerConfig

MODEL_NAME = "autoscale-model"


class StubContainer:
    def __init__(self, model_name: str, config: ContainerConfig):
        self.model_name = model_name
        self.config = config
        self.container_name = f"stub-{model_name}-{str(config)}"
        self._is_ready = True
        self.request_count = 0
        self.active_requests = 0
        self.queue_start_times = {}
        self.last_scale_evaluation = 0.0
        self.metrics_lock = asyncio.Lock()

    def get_endpoint(self) -> str:
        return "http://stub"

    async def estimate_processing_time(self, estimated_tokens: int = 100) -> float:
        return 0.01

    async def get_load_score(self, estimated_tokens: int = 100) -> float:
        return await self.estimate_processing_time(estimated_tokens)

    async def record_processing_time(self, duration: float, tokens: int = 0) -> None:
        return

    async def stop(self) -> None:
        self._is_ready = False


@pytest.fixture
def patched_proxy(monkeypatch):
    async def fake_spawn(self, model_name, model_path, config):
        container = StubContainer(model_name, config)
        self.container_pools.setdefault(model_name, []).append(container)
        return container

    async def fake_init():
        main.container_manager.container_pools.clear()
        main.container_manager.container_pools[MODEL_NAME] = []
        main.container_manager.workload_metrics.current_container_config[MODEL_NAME] = AVAILABLE_CONFIGS[0]
        main.container_manager.workload_metrics.current_hardware_type[MODEL_NAME] = AVAILABLE_CONFIGS[0].container_type
        await main.container_manager.spawn_container(MODEL_NAME, Path("."), AVAILABLE_CONFIGS[0])

    async def fake_non_streaming(request, container):
        total_tokens = request.max_tokens or 100
        choice = ChatCompletionChoice(
            index=0,
            message={"role": "assistant", "content": "stub"},
            finish_reason="stop",
        )
        response = ChatCompletionResponse(
            id="test-response",
            created=int(time.time()),
            model=request.model,
            choices=[choice],
            usage={
                "prompt_tokens": total_tokens,
                "completion_tokens": 0,
                "total_tokens": total_tokens,
            },
        )
        return response

    async def fake_streaming(request, container, request_id, start_time):
        yield "data: stub\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(main.ContainerManager, "spawn_container", fake_spawn, raising=False)
    monkeypatch.setattr(main.ContainerManager, "get_model_path", lambda self, name: Path('.') if name == MODEL_NAME else None, raising=False)
    monkeypatch.setattr(main, "initialize_all_model_clusters", fake_init, raising=False)
    monkeypatch.setattr(main, "non_streaming_chat_completion", fake_non_streaming, raising=False)
    monkeypatch.setattr(main, "stream_chat_completion_with_error_handling", fake_streaming, raising=False)

    async def fake_cleanup_single_container(self, container):
        return None
    monkeypatch.setattr(main.ContainerManager, "_cleanup_single_container", fake_cleanup_single_container, raising=False)

    with TestClient(main.app) as client:
        cm = main.container_manager
        cm.workload_metrics.scaling_cooldown = 0
        yield client, cm


def current_config_string(cm) -> str:
    return str(cm.workload_metrics.get_current_container_config(MODEL_NAME))


def test_proxy_autoscaling_e2e(patched_proxy):
    client, cm = patched_proxy

    high_payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "generate"}],
        "max_tokens": 5000,
        "temperature": 0.2,
        "stream": False,
    }

    expected_configs = [str(cfg) for cfg in AVAILABLE_CONFIGS[1:]]
    for expected in expected_configs:
        resp = client.post("/v1/chat/completions", json=high_payload)
        assert resp.status_code == 200
        assert current_config_string(cm) == expected

    cm.workload_metrics.token_events[MODEL_NAME].clear()

    low_payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "short"}],
        "max_tokens": 5,
        "temperature": 0.1,
        "stream": False,
    }

    for expected in reversed([str(cfg) for cfg in AVAILABLE_CONFIGS[:-1]]):
        resp = client.post("/v1/chat/completions", json=low_payload)
        assert resp.status_code == 200
        assert current_config_string(cm) == expected
