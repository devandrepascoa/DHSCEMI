"""
E2E tests for cost-aware autoscaler with GPU Docker containers.

Uses /test/inject_demand to synthetically set demand, avoiding slow
real-inference token accumulation. Only container startup + one real
inference call per scaling tier.

Configs: cpu_4, cpu_8, cpu_14, gpu_50
Cheapest is cpu_4 ($0.40/hr). Scale-up to gpu_50 requires demand > 24 tok/s
(cpu_14 throughput), so we inject enough tokens to exceed that.

Requires:
  - Docker running
  - NVIDIA GPU available with nvidia-container-toolkit installed
  - models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf present

Run with:
    uv run pytest tests/test_cost_aware_e2e_gpu.py -v -s --timeout=600
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest
import requests

TEST_MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MODEL_FILE = Path("models") / f"{TEST_MODEL}.gguf"

SERVER_STARTUP_TIMEOUT = 180
REQUEST_TIMEOUT = 180
SCALE_WAIT_TIMEOUT = 120


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _cleanup_docker_containers() -> None:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    for name in [n.strip() for n in result.stdout.splitlines() if n.strip()]:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _gpu_available_in_docker() -> bool:
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "--privileged",
             "nvidia/cuda:12.6.3-base-ubuntu24.04", "nvidia-smi"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "NVIDIA" in result.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 9.1  Session-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_e2e_server():
    """Start the GPU e2e test server on a random port.

    Yields the base URL. On teardown kills the server and removes
    all llama-* Docker containers.
    """
    if not MODEL_FILE.exists():
        pytest.skip(f"Model file not found: {MODEL_FILE}")

    if not _gpu_available_in_docker():
        pytest.skip("NVIDIA GPU not available in Docker")

    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = TEST_MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "tests.e2e_gpu_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    healthy = False
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            data = resp.json()
            if data.get("status") == "healthy" and data.get("ready_containers", 0) > 0:
                healthy = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(
                f"Server exited with code {proc.returncode} before becoming healthy.\n"
                f"Output:\n{stdout}"
            )
        time.sleep(3)

    if not healthy:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        proc.kill()
        proc.wait()
        _cleanup_docker_containers()
        pytest.fail(f"Server not healthy within {SERVER_STARTUP_TIMEOUT}s.\nOutput:\n{stdout}")

    yield base_url

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    _cleanup_docker_containers()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_request(base_url: str, *, max_tokens: int = 32, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    return requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)


def _get_status(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/status", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _current_config_id(base_url: str) -> Optional[str]:
    status = _get_status(base_url)
    model_info = status.get("models", {}).get(TEST_MODEL, {})
    return model_info.get("config_id")


def _inject_demand(base_url: str, tokens: int) -> dict:
    """Inject synthetic token demand via test endpoint."""
    resp = requests.post(
        f"{base_url}/test/inject_demand",
        json={"model": TEST_MODEL, "tokens": tokens},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _reset_cooldown(base_url: str) -> None:
    """Reset all scaling cooldown timers."""
    resp = requests.post(f"{base_url}/test/reset_cooldown", timeout=30)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCostAwareGPUE2E:
    """E2E tests for cost-aware autoscaler with GPU containers."""

    # 9.2 Server starts healthy on cheapest config (cpu_4)
    def test_server_healthy_cheapest_config(self, gpu_e2e_server: str) -> None:
        resp = requests.get(f"{gpu_e2e_server}/health", timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["ready_containers"] >= 1
        assert TEST_MODEL in data["models"]

        config_id = _current_config_id(gpu_e2e_server)
        assert config_id == "cpu_4", f"Expected cpu_4, got {config_id}"

    # 9.3 Chat completion works on initial CPU config
    def test_chat_completion_works(self, gpu_e2e_server: str) -> None:
        resp = _chat_request(gpu_e2e_server, max_tokens=32)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) > 0
        assert len(data["choices"][0]["message"]["content"]) > 0

    # 9.4 Scale from CPU to GPU after high synthetic demand
    def test_scale_cpu_to_gpu(self, gpu_e2e_server: str) -> None:
        """Inject demand > cpu_14 throughput (24 tok/s) to force GPU selection.

        With demand_window=30s, we need > 24*30 = 720 tokens in the window.
        Inject 1500 tokens to be safe.
        """
        assert _current_config_id(gpu_e2e_server) == "cpu_4"

        # Reset cooldown so scaling can happen immediately
        _reset_cooldown(gpu_e2e_server)

        # Inject high demand that exceeds all CPU configs
        _inject_demand(gpu_e2e_server, 1500)

        # Send a request to trigger scaling check
        resp = _chat_request(gpu_e2e_server, max_tokens=16)
        assert resp.status_code == 200

        # Poll until GPU container is ready
        deadline = time.time() + SCALE_WAIT_TIMEOUT
        scaled = False
        while time.time() < deadline:
            cid = _current_config_id(gpu_e2e_server)
            if cid == "gpu_50":
                scaled = True
                break
            time.sleep(5)

        assert scaled, (
            f"Expected scale to gpu_50 but config is {_current_config_id(gpu_e2e_server)}"
        )

    # 9.5 Scale back from GPU to CPU after demand drops
    def test_scale_gpu_to_cpu(self, gpu_e2e_server: str) -> None:
        """Wait for demand window to expire, reset cooldown, trigger scaling."""
        current = _current_config_id(gpu_e2e_server)
        if current != "gpu_50":
            pytest.skip(f"Not on gpu_50 (got {current}), cannot test scale-down")

        # Wait for demand window (30s) to expire + small buffer
        time.sleep(35)

        # Reset cooldown
        _reset_cooldown(gpu_e2e_server)

        # Send a request to trigger scaling check (demand is now ~0)
        resp = _chat_request(gpu_e2e_server, max_tokens=16)
        assert resp.status_code == 200

        # Poll until back on cpu_4
        deadline = time.time() + SCALE_WAIT_TIMEOUT
        scaled_down = False
        while time.time() < deadline:
            cid = _current_config_id(gpu_e2e_server)
            if cid == "cpu_4":
                scaled_down = True
                break
            time.sleep(5)

        assert scaled_down, (
            f"Expected scale-down to cpu_4 but config is {_current_config_id(gpu_e2e_server)}"
        )

    # 9.6 /status shows GPU details when on GPU
    def test_status_shows_gpu_details(self, gpu_e2e_server: str) -> None:
        """Re-scale to GPU and verify /status shows GPU fields."""
        _reset_cooldown(gpu_e2e_server)
        _inject_demand(gpu_e2e_server, 1500)

        # Trigger scaling
        _chat_request(gpu_e2e_server, max_tokens=16)

        # Wait for GPU
        deadline = time.time() + SCALE_WAIT_TIMEOUT
        on_gpu = False
        while time.time() < deadline:
            if _current_config_id(gpu_e2e_server) == "gpu_50":
                on_gpu = True
                break
            time.sleep(5)

        if not on_gpu:
            pytest.skip("Could not scale to GPU for status check")

        status = _get_status(gpu_e2e_server)
        model_info = status["models"][TEST_MODEL]

        assert model_info["config_id"] == "gpu_50"
        assert model_info["container_type"] == "gpu"
        assert model_info["gpu_percentage"] == 50
        assert model_info["hourly_cost"] == 5.00
        assert model_info["throughput_tps"] == 50.0
        assert model_info["is_ready"] is True
