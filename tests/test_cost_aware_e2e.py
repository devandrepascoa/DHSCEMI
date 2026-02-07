"""
E2E tests for cost-aware autoscaler with real Docker containers.

Requires:
  - Docker running
  - models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf present

Run with:
    uv run pytest tests/test_cost_aware_e2e.py -v -s --timeout=600
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MODEL_FILE = Path("models") / f"{TEST_MODEL}.gguf"

# Timeouts
SERVER_STARTUP_TIMEOUT = 180   # seconds to wait for server + first container
REQUEST_TIMEOUT = 120          # seconds per inference request
SCALE_WAIT_TIMEOUT = 180       # seconds to wait for a scaling event


def _free_port() -> int:
    """Return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _cleanup_docker_containers() -> None:
    """Remove all Docker containers whose name starts with 'llama-'."""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=llama-", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
    for name in names:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


# ---------------------------------------------------------------------------
# 8.1  Session-scoped fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def e2e_server():
    """Start the cost-aware proxy on a random port with CPU-only configs.

    Yields the base URL (e.g. ``http://localhost:12345``).
    On teardown the server process is killed and all ``llama-*`` Docker
    containers are removed.
    """
    if not MODEL_FILE.exists():
        pytest.skip(f"Model file not found: {MODEL_FILE}")

    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["E2E_MODEL_NAME"] = TEST_MODEL
    env["E2E_MODELS_DIR"] = str(Path("models").resolve())

    # Start the test server as a subprocess
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "tests.e2e_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for the server to become healthy (container startup is slow)
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
        # Check if process died
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            pytest.fail(
                f"Server process exited with code {proc.returncode} "
                f"before becoming healthy.\nOutput:\n{stdout}"
            )
        time.sleep(3)

    if not healthy:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        proc.kill()
        proc.wait()
        _cleanup_docker_containers()
        pytest.fail(
            f"Server did not become healthy within {SERVER_STARTUP_TIMEOUT}s.\n"
            f"Output:\n{stdout}"
        )

    yield base_url

    # ---- Teardown ----
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    _cleanup_docker_containers()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _chat_request(
    base_url: str,
    *,
    max_tokens: int = 32,
    timeout: int = REQUEST_TIMEOUT,
) -> requests.Response:
    """Send a single chat completion request and return the response."""
    payload = {
        "model": TEST_MODEL,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )


def _get_status(base_url: str) -> dict:
    """Fetch /status and return the JSON body."""
    resp = requests.get(f"{base_url}/status", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _current_config_id(base_url: str) -> Optional[str]:
    """Return the config_id the test model is currently running on."""
    status = _get_status(base_url)
    model_info = status.get("models", {}).get(TEST_MODEL, {})
    return model_info.get("config_id")


# ---------------------------------------------------------------------------
# 8.2  Server starts healthy on cheapest config
# ---------------------------------------------------------------------------

class TestCostAwareE2E:
    """E2E tests for cost-aware autoscaler with real Docker containers."""

    def test_server_healthy_cheapest_config(self, e2e_server: str) -> None:
        """8.2 – Server starts healthy and loads model into cpu_1."""
        resp = requests.get(f"{e2e_server}/health", timeout=30)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["ready_containers"] >= 1
        assert TEST_MODEL in data["models"]

        # Verify cheapest config (cpu_1)
        config_id = _current_config_id(e2e_server)
        assert config_id == "cpu_1", f"Expected cpu_1, got {config_id}"

    # -------------------------------------------------------------------
    # 8.3  Chat completion returns valid response
    # -------------------------------------------------------------------

    def test_chat_completion_valid_response(self, e2e_server: str) -> None:
        """8.3 – Sending a chat completion returns a valid response."""
        resp = _chat_request(e2e_server, max_tokens=32)
        assert resp.status_code == 200

        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

        choice = data["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        assert len(choice["message"]["content"]) > 0

        assert "usage" in data
        assert data["usage"].get("total_tokens", 0) > 0
        assert data["model"] == TEST_MODEL

    # -------------------------------------------------------------------
    # 8.4  /status endpoint
    # -------------------------------------------------------------------

    def test_status_endpoint(self, e2e_server: str) -> None:
        """8.4 – /status returns config, demand, and cost metrics."""
        status = _get_status(e2e_server)

        # Top-level keys
        assert "models" in status
        assert "cooldown_seconds" in status
        assert "demand_window_seconds" in status
        assert "available_configs" in status

        # Model-specific info
        model_info = status["models"].get(TEST_MODEL)
        assert model_info is not None, f"Model {TEST_MODEL} not in status"

        assert "config_id" in model_info
        assert "demand_tps" in model_info
        assert "cost_per_token" in model_info
        assert "throughput_tps" in model_info
        assert "hourly_cost" in model_info
        assert "is_ready" in model_info
        assert model_info["is_ready"] is True

        # Verify cost metrics are sensible
        assert model_info["hourly_cost"] > 0
        assert model_info["throughput_tps"] > 0
        assert model_info["cost_per_token"] > 0

    # -------------------------------------------------------------------
    # 8.5  Autoscaler scales up under load
    # -------------------------------------------------------------------

    def test_autoscaler_scales_up(self, e2e_server: str) -> None:
        """8.5 – Sustained load exceeding cpu_1 capacity triggers scale-up.

        cpu_1 throughput is 4 tok/s.  We need demand > 4 tok/s, i.e.
        total_tokens / window(30s) > 4  →  total_tokens > 120.

        We send sequential requests (cpu_1 has --parallel 1) with
        moderate max_tokens to accumulate enough demand, then wait for
        cooldown and trigger a scaling check.
        """
        # Confirm starting on cpu_1
        assert _current_config_id(e2e_server) == "cpu_1"

        # Send sequential requests to build up token demand.
        # Each request with max_tokens=64 generates ~80-100 total tokens.
        # We need >120 tokens in the 30s window, so 3-4 requests should suffice.
        for i in range(4):
            resp = _chat_request(e2e_server, max_tokens=64, timeout=180)
            assert resp.status_code == 200, (
                f"Request {i} failed: {resp.status_code} {resp.text}"
            )

        # Verify demand was recorded
        status = _get_status(e2e_server)
        model_info = status["models"].get(TEST_MODEL, {})
        demand = model_info.get("demand_tps", 0)

        # Wait for cooldown (10s) to elapse so the next request triggers scaling
        time.sleep(12)

        # Send another request – this triggers check_scaling()
        resp = _chat_request(e2e_server, max_tokens=16, timeout=180)
        assert resp.status_code == 200

        # The scaling happens during get_container().
        # Poll /status until we see cpu_4 or timeout.
        # During scaling, a new container starts (30-60s), so be patient.
        deadline = time.time() + SCALE_WAIT_TIMEOUT
        scaled = False
        while time.time() < deadline:
            cid = _current_config_id(e2e_server)
            if cid == "cpu_4":
                scaled = True
                break
            # Send a small request to trigger another scaling check
            try:
                _chat_request(e2e_server, max_tokens=16, timeout=180)
            except Exception:
                pass
            time.sleep(8)

        assert scaled, (
            f"Expected scale-up to cpu_4 but config is still "
            f"{_current_config_id(e2e_server)}. "
            f"Last demand_tps={demand}"
        )

    # -------------------------------------------------------------------
    # 8.6  Autoscaler scales back down
    # -------------------------------------------------------------------

    def test_autoscaler_scales_down(self, e2e_server: str) -> None:
        """8.6 – After load stops and cooldown elapses, scales back to cpu_1.

        We wait for the demand window (30s) to expire so all token events
        fall out of the sliding window, then wait for cooldown (10s),
        and send a request to trigger the scaling check.
        """
        # Should currently be on cpu_4 from the previous test
        current = _current_config_id(e2e_server)
        if current != "cpu_4":
            pytest.skip(
                f"Scale-up test did not leave server on cpu_4 (got {current}); "
                "cannot test scale-down"
            )

        # Wait for demand window + cooldown to expire
        # demand_window=30s + cooldown=10s + buffer
        time.sleep(45)

        # Send a lightweight request to trigger scaling check
        resp = _chat_request(e2e_server, max_tokens=16, timeout=180)
        assert resp.status_code == 200

        # Poll /status until we see cpu_1 or timeout
        deadline = time.time() + SCALE_WAIT_TIMEOUT
        scaled_down = False
        while time.time() < deadline:
            cid = _current_config_id(e2e_server)
            if cid == "cpu_1":
                scaled_down = True
                break
            # Send another small request to trigger scaling check
            try:
                _chat_request(e2e_server, max_tokens=16, timeout=180)
            except Exception:
                pass
            time.sleep(8)

        assert scaled_down, (
            f"Expected scale-down to cpu_1 but config is still "
            f"{_current_config_id(e2e_server)}"
        )
