import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests


def _terminate_process(process: subprocess.Popen) -> str:
    stderr_output = ""
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
            except (ProcessLookupError, OSError):
                pass
    if process.stderr:
        try:
            stderr_output = process.stderr.read().decode()
        except Exception:
            stderr_output = ""
        finally:
            process.stderr.close()
    return stderr_output


@pytest.fixture(scope="session")
def proxy_server():
    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]

    command = [
        "uv",
        "run",
        "uvicorn",
        "main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    popen_kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.PIPE,
    }
    if os.name != "nt":
        popen_kwargs["preexec_fn"] = os.setsid
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(command, **popen_kwargs)
    base_url = f"http://{host}:{port}"

    for _ in range(60):
        if process.poll() is not None:
            stderr_output = _terminate_process(process)
            pytest.fail(f"Proxy server exited prematurely. stderr:\n{stderr_output}")
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(1)
            continue
        time.sleep(1)
    else:
        stderr_output = _terminate_process(process)
        pytest.fail(f"Failed to start proxy server within timeout. stderr:\n{stderr_output}")

    yield base_url

    _terminate_process(process)


@pytest.fixture
def base_url(proxy_server):
    return proxy_server


class TestProxy:
    TEST_MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"

    def test_health_endpoint(self, base_url):
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert data["status"] in {"healthy", "down"}
            assert "total_containers" in data
            assert "ready_containers" in data
            assert "models" in data

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Health check failed: {e}")

    def test_models_endpoint(self, base_url):
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=10)
            assert response.status_code == 200

            data = response.json()

            assert "models" in data
            assert isinstance(data["models"], list)

            if data["models"]:
                model = data["models"][0]
                if isinstance(model, dict):
                    assert "id" in model
                    assert "object" in model
                else:
                    assert isinstance(model, str)

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Models endpoint failed: {e}")

    def test_containers_endpoint(self, base_url):
        try:
            response = requests.get(f"{base_url}/containers", timeout=10)
            assert response.status_code == 200

            data = response.json()

            assert "containers" in data
            assert isinstance(data["containers"], list)

            for container in data["containers"]:
                assert "container_name" in container
                assert "model" in container
                assert "is_ready" in container

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Containers endpoint failed: {e}")

    def test_metrics_endpoint(self, base_url):
        try:
            response = requests.get(f"{base_url}/v1/metrics", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, dict)

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Metrics endpoint failed: {e}")

    def test_chat_completion_non_streaming(self, base_url):
        payload = {
            "model": self.TEST_MODEL,
            "messages": [
                {"role": "user", "content": "Say hello"}
            ],
            "max_tokens": 16,
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)
            assert response.status_code in (200, 503)

            if response.status_code == 200:
                data = response.json()
                assert "choices" in data
                assert isinstance(data["choices"], list)
                if data["choices"]:
                    choice = data["choices"][0]
                    assert "message" in choice
                    assert "content" in choice["message"]

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Chat completion request failed: {e}")

    def test_chat_completion_streaming(self, base_url):
        payload = {
            "model": self.TEST_MODEL,
            "messages": [
                {"role": "user", "content": "Stream a short response"}
            ],
            "max_tokens": 16,
            "temperature": 0.7,
            "stream": True
        }

        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=30
            )
            assert response.status_code in (200, 503)

            if response.status_code != 200:
                return

            content_parts = []
            has_data = False

            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    content_parts.append(chunk)
                    if 'data: ' in chunk:
                        has_data = True

            if not has_data and not content_parts:
                pytest.fail("No streaming data received")

            content = ''.join(content_parts)
            if content.strip():
                lines = content.strip().split('\n')
                data_lines = [line for line in lines if line.startswith('data: ')]
                assert len(data_lines) > 0

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Streaming chat completion failed: {e}")

    def test_load_balancing_multiple_requests(self, base_url):
        payload = {
            "model": self.TEST_MODEL,
            "messages": [
                {"role": "user", "content": "Test load balancing"}
            ],
            "max_tokens": 5,
            "temperature": 0.7,
            "stream": False
        }

        results = []
        for _ in range(3):
            try:
                response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=10)
                results.append(response.status_code)
            except requests.exceptions.RequestException as e:
                results.append(f"error: {e}")

        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)

    def test_invalid_model_error(self, base_url):
        payload = {
            "model": "nonexistent-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10
        }

        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=10)
            assert response.status_code == 404

            data = response.json()
            assert "detail" in data
            assert "nonexistent-model" in data["detail"]

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Invalid model test failed: {e}")

    def test_invalid_request_error(self, base_url):
        payload = {
            "invalid_field": "test"
        }

        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=10)
            assert response.status_code in (400, 422)

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Invalid request test failed: {e}")


class TestModelDiscovery:

    def test_models_directory_exists(self):
        models_dir = Path("./models")
        assert models_dir.exists() or models_dir.mkdir(exist_ok=True)

    def test_model_file_extensions(self):
        models_dir = Path("./models")
        supported_extensions = {'.gguf', '.bin'}

        if models_dir.exists():
            model_files = [
                f for f in models_dir.iterdir()
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]
            assert isinstance(model_files, list)




class TestAutoscalingBenchmark:
    BENCH_MODEL = "autoscale-model"

    def _get_state(self, base_url):
        resp = requests.get(f"{base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _apply_load(self, base_url, model, tokens_per_hour, reset=False):
        payload = {
            "model": model,
            "tokens_per_hour": tokens_per_hour,
            "pulses": 10,
            "reset": reset,
            "override_cooldown": 0.0,
        }
        resp = requests.post(f"{base_url}/benchmark/apply_load", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def test_dynamic_scaling_benchmark(self, base_url):
        state = self._get_state(base_url)
        configs = state["global"]["available_configs"]
        capacities = state["global"]["config_capacity_tokens_per_hour"]

        # Ensure model is registered and cooldown disabled
        self._apply_load(base_url, self.BENCH_MODEL, 0, reset=True)

        # Scale up through all configurations
        for idx in range(len(configs) - 1):
            current_cfg = configs[idx]
            target_cfg = configs[idx + 1]
            cap = max(capacities[current_cfg], capacities[target_cfg])
            self._apply_load(base_url, self.BENCH_MODEL, cap * 0.9, reset=True)
            new_state = self._get_state(base_url)
            assert new_state["workload"][self.BENCH_MODEL]["current_config"] == target_cfg

        final_state = self._get_state(base_url)
        assert final_state["workload"][self.BENCH_MODEL]["current_config"] == configs[-1]

        # Reset history before scaling down
        self._apply_load(base_url, self.BENCH_MODEL, 0, reset=True)
        state_after_reset = self._get_state(base_url)
        current_cfg = state_after_reset["workload"][self.BENCH_MODEL]["current_config"]
        current_index = configs.index(current_cfg)

        for idx in reversed(range(1, current_index + 1)):
            current_cfg = configs[idx]
            target_cfg = configs[idx - 1]
            cap = min(capacities[current_cfg], capacities[target_cfg])
            self._apply_load(base_url, self.BENCH_MODEL, cap * 0.1, reset=True)
            new_state = self._get_state(base_url)
            assert new_state["workload"][self.BENCH_MODEL]["current_config"] == target_cfg

class TestContainerManagement:

    def test_container_health_check(self, base_url):
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "ready_containers" in data
            assert isinstance(data["ready_containers"], int)
            assert data["ready_containers"] >= 0

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Health check failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
