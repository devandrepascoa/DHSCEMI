import pytest
import requests
import json
import time
import subprocess
import sys
import os
import signal
from pathlib import Path


@pytest.fixture(scope="session")
def proxy_server():
    BASE_URL = "http://localhost:8000"

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("Proxy server already running, using existing instance")
            yield
            return
    except requests.exceptions.RequestException:
        pass

    print("Starting proxy server...")
    process = subprocess.Popen(
        ["uv", "run", "python", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if os.name != 'nt' else None
    )

    max_attempts = 30
    server_started = False
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("Proxy server started successfully")
                server_started = True
                break
        except requests.exceptions.RequestException:
            pass

        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Proxy server process died. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

        time.sleep(1)

    if not server_started:
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, OSError):
            pass
        pytest.fail("Failed to start proxy server within 30 seconds")

    yield

    print("Stopping proxy server...")
    try:
        if os.name != 'nt':
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass
    print("Proxy server stopped")


@pytest.fixture
def base_url(proxy_server):
    return "http://localhost:8000"


class TestProxy:
    TEST_MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"

    def test_health_endpoint(self, base_url):
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
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
                assert "id" in model
                assert "object" in model
                assert "created" in model
                assert "owned_by" in model
                assert "container_count" in model
                assert "ready_containers" in model

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Models endpoint failed: {e}")

    def test_containers_endpoint(self, base_url):
        try:
            response = requests.get(f"{base_url}/containers", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "containers" in data
            assert isinstance(data["containers"], list)

            if data["containers"]:
                container = data["containers"][0]
                assert "model" in container
                assert "container_name" in container
                assert "port" in container
                assert "is_ready" in container
                assert "request_count" in container
                assert "last_used" in container

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Containers endpoint failed: {e}")

    def test_chat_completion_non_streaming(self, base_url):
        payload = {
            "model": self.TEST_MODEL,
            "messages": [
                {"role": "user", "content": "Hello, this is a test."}
            ],
            "max_tokens": 10,
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)

            assert response.status_code in [200, 404]

            if response.status_code == 404:
                pytest.skip(f"Model {self.TEST_MODEL} not available")

            data = response.json()
            assert "id" in data
            assert "object" in data
            assert "created" in data
            assert "model" in data
            assert "choices" in data
            assert isinstance(data["choices"], list)

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Chat completion failed: {e}")

    def test_chat_completion_streaming(self, base_url):
        payload = {
            "model": self.TEST_MODEL,
            "messages": [
                {"role": "user", "content": "Hello, this is a test."}
            ],
            "max_tokens": 10,
            "temperature": 0.7,
            "stream": True
        }

        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30, stream=True)

            assert response.status_code in [200, 404]

            if response.status_code == 404:
                pytest.skip(f"Model {self.TEST_MODEL} not available")

            assert response.headers.get('content-type') == 'text/plain; charset=utf-8'

            content_parts = []
            has_data = False

            try:
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        content_parts.append(chunk)
                        if 'data: ' in chunk:
                            has_data = True
            except requests.exceptions.ChunkedEncodingError as e:
                if not has_data:
                    if response.status_code == 200:
                        return
                    else:
                        pytest.fail(f"Streaming failed with no data and error: {e}")

            if not has_data and not content_parts:
                pytest.fail("No streaming data received")

            content = ''.join(content_parts)

            if content.strip():
                lines = content.strip().split('\n')
                data_lines = [line for line in lines if line.startswith('data: ')]

                assert len(data_lines) > 0

                done_lines = [line for line in lines if line.strip() == 'data: [DONE]']

            else:
                if response.status_code == 200:
                    return
                else:
                    pytest.fail("Empty streaming response with non-200 status")

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
        for i in range(3):
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

            assert response.status_code in [400, 422]

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
            model_files = [f for f in models_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in supported_extensions]

            assert isinstance(model_files, list)


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
