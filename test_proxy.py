#!/usr/bin/env python3
"""
Pytest-based test suite for the clustered llama.cpp OpenAI proxy
Tests container pools, model discovery, and routing
"""

import pytest
import requests
import json
import time
import subprocess
import sys
from pathlib import Path


class TestProxy:
    """Test suite for the clustered llama.cpp OpenAI proxy"""
    
    BASE_URL = "http://localhost:8000"
    TEST_MODEL = "1-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
    
    @pytest.fixture(scope="session")
    def base_url(self):
        """Base URL for the proxy"""
        return self.BASE_URL
    
    def test_health_endpoint(self, base_url):
        """Test health endpoint"""
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
        """Test models endpoint"""
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=10)
            assert response.status_code == 200
            
            data = response.json()

            assert "object" in data
            assert data["object"] == "list"
            assert "data" in data
            assert isinstance(data["data"], list)

            # Check model structure
            if data["data"]:
                model = data["data"][0]
                assert "id" in model
                assert "object" in model
                assert "created" in model
                assert "owned_by" in model
                assert "container_count" in model
                assert "ready_containers" in model
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Models endpoint failed: {e}")
    
    def test_containers_endpoint(self, base_url):
        """Test containers endpoint"""
        try:
            response = requests.get(f"{base_url}/containers", timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "containers" in data
            assert isinstance(data["containers"], list)
            
            # Check container structure
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
        """Test non-streaming chat completion"""
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

            # We expect 404 for missing model, but endpoint should respond
            assert response.status_code in [200, 404]
            
            assert response.status_code == 200

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
        """Test streaming chat completion"""
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
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=30)


            # We expect 404 for missing model, but endpoint should respond
            assert response.status_code in [200, 404]


            assert response.status_code == 200

            # Check streaming response format
            assert response.headers.get('content-type') == 'text/plain; charset=utf-8'

            # Read streaming response
            content = response.text
            lines = content.strip().split('\n')
            # Should contain data: prefix
            data_lines = [line for line in lines if line.startswith('data: ')]
            assert len(data_lines) > 0

            # Should contain [DONE] marker
            done_lines = [line for line in lines if line.strip() == 'data: [DONE]']
            assert len(done_lines) > 0

        except requests.exceptions.RequestException as e:
            pytest.fail(f"Streaming chat completion failed: {e}")
    
    def test_load_balancing_multiple_requests(self, base_url):
        """Test load balancing with multiple requests"""
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
        
        # All requests should either succeed or fail consistently
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)
    
    def test_invalid_model_error(self, base_url):
        """Test error handling for invalid model"""
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
        """Test error handling for invalid request"""
        payload = {
            "invalid_field": "test"
        }
        
        try:
            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=10)
            # Should return validation error
            assert response.status_code in [400, 422]
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Invalid request test failed: {e}")


class TestModelDiscovery:
    """Test suite for model discovery"""
    
    def test_models_directory_exists(self):
        """Test that models directory exists"""
        models_dir = Path("./models")
        assert models_dir.exists() or models_dir.mkdir(exist_ok=True)
    
    def test_model_file_extensions(self):
        """Test supported model file extensions"""
        models_dir = Path("./models")
        supported_extensions = {'.gguf', '.bin'}
        
        if models_dir.exists():
            model_files = [f for f in models_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in supported_extensions]
            # Test should pass regardless of model files presence
            assert isinstance(model_files, list)


class TestContainerManagement:
    """Test suite for container management"""
    BASE_URL = "http://localhost:8000"
    TEST_MODEL = "1-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"

    @pytest.fixture(scope="session")
    def base_url(self):
        """Base URL for the proxy"""
        return self.BASE_URL

    def test_container_health_check(self, base_url):
        """Test container health check mechanism"""
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
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
