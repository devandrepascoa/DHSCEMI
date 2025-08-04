# Dynamic Hardware Selection for Cost Effective ML Inference

This proxy dynamically selects hardware resources (CPU/GPU/accelerators) for incoming workloads to maximize cost efficiency. 

This proxy uses llama-server as the backend inference engine, and orchestrates the servers with docker containers.

## Prerequisites
- Docker installed and running
- UV package manager

## Installation

Install UV if you haven't already:
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install project dependencies:
```bash
# Install dependencies and create virtual environment
uv sync
```

## Model Setup

1. Create a `models` directory in the project root:
```bash
mkdir -p models
```

2. Place your llama.cpp compatible model files (`.gguf` format) in the `models` directory:
```bash
models/
├── model1.gguf
├── model2.gguf
└── ...
```

## Running the Proxy Server

### Start the Server

Run the main proxy server:
```bash
uv run main.py
```

An OpenAI-compatible proxy server will start on `http://localhost:8000` by default.

## Running Tests

The test suite validates proxy functionality, model discovery, and container management.

### Prerequisites for Testing

1. Ensure the proxy server is running on `http://localhost:8000`
2. Have at least one model available (tests expect `01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M` by default)

### Run All Tests
```bash
uv run pytest
```

### Run Specific Test Classes
```bash
uv run pytest test_proxy.py::TestProxy -v

uv run pytest test_proxy.py::TestModelDiscovery -v

uv run pytest test_proxy.py::TestContainerManagement -v
```
