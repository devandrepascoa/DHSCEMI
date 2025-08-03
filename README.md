# llama.cpp OpenAI Proxy

An OpenAI-compatible proxy server for llama.cpp Docker containers with intelligent container management and workload optimization (work in progress).

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

The server will start on `http://localhost:8000` by default.

## Running Tests

The test suite validates proxy functionality, model discovery, and container management.

### Prerequisites for Testing

1. Ensure the proxy server is running on `http://localhost:8000`
2. Have at least one model available (tests expect `1-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M` by default)

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

### Project Structure
```
thesis_proxy/
├── main.py              # Main proxy server
├── test_proxy.py         # Test suite
├── pyproject.toml        # Project configuration
├── uv.lock              # UV lock file
├── models/               # Model files directory
└── benchmarks/           # Benchmarking tools
```
