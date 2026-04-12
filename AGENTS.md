# AGENTS.md - AI Coding Agent Guidelines

## Project Overview

This is a cost-aware autoscaling proxy for ML inference. It provides an OpenAI-compatible API that routes requests to llama.cpp Docker containers, dynamically selecting hardware configurations (CPU with varying core counts, GPU) based on per-request throughput and cost efficiency.

The main server is `main_cost_aware.py`. Hardware configurations and measured throughput are defined in `hardware_configs.json`. The project also includes benchmarking tools, scaling simulations, and thesis figure generation.

## Build/Lint/Test Commands

### Package Manager
This project uses **UV** as the package manager.

```bash
# Install dependencies
uv sync

# Run the proxy server
uv run uvicorn main_cost_aware:app --port 8000
```

### Testing

```bash
# Run all tests
uv run pytest

# Run unit tests (no server required)
uv run pytest tests/test_cost_aware.py -v

# Run e2e tests (requires proxy server running)
uv run pytest tests/test_cost_aware_e2e.py -v

# Run GPU e2e tests
uv run pytest tests/test_cost_aware_e2e_gpu.py -v

# Run tests with output
uv run pytest -v -s

# Run tests matching a pattern
uv run pytest -k "scaling" -v
```

### Prerequisites for Testing
1. At least one model file in `./models/` directory (`.gguf` format)
2. Docker installed and running
3. For e2e tests: proxy server must be running

## Code Style Guidelines

### Imports
Order imports as follows:
1. `from __future__ import annotations` (if used)
2. Standard library imports
3. Third-party imports
4. Local imports

```python
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
```

### Type Hints
- Use type hints for all function parameters and return types
- Use `Optional[T]` for nullable types
- Import from `typing` module: `Dict`, `List`, `Optional`, `Any`, `Tuple`, `AsyncGenerator`

### Naming Conventions
- Classes: `PascalCase` (e.g., `CostAwareAutoscaler`, `HardwareConfig`)
- Functions/methods: `snake_case` (e.g., `select_config_per_request`, `get_cost_per_token`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MIN_TPS_THRESHOLD`, `COOLDOWN_SECONDS`)
- Private methods: prefix with `_` (e.g., `_update_per_request_ema`, `_poll_metrics`)
- Instance variables: `snake_case`

### Data Classes and Models
- Use `@dataclass` for simple data containers (e.g., `HardwareConfig`)
- Use Pydantic `BaseModel` for API request/response models with validation
- Use `Field()` for default values and validation constraints

### Async Patterns
- Use `async/await` for I/O operations
- Use `asyncio.Lock()` for thread-safe access to shared state
- Use `aiohttp` for async HTTP requests to llama.cpp containers

### Error Handling
- Use try/except blocks for external operations (Docker, HTTP)
- Log errors with appropriate severity levels
- Rollback to previous container on failed scaling

### Logging
- Use Python logging: `logger = logging.getLogger(__name__)`
- Structured JSON logging via `_log_json()` for container lifecycle events

### FastAPI Patterns
- Use lifespan context manager for startup/shutdown
- Return Pydantic models or dicts from endpoints

### Docker Integration
- Use `asyncio.create_subprocess_exec` for async Docker commands in the demo server
- Use `subprocess.run` for sync Docker commands in the `Container` class
- Always clean up containers on shutdown
- Use health checks to verify container readiness

## Project Structure

```
├── main_cost_aware.py           # Main proxy server with autoscaling logic
├── hardware_configs.json        # Hardware tiers and measured throughput
├── throughput_benchmark.py      # Throughput benchmarking tool
├── throughput_benchmark_results*.json # Benchmark result snapshots
├── tests/
│   ├── test_cost_aware.py       # Unit tests for scaling logic
│   ├── test_cost_aware_e2e.py   # End-to-end CPU scaling tests
│   ├── test_cost_aware_e2e_gpu.py # End-to-end GPU scaling tests
│   ├── e2e_server.py            # E2E test server helper
│   └── e2e_gpu_server.py        # E2E GPU test server helper
├── benchmarks/
│   ├── benchmark_results/       # Stored benchmark output data
│   ├── scaling_demo_logs/       # Logs from scaling demo runs
│   ├── thesis_figures/          # Generated thesis figures
│   ├── scaling_demo.py          # Live scaling demonstration
│   ├── scaling_simulation.py    # Scaling behavior simulation
│   ├── scaling_simulation_plots.py # Plot generation for simulation
│   ├── thesis_plots.py          # Thesis figure generation
│   ├── live_plot.py             # Real-time plotting
│   ├── test_cpu12_load.py       # CPU load testing
│   └── test_gpu_transition.py   # GPU transition testing
├── scripts/
│   ├── download_models.py       # Model downloader from HuggingFace
│   ├── setup.sh                 # Environment setup
│   ├── test_setup.py            # Setup validation
│   ├── kill_all_containers.sh   # Docker cleanup
│   ├── enable_mps.sh            # Enable NVIDIA MPS
│   └── disable_mps.sh           # Disable NVIDIA MPS
├── _models/                     # Local model files for testing
├── models/                      # GGUF model files (not in git)
├── docs/                        # Design documents
├── Thesis/                      # LaTeX thesis (separate git repo)
└── pyproject.toml               # Project configuration
```

## Key Classes (in main_cost_aware.py)

- `HardwareConfig`: Dataclass for a hardware tier (CPU cores, memory, GPU %, hourly cost, parallel slots) with helpers for Docker image and config ID
- `Message`, `ChatCompletionRequest`, `ChatCompletionChoice`, `ChatCompletionResponse`: Pydantic models for the OpenAI-compatible API
- `ThroughputTracker`: Tracks throughput using a time-weighted EMA fed by streaming token counts, with per-model state
- `DemandTracker`: Simple sliding-window token-rate tracker (legacy, used by scaling demo lifecycle)
- `Container`: Manages a single Docker container running a llama.cpp server (start, stop, health check, endpoint URL)
- `CostAwareAutoscaler`: Orchestrates container lifecycle, scaling decisions, cost tracking, and model discovery across hardware tiers
- `DisaggregatedPrefillManager`: Manages a persistent GPU container for prefill; handles prefill_and_save, restore_on_cpu, and slot file cleanup

### Key Top-Level Functions

- `_load_hardware_configs()`: Loads hardware configs and measured throughput from `hardware_configs.json`
- `get_cost_per_token(model, config)`: Computes cost per token as `hourly_cost / (throughput * 3600)`
- `select_config_per_request()`: Core scaling decision function based on per-request tok/s and active concurrency
- `_async_container_start(container)`: Async container start with up to 90 health-check retries
- `_async_container_stop(container)`: Async container stop via `docker stop`
- `_poll_metrics(container)`: Fetches `/metrics` from a llama.cpp container and parses Prometheus-style counters
- `_update_per_request_ema()`: Updates the per-request tok/s EMA with time-weighted exponential decay
- `_update_active_requests_ema()`: Updates the active-requests EMA with time-weighted exponential decay
- `_streaming_counter_loop()`: Background loop: samples active requests EMA and computes per-request tok/s
- `_metrics_polling_loop()`: Background loop: polls `/metrics` endpoint for predicted TPS gauge (logging only)
- `_background_scaling_loop()`: Background loop: checks scaling conditions every `SCALING_CHECK_INTERVAL` seconds and executes scale up/down
- `lifespan(app)`: FastAPI lifespan context manager for startup/shutdown

### FastAPI Endpoints

- `GET /health`: Server health status, count of ready containers, loaded model names
- `GET /status`: Full autoscaler status: per-model config, EMA values, throughput, cost, scaling state
- `GET /v1/models`: List of loaded model names (OpenAI-compatible)
- `POST /v1/chat/completions`: OpenAI-compatible chat completion endpoint; proxies to llama.cpp via SSE streaming

## Scaling Logic

- Per-request tok/s measured from SSE streaming feeds into an EMA
- Scale UP: `per_request_tps_ema < MIN_TPS_THRESHOLD`
- Scale DOWN requires all three conditions:
  1. `per_request_tps_ema >= MIN_TPS_THRESHOLD`
  2. `active_requests_ema <= SCALE_DOWN_CONCURRENCY`
  3. The next cheaper tier can still serve current concurrency above threshold with a 1.5x safety margin
- Directional cooldowns: scale-up uses `COOLDOWN`, scale-down uses `COOLDOWN_DOWN` (can be longer)
- Recent activity hysteresis: if the model had active requests within `RECENT_ACTIVITY_WINDOW` seconds but currently has 0, the effective concurrency is inflated to block premature scale-down
- After scaling, the EMA is reset and the system waits for the first token on the new config before resuming scaling checks (EMA seeded at `MIN_TPS_THRESHOLD`)
- On failure, the old container is rolled back
- Hardware configs sorted by cost; scaling moves one tier up/down at a time

## Module-Level Constants

- `METRICS_POLL_INTERVAL`: `1.0` seconds
- `SCALING_CHECK_INTERVAL`: `10` seconds
- `EMA_ALPHA`: `2.0 / (EMA_WINDOW + 1)` (default ~0.0083 for 240s window)
- `COOLDOWN_SECONDS`: `300` (hardcoded default for `CostAwareAutoscaler`)
- `DEMAND_WINDOW`: `180` seconds
- `_RE_PREDICTED`, `_RE_PROMPT`, `_RE_PREDICTED_TPS`: Compiled regex patterns for parsing Prometheus-style `/metrics` output from llama.cpp

## Environment Variables

- `E2E_COOLDOWN`: Cooldown between scaling events in seconds (default: 300)
- `E2E_COOLDOWN_DOWN`: Separate cooldown for scale-down (default: same as E2E_COOLDOWN)
- `E2E_MIN_TPS`: Minimum tokens/second threshold for scale-up (default: 10.0)
- `E2E_SCALE_DOWN_CONCURRENCY`: Max active requests EMA for scale-down (default: 5.0)
- `E2E_EMA_WINDOW`: EMA window size in seconds (default: 240)
- `E2E_MODELS_DIR`: Models directory path (default: ./models)
- `E2E_MODEL_NAME`: Specific model to load (default: empty, auto-detect first model)
- `E2E_RECENT_ACTIVITY_WINDOW`: Seconds of inactivity before allowing scale-down (default: 30.0)
- `E2E_INITIAL_CONFIG`: Initial hardware config ID to start with (default: empty, cheapest config)
- `E2E_DISAGGREGATED`: Enable disaggregated prefill mode (default: empty/disabled, set to "1" to enable)
- `E2E_SLOT_SAVE_DIR`: Shared volume path for KV cache transfer (default: /tmp/llama_slots)
- `E2E_DISAGG_CTX_PER_SLOT`: Per-slot context size in disaggregated mode (default: 2048)



# Academic Research Skills

A suite of Claude Code skills for rigorous academic research, paper writing, peer review, and pipeline orchestration.

## Skills Overview

| Skill | Purpose | Key Modes |
|-------|---------|-----------|
| `deep-research` v2.2 | Universal 10-agent research team | full, quick, socratic, review, lit-review, fact-check |
| `academic-paper` v2.2 | 10-agent academic paper writing | full, plan, outline-only, revision, abstract-only, lit-review, format-convert, citation-check |
| `academic-paper-reviewer` v1.3 | Multi-perspective paper review (5 reviewers) | full, re-review, quick, methodology-focus, guided |
| `academic-pipeline` v2.2 | Full pipeline orchestrator | (coordinates all above) |

## Routing Rules

1. **academic-pipeline vs individual skills**: academic-pipeline = full pipeline orchestrator (research → write → review → revise → finalize). If the user only needs a single function (just research, just write, just review), trigger the corresponding skill directly without the pipeline.

2. **deep-research vs academic-paper**: Complementary. deep-research = upstream research engine (investigation + fact-checking), academic-paper = downstream publication engine (paper writing + bilingual abstracts). Recommended flow: deep-research → academic-paper.

3. **deep-research socratic vs full**: socratic = guided Socratic dialogue to help users clarify their research question. full = direct production of research report. When the user's research question is unclear, suggest socratic mode.

4. **academic-paper plan vs full**: plan = chapter-by-chapter guided planning via Socratic dialogue. full = direct paper production. When the user wants to think through their paper structure, suggest plan mode.

5. **academic-paper-reviewer guided vs full**: guided = Socratic review that engages the author in dialogue about issues. full = standard multi-perspective review report. When the user wants to learn from the review, suggest guided mode.

## Key Rules

- All claims must have citations
- Evidence hierarchy respected (meta-analyses > RCTs > cohort > case reports > expert opinion)
- Contradictions disclosed with evidence quality comparison
- AI disclosure in all reports
- Default output language matches user input (Traditional Chinese or English)

## Full Academic Pipeline

```
deep-research (socratic/full)
  → academic-paper (plan/full)
    → academic-paper-reviewer (full/guided)
      → academic-paper (revision)
        → academic-paper-reviewer (re-review, max 2 loops)
          → academic-paper (format-convert → final output)
```

## Handoff Protocol

### deep-research → academic-paper
Materials: RQ Brief, Methodology Blueprint, Annotated Bibliography, Synthesis Report, INSIGHT Collection

### academic-paper → academic-paper-reviewer
Materials: Complete paper text. field_analyst_agent auto-detects domain and configures reviewers.

### academic-paper-reviewer → academic-paper (revision)
Materials: Editorial Decision Letter, Revision Roadmap, Per-reviewer detailed comments

## Version Info
- **Version**: 2.0
- **Last Updated**: 2025-03-05
- **Author**: Cheng-I Wu
- **License**: CC-BY-NC 4.0