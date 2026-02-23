# Implementation Tasks: Cost-Aware Autoscaler

All implementation goes in `main_cost_aware.py` (new file, based on `main_simple.py`).
All tests go in `tests/test_cost_aware.py` (plain pytest, mocked inference).

## Task 1: Core Data Structures

- [x] 1.1 Create `HardwareConfig` dataclass with `cpu_cores`, `memory`, `gpu_percentage`, `hourly_cost` fields
- [x] 1.2 Implement `config_id()` method returning unique string identifier (`cpu_{cores}` or `gpu_{pct}`)
- [x] 1.3 Implement `image` property returning correct Docker image (CPU: `full`, GPU: `full-cuda`)
- [x] 1.4 Implement `container_type` property returning `"cpu"` or `"gpu"`
- [x] 1.5 Define `HARDWARE_CONFIGS` list with CPU (1, 4, 8 cores) and GPU (50%, 100%) configs — GPU configs must include `memory` and `cpu_cores`
- [x] 1.6 Define `DEFAULT_THROUGHPUT` dict mapping config_id to tokens/second

## Task 2: Throughput and Cost Functions

- [x] 2.1 Implement `get_throughput(model, config)` — check `MODEL_THROUGHPUT_OVERRIDES` first, fall back to `DEFAULT_THROUGHPUT`
- [x] 2.2 Implement `get_cost_per_token(model, config)` — `hourly_cost / (throughput * 3600)`, return `inf` if throughput <= 0

## Task 3: DemandTracker Class

- [x] 3.1 Create `DemandTracker` class with `window_seconds` and injectable `clock` parameter (default `time.time`)
- [x] 3.2 Implement `record_tokens(model, token_count)` storing `(timestamp, count)` tuples in a deque
- [x] 3.3 Implement `get_demand(model)` returning `sum(tokens) / window_seconds` for events within window
- [x] 3.4 Implement `_cleanup_old_events(model)` to evict events older than `clock() - window_seconds`

## Task 4: CostAwareAutoscaler Class

- [x] 4.1 Create `CostAwareAutoscaler` class with configs list, `cooldown_seconds`, and injectable `clock`
- [x] 4.2 Implement `select_optimal_config(model, demand)` — lowest `cost_per_token` among configs with `throughput >= demand`; fall back to highest throughput if none viable
- [x] 4.3 Implement `check_scaling(model)` — return new config if optimal differs from current and cooldown has elapsed, else `None`
- [x] 4.4 Implement `scale_to(model, config)` — start new container, swap references, drain old container (max 60s timeout), stop old container

## Task 5: Container and FastAPI Integration

- [x] 5.1 Port `Container` class from `main_simple.py`, update to accept `HardwareConfig` and generate correct docker args for CPU and GPU
- [x] 5.2 Wire up FastAPI endpoints: `/v1/chat/completions` calls `check_scaling` on each request, records token usage after completion
- [x] 5.3 Update `/status` endpoint to show current config, demand, cost_per_token, and container info per model
- [x] 5.4 Update lifespan to initialize `CostAwareAutoscaler` with `HARDWARE_CONFIGS`

## Task 6: Unit Tests (plain pytest, mocked)

- [x] 6.1 Test `HardwareConfig.image` returns correct image for CPU vs GPU configs (Property 1)
- [x] 6.2 Test `get_cost_per_token()` with known values matches `hourly_cost / (throughput * 3600)` (Property 2)
- [x] 6.3 Test `DemandTracker.get_demand()` with injected clock and simulated token events (Property 3)
- [x] 6.4 Test `select_optimal_config()` returns cheapest viable config for various demand levels (Property 4)
- [x] 6.5 Test `check_scaling()` returns `None` during cooldown period using injected clock (Property 5)
- [x] 6.6 Test `get_throughput()` falls back to `DEFAULT_THROUGHPUT` for unknown models

## Task 7: Integration Tests (plain pytest, mocked containers)

- [x] 7.1 Test autoscaler selects cheapest config at low demand (cpu_1)
- [x] 7.2 Test autoscaler scales up to higher config when demand increases
- [x] 7.3 Test autoscaler scales down when demand drops
- [x] 7.4 Test cooldown prevents rapid oscillation between configs

## Task 8: E2E Tests (plain pytest, real Docker containers)

Tests in `tests/test_cost_aware_e2e.py`. Requires Docker running and `models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` present.
Uses CPU-only configs with short cooldown (~10s) for observable scaling.

- [x] 8.1 Create session-scoped fixture that starts `main_cost_aware.py` on a random port with CPU-only configs and short cooldown, tears down on completion
- [x] 8.2 Test server starts healthy and loads model into cheapest config (cpu_1)
- [x] 8.3 Test sending a chat completion request returns a valid response with the test model
- [x] 8.4 Test `/status` endpoint returns current config, demand, and cost metrics
- [x] 8.5 Test autoscaler scales up after sustained request load exceeds cpu_1 capacity
- [x] 8.6 Test autoscaler scales back down after load stops and cooldown elapses

## Task 9: E2E Tests with GPU (plain pytest, real Docker containers + GPU)

Tests in `tests/test_cost_aware_e2e_gpu.py`. Requires Docker running, NVIDIA GPU available, nvidia-container-toolkit installed, and `models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` present.
Uses CPU (4, 8, 14 cores) + GPU (50%) configs with short cooldown (~10s). Uses `/test/inject_demand` for synthetic demand injection.

- [x] 9.1 Create session-scoped fixture that starts GPU e2e server on a random port with cpu_4/cpu_8/cpu_14/gpu_50 configs and short cooldown, tears down on completion
- [x] 9.2 Test server starts healthy and loads model into cheapest config (cpu_4)
- [x] 9.3 Test sending a chat completion request returns a valid response on initial CPU config
- [x] 9.4 Test autoscaler scales from CPU to GPU config after injecting synthetic demand exceeding all CPU throughputs
- [x] 9.5 Test autoscaler scales back from GPU to CPU config after demand window expires and cooldown is reset
- [x] 9.6 Test `/status` endpoint shows GPU config details (gpu_percentage, cost metrics) when running on GPU
