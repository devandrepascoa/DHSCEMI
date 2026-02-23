# Design: Per-Request tok/s Scaling Metric

## Overview

Replace the aggregate streaming tok/s scaling metric with per-request tok/s. The scaling decision becomes:

- **Scale UP**: `per_request_tps_ema < MIN_TPS_THRESHOLD`
- **Scale DOWN**: `per_request_tps_ema >= MIN_TPS_THRESHOLD` AND `active_requests_ema <= SCALE_DOWN_CONCURRENCY`

## Architecture Changes

The existing request flow in `/v1/chat/completions` already streams SSE tokens and counts them via `_streaming_token_count`. We add per-request tracking alongside it.

```
Request arrives
  → SSE stream starts
  → First token: record first_token_time
  → Each token: increment per-request counter (+ existing global counter)
  → Last token: compute per_request_tps = count / (now - first_token_time)
  → Feed per_request_tps into per_request_tps_ema
```

### New Module-Level State

```python
MIN_TPS_THRESHOLD = float(os.environ.get("E2E_MIN_TPS", "8.0"))
SCALE_DOWN_CONCURRENCY = float(os.environ.get("E2E_SCALE_DOWN_CONCURRENCY", "2.0"))

# Per-request EMA (replaces aggregate throughput EMA for scaling decisions)
_per_request_tps_ema: Dict[str, float] = {}
_per_request_tps_ema_time: Dict[str, float] = {}
_per_request_waiting_first: Dict[str, bool] = {}

# Active requests EMA
_active_requests_ema: Dict[str, float] = {}
_active_requests_ema_time: Dict[str, float] = {}
```

### Removed/Replaced

- `_throughput_ema` / `_last_ema_time` — replaced by `_per_request_tps_ema`
- `_streaming_throughput_loop` — no longer needed for scaling decisions (keep `_streaming_token_count` for logging/status)
- `select_config()` function — replaced by `select_config_per_request()`
- `SCALE_UP_MULT` / `SCALE_DOWN_MULT` / `MEASURED_THROUGHPUT` — no longer used for scaling decisions (keep `MEASURED_THROUGHPUT` for cost-per-token display in /status)

### New `select_config_per_request()`

```python
def select_config_per_request(
    current_config: HardwareConfig,
    per_request_tps_ema: float,
    active_requests_ema: float,
    configs_by_cost: Optional[List[HardwareConfig]] = None,
) -> HardwareConfig:
    if configs_by_cost is None:
        configs_by_cost = sorted(HARDWARE_CONFIGS, key=lambda c: c.hourly_cost)

    current_idx = next(
        i for i, c in enumerate(configs_by_cost)
        if c.config_id() == current_config.config_id()
    )

    # Scale UP: per-request speed below minimum
    if per_request_tps_ema < MIN_TPS_THRESHOLD:
        if current_idx + 1 < len(configs_by_cost):
            return configs_by_cost[current_idx + 1]

    # Scale DOWN: speed acceptable AND low concurrency
    if current_idx > 0:
        if (per_request_tps_ema >= MIN_TPS_THRESHOLD
                and active_requests_ema <= SCALE_DOWN_CONCURRENCY):
            return configs_by_cost[current_idx - 1]

    return current_config
```

### Per-Request Tracking in `/v1/chat/completions`

Inside the existing SSE loop, add per-request timing:

```python
req_token_count = 0
req_first_token_time = None

async for raw_line in resp.content:
    # ... existing parsing ...
    if n_tok > 0:
        predicted_n += n_tok
        _streaming_token_count[request.model] += n_tok  # keep global counter
        req_token_count += n_tok
        if req_first_token_time is None:
            req_first_token_time = time.time()

# After stream ends:
if req_token_count > 0 and req_first_token_time is not None:
    req_duration = time.time() - req_first_token_time
    if req_duration > 0:
        req_tps = req_token_count / req_duration
        _update_per_request_ema(request.model, req_tps, time.time())
```

### Active Requests EMA Sampling

A background loop samples `container.active_requests` every 1s and feeds into `_active_requests_ema`:

```python
async def _active_requests_ema_loop():
    while True:
        await asyncio.sleep(METRICS_POLL_INTERVAL)
        now = time.time()
        for model_name, container in autoscaler.containers.items():
            if not container.is_ready:
                continue
            _update_active_requests_ema(model_name, float(container.active_requests), now)
```

### Background Scaling Loop Changes

Replace the throughput EMA check with per-request EMA + active requests EMA:

```python
ema = _per_request_tps_ema.get(model_name, MIN_TPS_THRESHOLD)
active_ema = _active_requests_ema.get(model_name, 0.0)
optimal = select_config_per_request(current_config, ema, active_ema)
```

### /status Endpoint Changes

Add `per_request_tps_ema` and `active_requests_ema` to the status output. Keep `throughput_ema` as the streaming aggregate for display/logging.

## Files Changed

- `main_cost_aware.py` — all scaling logic changes
- `tests/test_cost_aware.py` — rewrite select_config tests, add per-request tests
- `benchmarks/scaling_demo.py` — pass `E2E_MIN_TPS` and `E2E_SCALE_DOWN_CONCURRENCY` env vars
