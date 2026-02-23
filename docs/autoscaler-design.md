# Cost-Aware Autoscaler Design

## Overview

The cost-aware autoscaler dynamically selects hardware configurations for
ML inference containers based on real-time throughput measurements, optimizing
for the lowest cost-per-token while maintaining sufficient capacity.

## Architecture

```
                    ┌─────────────────┐
  Requests ──────► │  FastAPI Proxy   │
                    │  (scaling_demo_ │
                    │   server.py)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Scaling Logic  │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │  Streaming  │ │  ◄── count tokens from SSE chunks
                    │ │  counter    │ │      delta(count) / delta(wall_time)
                    │ └──────┬──────┘ │
                    │        ▼        │
                    │ ┌─────────────┐ │
                    │ │  4min EMA   │ │  ◄── α = 2/(240+1), time-correct decay
                    │ └──────┬──────┘ │
                    │        ▼        │
                    │  select_config() │  ◄── Threshold-based (80% up / 30% down)
                    │                 │
                    │  cooldown (5m)  │  ◄── Starts after first token on new container
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Docker   │  │ Docker   │  │ Docker   │
        │ cpu_4    │  │ gpu_25   │  │ gpu_100  │
        │ llama.cpp│  │ llama.cpp│  │ llama.cpp│
        └──────────┘  └──────────┘  └──────────┘
```

## Throughput Signal: Streaming Token Counter

### Why Not /metrics?

The initial approach polled the llama.cpp `/metrics` endpoint for the
`tokens_predicted_total` Prometheus counter. This counter only increments
when a request completes — not incrementally during generation. With
`--parallel 32` and long generations (256 tokens), this means:

- Long periods of 0 tok/s while requests are in-flight
- Sudden spikes when multiple requests finish simultaneously
  (e.g., 2048 tokens in 1s when 8 requests complete at once)
- The EMA gets inflated by these spikes (~22 tok/s observed vs ~5 tok/s
  actual per-slot speed on cpu_12)

The `predicted_tokens_seconds` gauge from `/metrics` is per-slot speed,
not aggregate throughput — it shows ~5.2 tok/s regardless of how many
slots are active.

### Streaming Approach

The proxy calls llama.cpp's `/completion` endpoint with `"stream": true`.
Each SSE chunk includes a `tokens` field containing the actual token IDs
generated in that chunk. The proxy counts `len(tokens)` per chunk and
increments a global `_streaming_token_count[model]` counter in real-time.

A background loop (`_streaming_throughput_loop`) reads this counter every
1s and computes:

```
tps = delta(streaming_token_count) / delta(wall_time)
```

This gives smooth, real-time aggregate throughput across all concurrent
slots — no spikes, no idle gaps.

### Accuracy Guarantee

At the end of each request, the `stop` chunk includes `timings.predicted_n`
(the authoritative token count from llama.cpp). The proxy asserts that
the sum of `len(tokens)` across all chunks equals `timings.predicted_n`.
A mismatch raises a RuntimeError — no silent reconciliation.

### /metrics Still Used

The `/metrics` endpoint is still polled for the `predicted_tokens_seconds`
gauge value, which is logged and plotted for comparison. It no longer
drives the EMA or scaling decisions.

## 4-Minute EMA

The instantaneous tok/s feeds a continuous-time EMA:

```python
EMA_ALPHA = 2.0 / (240 + 1)  # ~4min window
decay = (1.0 - EMA_ALPHA) ** dt
ema = ema * decay + (1.0 - decay) * tps
```

The `(1 - decay)` additive term is critical for time-correctness — using
`EMA_ALPHA` directly is only correct when `dt = 1s`.

Seeded with the first observation.

## Scaling Decision

```python
SCALE_UP_MULT = 0.8    # scale up at 80% of current capacity
SCALE_DOWN_MULT = 0.3  # scale down at 30% of cheaper capacity

def select_config(current_config, throughput_ema):
    current_capacity = MEASURED_THROUGHPUT[current_config]

    # Scale UP
    if throughput_ema >= SCALE_UP_MULT * current_capacity:
        return next_more_expensive_config

    # Scale DOWN
    cheaper_capacity = MEASURED_THROUGHPUT[cheaper_config]
    if throughput_ema <= SCALE_DOWN_MULT * cheaper_capacity:
        return cheaper_config

    return current_config
```

## Effective Thresholds

| Transition        | Scale-Up (80% of current) | Scale-Down (30% of cheaper) | Hysteresis Band |
|-------------------|---------------------------|-----------------------------|-----------------|
| cpu_4 ↔ cpu_12    | ≥ 25.6 tok/s              | ≤ 9.6 tok/s                 | 16.0 tok/s      |
| cpu_12 ↔ gpu_25   | ≥ 37.6 tok/s              | ≤ 14.1 tok/s                | 23.5 tok/s      |
| gpu_25 ↔ gpu_100  | ≥ 117.6 tok/s             | ≤ 44.1 tok/s                | 73.5 tok/s      |

## First-Token Gating

After a scaling event (stop old → start new container):

1. EMA updates are paused
2. Scaling decisions are suspended
3. Cooldown timer has not started

Once the streaming token counter shows new tokens arriving:
- The wait flag clears
- EMA seeds with the first observation
- Cooldown timer starts (5 minutes)

## Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `SCALE_UP_MULT` | 0.8 | Scale up when using 80% of current capacity |
| `SCALE_DOWN_MULT` | 0.3 | Scale down when load fits in 30% of cheaper config |
| `EMA_ALPHA` | 2/(240+1) | 4-minute EMA window |
| `COOLDOWN` | 300s | 5 minutes between scaling events |
| `METRICS_POLL_INTERVAL` | 1.0s | Poll /metrics every second |

## Scaling Transitions

All scaling transitions are simple: stop old container, start new container.
No drain, no request queuing. In-flight requests to the old container will
fail and be retried by the client.

## Key Server Functions

- `_streaming_throughput_loop()` — reads streaming token counter every 1s, computes delta tok/s, feeds EMA
- `_poll_metrics()` — fetches /metrics, parses prometheus gauge (logging only)
- `_update_ema()` — time-correct EMA update
- `select_config()` — threshold-based scaling decision

## Background Loops

- `_streaming_throughput_loop()` — runs every 1s, reads `_streaming_token_count`, computes delta tok/s, feeds EMA, handles first-token gating
- `_metrics_polling_loop()` — runs every 1s, polls /metrics for `predicted_tokens_seconds` gauge (logging/plotting only, does not drive EMA)
- `_background_scaling_loop()` — runs every 10s, checks thresholds, triggers container swap if needed

## Validation

Simulation: 320/320 passes. Real hardware demo exercises the full staircase.
See `docs/scaling-simulation-experiments.md`.
