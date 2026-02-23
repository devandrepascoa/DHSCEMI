# Scaling Simulation Experiments & Insights

This document records the full experimental journey of tuning the cost-aware
autoscaler to produce a clean 4-tier staircase pattern:

```
cpu_4 → cpu_12 → gpu_25 → gpu_100 → gpu_25 → cpu_12 → cpu_4
```

## Hardware Configs & Measured Throughput

All throughput values come from real benchmarks (`throughput_benchmark.py`)
running DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M on Docker containers with
`--parallel 32` and `max_tokens=256`.

| Config   | CPU Cores | GPU % | Max Throughput (tok/s) | Single-Request (tok/s) | Hourly Cost |
|----------|-----------|-------|------------------------|------------------------|-------------|
| cpu_4    | 4         | —     | 32.0 (batch=4)         | 9.0 (batch=1)          | $0.05       |
| cpu_12   | 12        | —     | 47.0 (batch=4 peak)    | 15.4 (batch=1)         | $0.12       |
| gpu_25   | 2         | 25%   | 147.0 (batch=32)       | 13.3 (batch=1)         | $0.50       |
| gpu_100  | 2         | 100%  | 1064.0 (batch=32)      | 152.9 (batch=1)        | $4.00       |

Cost-per-token ordering (cheapest first):
- cpu_4: 0.43 μ$/tok
- cpu_12: 0.71 μ$/tok
- gpu_25: 0.94 μ$/tok
- gpu_100: 1.04 μ$/tok

Prices are synthetic but chosen to maintain this ordering.

### Why cpu_8 Was Dropped

cpu_8 throughput (42 tok/s) was too close to cpu_12 (47 tok/s) — only 5 tok/s
gap, smaller than EMA noise. Removed for the cleaner cpu_4 → cpu_12 jump.

## Experiment 1: Uniform Headroom (HEADROOM = 0.15)

A config is viable only if `throughput >= demand * (1 + headroom)`.

Results: 135/320 (42%). Oscillation between cpu_12 and gpu_25 during ramp-down.

## Experiment 2: Increased Headroom (HEADROOM = 0.25)

Results: 73/320 (23%) — worse. EMA peak of 37.8 × 1.25 = 47.25 exceeded
cpu_12's 47.0 by 0.25 tok/s.

## Experiment 3: Full Stickiness

If current config handles raw demand, always stay. Results: 0/320 (0%).
Prevented all scale-downs.

## Experiment 4: Asymmetric Hysteresis

Different behavior for scale-up vs scale-down using headroom-based demand
tracking. Results: 320/320 (100%). Worked but relied on demand tracking via
token counting, which produced a sawtooth pattern.

## Experiment 5: /metrics Throughput EMA (Current)

### Key Insight
Instead of tracking demand (tokens requested), track actual throughput
(tokens processed per second) from the llama.cpp `/metrics` endpoint and
compare against measured hardware capacities.

### Throughput Signal

Poll `/metrics` every 1 second, read two Prometheus counters:
- `llamacpp:tokens_predicted_total`
- `llamacpp:prompt_tokens_processed_total`

Compute: `tps = delta(predicted + prompt) / delta_time`

This gives the real aggregate tok/s across all concurrent slots.

### 4-Minute EMA

```python
EMA_ALPHA = 2.0 / (240 + 1)
decay = (1.0 - EMA_ALPHA) ** dt
ema = ema * decay + (1.0 - decay) * tps
```

The additive term must be `(1 - decay)`, not `EMA_ALPHA`. The naive formula
is only correct when `dt = 1`. Seeded with first observation.

### Scaling Decision

```python
SCALE_UP_MULT = 0.8    # scale up at 80% of current capacity
SCALE_DOWN_MULT = 0.3  # scale down at 30% of cheaper capacity
```

### Effective Thresholds

| Transition        | Scale-Up (80% of current) | Scale-Down (30% of cheaper) | Hysteresis Band |
|-------------------|---------------------------|-----------------------------|-----------------|
| cpu_4 ↔ cpu_12    | ≥ 25.6 tok/s              | ≤ 9.6 tok/s                 | 16.0 tok/s      |
| cpu_12 ↔ gpu_25   | ≥ 37.6 tok/s              | ≤ 14.1 tok/s                | 23.5 tok/s      |
| gpu_25 ↔ gpu_100  | ≥ 117.6 tok/s             | ≤ 44.1 tok/s                | 73.5 tok/s      |

### First-Token Gating

After scaling, wait for `/metrics` counters to show new tokens before
resuming EMA updates, scaling decisions, and cooldown timer.

### Results: 320/320 (100%)
- Named scenarios: 20/20
- Realistic noise (dur±20%, tok±15%): 100/100
- Moderate noise (dur±30%, tok±20%, rpm±10%): 100/100
- Extreme noise (dur±40%, tok±30%, rpm±25%): 100/100

### Parameters
- `SCALE_UP_MULT = 0.8`
- `SCALE_DOWN_MULT = 0.3`
- `EMA_ALPHA = 2/(240+1)` (~4min window)
- `COOLDOWN = 300` (5 minutes)
- `METRICS_POLL_INTERVAL = 1.0` (1s)

## Workload Phases

### Simulation Phases (starting from cpu_4)

| Phase       | Duration | Workers | RPM | Target Demand | Expected Config |
|-------------|----------|---------|-----|---------------|-----------------|
| low load    | 15 min   | 1       | 3   | ~7 tok/s      | cpu_4           |
| medium load | 15 min   | 8       | sat | ~72 tok/s     | → cpu_12        |
| high load   | 15 min   | 16      | sat | ~213 tok/s    | → gpu_25        |
| peak load   | 15 min   | 30      | sat | ~4587 tok/s   | → gpu_100       |
| sustain gpu | 10 min   | 30      | sat | ~4587 tok/s   | gpu_100         |
| ramp-down 1 | 15 min   | 4       | 43  | ~100 tok/s    | → gpu_25        |
| ramp-down 2 | 15 min   | 4       | 15  | ~35 tok/s     | → cpu_12        |
| ramp-down 3 | 15 min   | 1       | 3   | ~7 tok/s      | → cpu_4         |
| low load    | 10 min   | 1       | 3   | ~7 tok/s      | cpu_4           |

Expected sequence: `cpu_4 → cpu_12 → gpu_25 → gpu_100 → gpu_25 → cpu_12 → cpu_4`

### Benchmark Phases (starting from cpu_12)

| Phase       | Duration | Workers | RPM | Expected Config |
|-------------|----------|---------|-----|-----------------|
| medium load | 15 min   | 8       | sat | cpu_12 (start)  |
| high load   | 15 min   | 16      | sat | → gpu_25        |
| peak load   | 15 min   | 30      | sat | → gpu_100       |
| ramp-down 1 | 15 min   | 4       | 43  | → gpu_25        |
| ramp-down 2 | 15 min   | 4       | 15  | → cpu_12        |
| ramp-down 3 | 15 min   | 1       | 3   | → cpu_4         |
| low load    | 10 min   | 1       | 3   | cpu_4           |

Expected sequence: `cpu_12 → gpu_25 → gpu_100 → gpu_25 → cpu_12 → cpu_4`

## Real Hardware Scaling Demo

### Architecture

Three processes:

1. **Server** (`benchmarks/scaling_demo_server.py`): FastAPI app with
   /metrics-based throughput EMA scaling. Background tasks handle metrics
   polling (1s) and scaling decisions (10s). Uses
   `asyncio.create_subprocess_exec` for non-blocking container lifecycle.

2. **Client** (`benchmarks/scaling_demo.py`): Drives 7 phases starting from
   cpu_12. Polls `/status` every 5s, logs structured JSON, generates plot.

3. **Live plotter** (`benchmarks/live_plot.py`): Watches server log, parses
   DEMAND_CHECK lines, regenerates 5-panel PNG + HTML every 10s. Panel 2
   shows EMA (green) and raw metrics tok/s (orange).

### Container Configuration

All containers use `--parallel 32`, `--metrics`, and the llama.cpp `:full` image.

| Config   | Docker Flags | Threads | GPU Layers |
|----------|-------------|---------|------------|
| cpu_4    | `--cpus=4`, `--memory=8g` | 4 | 0 |
| cpu_12   | `--cpus=12`, `--memory=8g` | 12 | 0 |
| gpu_25   | `--gpus "device=0"`, `--privileged`, `--memory=8g` | 2 | 99 |
| gpu_100  | `--gpus "device=0"`, `--privileged`, `--memory=16g` | 2 | 99 |

### Logging

Server DEMAND_CHECK logs (every 10s): `throughput_ema`, `metrics_tps`,
`capacity`, `ema_pct_of_capacity`, `scale_up_threshold`, `active_requests`.

### Running

```bash
uv run python benchmarks/scaling_demo.py 2>&1 \
  | tee benchmarks/scaling_demo_logs/run_$(date +%Y%m%d_%H%M%S).log

# In another terminal:
uv run python benchmarks/live_plot.py

# Cleanup:
./scripts/kill_all_containers.sh
```

### Known Issues & Fixes

**EMA formula bug:** `ema * decay + alpha * value` diverges with sub-second
polling. Fixed to `ema * decay + (1 - decay) * value`.

**Pipe buffer deadlock:** Server stdout piped to client filled 64KB buffer.
Fixed by redirecting server stdout to a file.

### Output Files

| File | Description |
|------|-------------|
| `benchmarks/scaling_demo_logs/run_*.log` | Client log |
| `benchmarks/scaling_demo_logs/server_*.log` | Server log |
| `benchmarks/thesis_figures/scaling_demo.{pdf,png}` | Final thesis plot |
| `benchmarks/thesis_figures/scaling_demo_data.json` | Raw data |
| `benchmarks/thesis_figures/scaling_demo_live.{png,html}` | Live plots |

## Files

| File | Role |
|------|------|
| `main_cost_aware.py` | Core data classes (HardwareConfig, Container, etc.) |
| `benchmarks/scaling_demo_server.py` | Server with /metrics throughput EMA scaling |
| `benchmarks/scaling_demo.py` | Benchmark client |
| `benchmarks/scaling_simulation.py` | Offline simulation (320/320 Monte Carlo) |
| `benchmarks/scaling_simulation_plots.py` | Thesis figure generation |
| `benchmarks/live_plot.py` | Real-time plot from server logs |
| `scripts/kill_all_containers.sh` | Kill all running Docker containers |
