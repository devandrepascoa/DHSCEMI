# Design Document: Cost-Aware Autoscaler

## Overview

This design extends `main_simple.py` into a new file `main_cost_aware.py` to implement cost-aware autoscaling for LLM inference. The autoscaler selects the hardware configuration (CPU or GPU) with the lowest cost-per-token that can handle the current workload demand.

The key insight: at low demand, cheap CPU configs are most cost-efficient; at high demand, GPU becomes more cost-efficient despite higher hourly cost due to much higher throughput.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│  /v1/chat/completions  →  Autoscaler.get_container()        │
│  /status               →  Autoscaler.get_status()           │
│  /health               →  Health check                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CostAwareAutoscaler                       │
├─────────────────────────────────────────────────────────────┤
│  - configs: List[HardwareConfig]  (with cost + throughput)  │
│  - demand_tracker: DemandTracker  (tokens/sec sliding window)│
│  - current_config: Dict[model, HardwareConfig]              │
│  - last_scale_time: Dict[model, float]                      │
├─────────────────────────────────────────────────────────────┤
│  + select_optimal_config(model) → HardwareConfig            │
│  + check_scaling(model) → Optional[HardwareConfig]          │
│  + scale_to(model, config) → bool                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Container                              │
│  (Docker container running llama.cpp server)                │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### HardwareConfig (dataclass)

```python
@dataclass
class HardwareConfig:
    cpu_cores: Optional[int] = None
    memory: Optional[str] = None
    gpu_percentage: Optional[int] = None
    hourly_cost: float = 0.0

    @property
    def container_type(self) -> str:
        return "gpu" if self.gpu_percentage else "cpu"

    @property
    def image(self) -> str:
        if self.container_type == "gpu":
            return "ghcr.io/ggml-org/llama.cpp:full-cuda"
        return "ghcr.io/ggml-org/llama.cpp:full"

    def config_id(self) -> str:
        if self.gpu_percentage:
            return f"gpu_{self.gpu_percentage}"
        return f"cpu_{self.cpu_cores}"
```

### Available Configurations

GPU configs include `memory` and `cpu_cores` since GPU containers still need CPU/memory for host-side work:

```python
HARDWARE_CONFIGS = [
    HardwareConfig(cpu_cores=1, memory="4g", hourly_cost=0.10),                          # Cheapest CPU
    HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40),                          # Mid-tier CPU
    HardwareConfig(cpu_cores=8, memory="16g", hourly_cost=0.80),                         # High-tier CPU
    HardwareConfig(cpu_cores=2, memory="8g", gpu_percentage=50, hourly_cost=1.00),       # Partial GPU
    HardwareConfig(cpu_cores=2, memory="16g", gpu_percentage=100, hourly_cost=2.00),     # Full GPU
]
```

### Throughput and Cost Functions

```python
# Default throughput by config_id (tokens/second)
DEFAULT_THROUGHPUT: Dict[str, float] = {
    "cpu_1": 4.0,
    "cpu_4": 12.0,
    "cpu_8": 18.0,
    "gpu_50": 50.0,
    "gpu_100": 100.0,
}

# Optional per-model overrides (model_name -> config_id -> throughput)
MODEL_THROUGHPUT_OVERRIDES: Dict[str, Dict[str, float]] = {}

def get_throughput(model: str, config: HardwareConfig) -> float:
    """Get tokens/second. Checks model-specific overrides first, then defaults."""
    config_id = config.config_id()
    if model in MODEL_THROUGHPUT_OVERRIDES:
        if config_id in MODEL_THROUGHPUT_OVERRIDES[model]:
            return MODEL_THROUGHPUT_OVERRIDES[model][config_id]
    return DEFAULT_THROUGHPUT.get(config_id, 1.0)

def get_cost_per_token(model: str, config: HardwareConfig) -> float:
    """Cost per token = hourly_cost / (throughput * 3600). Returns inf if throughput <= 0."""
    throughput = get_throughput(model, config)
    if throughput <= 0:
        return float('inf')
    return config.hourly_cost / (throughput * 3600)
```

### DemandTracker

Tracks recent token usage to estimate current demand. Accepts an optional `clock` callable for testability (defaults to `time.time`):

```python
class DemandTracker:
    def __init__(self, window_seconds: int = 60, clock: Callable[[], float] = None):
        self.window_seconds = window_seconds
        self.clock = clock or time.time
        self.token_events: Dict[str, deque] = defaultdict(deque)

    def record_tokens(self, model: str, token_count: int) -> None:
        now = self.clock()
        self.token_events[model].append((now, token_count))
        self._cleanup_old_events(model)

    def get_demand(self, model: str) -> float:
        self._cleanup_old_events(model)
        events = self.token_events.get(model, [])
        if not events:
            return 0.0
        total_tokens = sum(tokens for _, tokens in events)
        return total_tokens / self.window_seconds

    def _cleanup_old_events(self, model: str) -> None:
        cutoff = self.clock() - self.window_seconds
        events = self.token_events[model]
        while events and events[0][0] < cutoff:
            events.popleft()
```

### CostAwareAutoscaler

Scaling decisions are evaluated **on each incoming request** (inside `get_container()`). The cooldown period prevents actual scaling from happening too frequently. Also accepts a `clock` for testability:

```python
class CostAwareAutoscaler:
    def __init__(
        self,
        configs: List[HardwareConfig],
        cooldown_seconds: float = 300.0,
        clock: Callable[[], float] = None,
    ):
        self.configs = configs
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock or time.time

        self.demand_tracker = DemandTracker(clock=self.clock)
        self.current_config: Dict[str, HardwareConfig] = {}
        self.last_scale_time: Dict[str, float] = {}
        self.containers: Dict[str, Container] = {}

    def select_optimal_config(self, model: str, demand: float) -> HardwareConfig:
        """Select config with lowest cost_per_token that can handle demand."""
        viable = [c for c in self.configs if get_throughput(model, c) >= demand]
        if not viable:
            return max(self.configs, key=lambda c: get_throughput(model, c))
        return min(viable, key=lambda c: get_cost_per_token(model, c))

    def check_scaling(self, model: str) -> Optional[HardwareConfig]:
        """Check if scaling is needed, respecting cooldown."""
        now = self.clock()
        last_scale = self.last_scale_time.get(model, 0)
        if now - last_scale < self.cooldown_seconds:
            return None

        current = self.current_config.get(model)
        demand = self.demand_tracker.get_demand(model)
        optimal = self.select_optimal_config(model, demand)

        if current is None or optimal.config_id() != current.config_id():
            return optimal
        return None
```

### Graceful Scaling (scale_to)

Start new container first, then drain and stop old. Enforces a max drain timeout (default 60s) to prevent indefinite waiting on stuck requests:

```python
MAX_DRAIN_TIMEOUT_SECONDS = 60

async def scale_to(self, model: str, new_config: HardwareConfig):
    new_container = Container(model, model_path, new_config, port)
    if not await new_container.start():
        logger.error(f"Failed to start new container, keeping current")
        return

    old_container = self.containers.get(model)
    self.containers[model] = new_container
    self.current_config[model] = new_config
    self.last_scale_time[model] = self.clock()

    if old_container:
        deadline = self.clock() + MAX_DRAIN_TIMEOUT_SECONDS
        while old_container.active_requests > 0 and self.clock() < deadline:
            await asyncio.sleep(1)
        await old_container.stop()
```

## Configuration Constants

```python
COOLDOWN_SECONDS = 300          # 5 minutes between scaling actions
DEMAND_WINDOW_SECONDS = 60      # 1-minute sliding window for demand
MAX_DRAIN_TIMEOUT_SECONDS = 60  # Max wait for in-flight requests during scaling
```

## Correctness Properties

### Property 1: Docker Image Selection

*For any* HardwareConfig, if `gpu_percentage` is set (not None and > 0), the `image` property SHALL return the CUDA image, otherwise it SHALL return the CPU image.

**Validates: Requirements 1.6**

### Property 2: Cost-Per-Token Calculation

*For any* HardwareConfig with `hourly_cost > 0` and throughput `t > 0`, `get_cost_per_token()` SHALL equal `hourly_cost / (t * 3600)`.

**Validates: Requirements 2.1**

### Property 3: Demand Calculation

*For any* sequence of token events recorded within the last `window_seconds`, `get_demand()` SHALL return `sum(tokens) / window_seconds`.

**Validates: Requirements 3.1**

### Property 4: Optimal Config Selection

*For any* demand value `d`, `select_optimal_config()` SHALL return the config with minimum `cost_per_token` among configs where `throughput >= d`. If no config can meet demand, return the highest throughput config.

**Validates: Requirements 3.2**

### Property 5: Cooldown Enforcement

*For any* model, if a scaling action occurred at time `t1`, `check_scaling()` SHALL return `None` for all calls where `current_time < t1 + cooldown_seconds`.

**Validates: Requirements 4.1**

### Property 6: Graceful Scaling Order

*For any* scaling operation, the new container SHALL be started and ready before the old container is stopped, and the old container SHALL be drained (up to max timeout) before stopping.

**Validates: Requirements 5.1, 5.2, 5.3**

## Error Handling

1. **Container start failure**: Log error, keep current config running
2. **Zero throughput config**: Return `float('inf')` for cost_per_token (never selected)
3. **No viable config**: Fall back to highest capacity config
4. **Drain timeout**: Force-stop old container after MAX_DRAIN_TIMEOUT_SECONDS

## Testing Strategy

All tests use plain pytest with mocked inference — no Docker containers or model inference needed. DemandTracker and CostAwareAutoscaler accept injectable `clock` functions for deterministic time control in tests.

### Unit Tests (pytest)

- HardwareConfig: creation, config_id, image, container_type
- get_cost_per_token: formula verification with known values
- get_throughput: default fallback and model-specific overrides
- DemandTracker: record/get_demand with controlled clock
- select_optimal_config: cheapest viable config for various demands
- check_scaling: cooldown enforcement with controlled clock

### Integration Tests (pytest, mocked containers)

- Autoscaler selects cheapest config at low demand
- Autoscaler scales up when demand increases
- Autoscaler scales down when demand decreases
- Cooldown prevents rapid oscillation
- Graceful scaling: new container ready before old stopped, drain timeout enforced

### E2E Tests (pytest, real Docker containers)

Tests in `tests/test_cost_aware_e2e.py`. These spin up the actual `main_cost_aware.py` server and real llama.cpp Docker containers using the model `models/01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf`. Requires Docker running and the model file present.

Uses a session-scoped fixture that starts the proxy on a random port. Tests use CPU-only configs (1 and 4 cores) with a short cooldown (e.g., 10s) to make scaling observable in reasonable time. Sends real chat completion requests and verifies:

- Server starts and loads model into cheapest config (cpu_1)
- Sending requests returns valid chat completions
- After sustained load, autoscaler scales up to a higher CPU config
- After load stops and cooldown elapses, autoscaler scales back down
- `/status` endpoint reflects current config, demand, and cost metrics

### E2E Tests with GPU (pytest, real Docker containers + GPU)

Tests in `tests/test_cost_aware_e2e_gpu.py`. Same model file, but uses the full config list (CPU + GPU). Requires NVIDIA GPU and nvidia-container-toolkit installed. Verifies:

- Server starts on cheapest CPU config
- Under sustained high load that exceeds all CPU configs, autoscaler transitions to a GPU config (gpu_50 or gpu_100)
- Chat completions work correctly on GPU containers
- After load drops and cooldown elapses, autoscaler scales back down to CPU
- `/status` shows GPU-specific details (gpu_percentage, CUDA image, cost metrics)
