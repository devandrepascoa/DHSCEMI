# Requirements: Per-Request tok/s Scaling Metric

## Introduction

Replace the current aggregate streaming tok/s scaling metric with a per-request tok/s metric. The current approach fails to trigger scale-up at low concurrency because aggregate throughput stays low even when the hardware is the bottleneck. Per-request tok/s captures actual user-experienced performance: when hardware is saturated, each request's generation speed drops.

## Problem

Current metric: aggregate streaming tok/s (sum of all tokens generated per second across all requests).
- At 1 worker on cpu_4: aggregate ~7-9 tok/s. Scale-up threshold is 25.6 tok/s (80% of 32). Never triggers.
- Only triggers scale-up when enough concurrent requests saturate aggregate throughput.
- Conflates concurrency with per-request performance.

## Glossary

- **per_request_tps**: The generation tok/s for a single request, measured from the SSE token stream as `token_count / (last_token_time - first_token_time)`.
- **per_request_tps_ema**: Exponential moving average of per_request_tps values, updated on each request completion.
- **MIN_TPS_THRESHOLD**: Configurable minimum acceptable per-request tok/s. When per_request_tps_ema drops below this, scale up. Single parameter that defines acceptable user experience.
- **active_requests**: Number of concurrent in-flight requests at any point in time.
- **active_requests_ema**: EMA of the active request count, sampled periodically.

## Requirements

### Requirement 1: Track Per-Request Generation Speed via Streaming

**User Story:** As the autoscaler, I want to track the generation tok/s of each completed request measured from the SSE token stream, so I capture real wall-clock performance including queuing and contention effects.

#### Acceptance Criteria

1. THE system already counts tokens per SSE chunk in the `/v1/chat/completions` handler (via `_streaming_token_count`). Instead of only incrementing a global counter, each request SHALL also track its own token count and wall-clock time (first token to last token).
2. WHEN a request completes, THE system SHALL compute `per_request_tps = token_count / (last_token_time - first_token_time)` for that request.
3. THE system SHALL maintain a per_request_tps_ema per model, updated on each request completion with the computed per_request_tps.
4. THE EMA alpha SHALL be configurable (default: time-based EMA with configurable window, same as current E2E_EMA_WINDOW).
5. AFTER a scaling event, THE per_request_tps_ema SHALL be reset and wait for the first completed request on the new hardware before resuming scaling decisions.

### Requirement 2: Track Active Request Concurrency

**User Story:** As the autoscaler, I want to know the current concurrency level so I can distinguish "hardware saturated" from "low demand" when per-request tok/s is above threshold.

#### Acceptance Criteria

1. THE system SHALL track active_requests per model (already exists via container.active_requests).
2. THE system SHALL maintain an active_requests_ema, sampled every METRICS_POLL_INTERVAL (1s).
3. THE active_requests_ema SHALL use the same EMA alpha as per_request_tps_ema.

### Requirement 3: Scale-Up Decision Using MIN_TPS_THRESHOLD

**User Story:** As the autoscaler, I want to scale up when per-request performance drops below a configurable minimum, indicating the current hardware can't deliver acceptable user experience.

#### Acceptance Criteria

1. MIN_TPS_THRESHOLD SHALL be a configurable parameter (env var `E2E_MIN_TPS`, default 8.0 tok/s).
2. THE system SHALL scale UP when: `per_request_tps_ema < MIN_TPS_THRESHOLD`.
3. Scale-up SHALL still respect cooldown.
4. Scale-up SHALL move to the next more expensive config in the ordered list (same as current behavior).
5. THE system SHALL NOT scale up if already on the most expensive config.

### Requirement 4: Scale-Down Decision Using Composite Signal

**User Story:** As the autoscaler, I want to scale down when the hardware is underutilized — per-request speed is above threshold AND concurrency is low enough that a cheaper config could handle it.

#### Acceptance Criteria

1. THE system SHALL scale DOWN when BOTH conditions are met:
   a. `per_request_tps_ema >= MIN_TPS_THRESHOLD` (performance is acceptable, hardware not stressed).
   b. `active_requests_ema <= SCALE_DOWN_CONCURRENCY` (low concurrency = not much demand). SCALE_DOWN_CONCURRENCY SHALL be configurable (env var `E2E_SCALE_DOWN_CONCURRENCY`, default 2.0).
2. Scale-down SHALL move to the next cheaper config in the ordered list.
3. Scale-down SHALL still respect cooldown.
4. THE system SHALL NOT scale down if already on the cheapest config.

### Requirement 5: Update Tests

**User Story:** As a developer, I want unit tests that validate the new scaling logic and the scaling demo tests updated accordingly.

#### Acceptance Criteria

1. Tests SHALL verify scale-up triggers when per_request_tps_ema drops below MIN_TPS_THRESHOLD.
2. Tests SHALL verify scale-down triggers only when BOTH per_request_tps_ema >= MIN_TPS_THRESHOLD AND active_requests_ema is low.
3. Tests SHALL verify no scaling during cooldown.
4. Tests SHALL verify EMA reset after scaling events.
5. Existing tests in `tests/test_cost_aware.py` that test `select_config` with aggregate throughput SHALL be rewritten to use the new per-request metric.
6. The scaling demo (`benchmarks/scaling_demo.py`) SHALL pass MIN_TPS_THRESHOLD and SCALE_DOWN_CONCURRENCY to the server via env vars.
7. The scaling demo phases and expected sequence SHALL be reviewed and updated if needed to work with the new metric.
