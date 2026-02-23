# Tasks: Per-Request tok/s Scaling Metric

## Task 1: Add per-request tracking constants and state

- [ ] 1.1 Add `MIN_TPS_THRESHOLD` constant from `E2E_MIN_TPS` env var (default 8.0)
- [ ] 1.2 Add `SCALE_DOWN_CONCURRENCY` constant from `E2E_SCALE_DOWN_CONCURRENCY` env var (default 2.0)
- [ ] 1.3 Add module-level dicts: `_per_request_tps_ema`, `_per_request_tps_ema_time`, `_per_request_waiting_first`
- [ ] 1.4 Add module-level dicts: `_active_requests_ema`, `_active_requests_ema_time`

## Task 2: Implement `select_config_per_request()`

- [ ] 2.1 Create `select_config_per_request(current_config, per_request_tps_ema, active_requests_ema, configs_by_cost)` function
- [ ] 2.2 Scale UP when `per_request_tps_ema < MIN_TPS_THRESHOLD` → next more expensive config
- [ ] 2.3 Scale DOWN when `per_request_tps_ema >= MIN_TPS_THRESHOLD` AND `active_requests_ema <= SCALE_DOWN_CONCURRENCY` → next cheaper config
- [ ] 2.4 Return current config if no scaling needed, already at min, or already at max

## Task 3: Add per-request tok/s tracking in `/v1/chat/completions`

- [ ] 3.1 Add `req_token_count` and `req_first_token_time` variables in the SSE loop
- [ ] 3.2 Record `first_token_time` on first token received
- [ ] 3.3 After stream ends, compute `req_tps = req_token_count / (now - first_token_time)` and feed into per-request EMA
- [ ] 3.4 Add `_update_per_request_ema()` helper using same time-based EMA as existing code
- [ ] 3.5 Log per-request tps in the REQ_OK json log

## Task 4: Add active requests EMA loop

- [ ] 4.1 Create `_active_requests_ema_loop()` that samples `container.active_requests` every `METRICS_POLL_INTERVAL`
- [ ] 4.2 Add `_update_active_requests_ema()` helper
- [ ] 4.3 Start the loop in `lifespan()` alongside existing background tasks

## Task 5: Update background scaling loop

- [ ] 5.1 Replace `_throughput_ema` check with `_per_request_tps_ema` + `_active_requests_ema`
- [ ] 5.2 Call `select_config_per_request()` instead of `select_config()`
- [ ] 5.3 Use `_per_request_waiting_first` instead of `_waiting_for_first_token` for the post-scaling gate
- [ ] 5.4 On scaling, reset `_per_request_tps_ema` and `_active_requests_ema` for the model
- [ ] 5.5 Update DEMAND_CHECK log to include `per_request_tps_ema` and `active_requests_ema`

## Task 6: Update /status endpoint

- [ ] 6.1 Add `per_request_tps_ema` and `active_requests_ema` to model status
- [ ] 6.2 Add `min_tps_threshold` and `scale_down_concurrency` to top-level status
- [ ] 6.3 Keep existing `throughput_ema` (streaming aggregate) for display/logging

## Task 7: Remove old aggregate scaling logic

- [ ] 7.1 Remove `_streaming_throughput_loop()` (keep `_streaming_token_count` for logging)
- [ ] 7.2 Remove old `select_config()` function
- [ ] 7.3 Remove `SCALE_UP_MULT` and `SCALE_DOWN_MULT` constants
- [ ] 7.4 Clean up `_throughput_ema`, `_last_ema_time`, `_waiting_for_first_token` state dicts
- [ ] 7.5 Remove the streaming throughput task from `lifespan()`

## Task 8: Update tests

- [ ] 8.1 Rewrite `TestSelectConfig` to test `select_config_per_request()` with MIN_TPS_THRESHOLD
- [ ] 8.2 Test scale-up triggers when per_request_tps_ema < MIN_TPS_THRESHOLD
- [ ] 8.3 Test scale-down triggers when per_request_tps_ema >= MIN_TPS_THRESHOLD AND active_requests_ema <= SCALE_DOWN_CONCURRENCY
- [ ] 8.4 Test no scale-down when concurrency is high even if tps is good
- [ ] 8.5 Test no scale-up when already at max config
- [ ] 8.6 Test no scale-down when already at min config
- [ ] 8.7 Update `TestCheckScaling` to use new metric
- [ ] 8.8 Update imports in test file (remove old `select_config`, add `select_config_per_request`, `MIN_TPS_THRESHOLD`, `SCALE_DOWN_CONCURRENCY`)

## Task 9: Update scaling demo

- [ ] 9.1 Pass `E2E_MIN_TPS` and `E2E_SCALE_DOWN_CONCURRENCY` env vars to server process in `benchmarks/scaling_demo.py`
- [ ] 9.2 Update benchmark metadata log with new parameters
- [ ] 9.3 Update docstring to reflect new scaling metric
