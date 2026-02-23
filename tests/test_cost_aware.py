"""
Tests for cost-aware autoscaler (plain pytest, mocked inference).
No Docker containers or real model inference needed.
CostAwareAutoscaler uses injectable clock for deterministic time control.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from main_cost_aware import (
    HardwareConfig,
    ThroughputTracker,
    CostAwareAutoscaler,
    MEASURED_THROUGHPUT,
    MIN_TPS_THRESHOLD,
    SCALE_DOWN_CONCURRENCY,
    get_cost_per_token,
    select_config_per_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CPU_4 = HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.05)
CPU_12 = HardwareConfig(cpu_cores=12, memory="8g", hourly_cost=0.12)
GPU_25 = HardwareConfig(cpu_cores=2, memory="8g", gpu_percentage=25, hourly_cost=0.50)
GPU_100 = HardwareConfig(cpu_cores=2, memory="16g", gpu_percentage=100, hourly_cost=4.00)

ALL_CONFIGS = [CPU_4, CPU_12, GPU_25, GPU_100]
CONFIGS_BY_COST = sorted(ALL_CONFIGS, key=lambda c: c.hourly_cost)

TEST_MODEL = "test-model"


# ===================================================================
# HardwareConfig tests
# ===================================================================

class TestHardwareConfigImage:
    def test_cpu_config_returns_cpu_image(self):
        for cfg in [CPU_4, CPU_12]:
            assert cfg.image == "ghcr.io/ggml-org/llama.cpp:full"

    def test_gpu_config_returns_cuda_image(self):
        for cfg in [GPU_25, GPU_100]:
            assert cfg.image == "ghcr.io/ggml-org/llama.cpp:full-cuda"

    def test_container_type_cpu(self):
        assert CPU_4.container_type == "cpu"

    def test_container_type_gpu(self):
        assert GPU_25.container_type == "gpu"


# ===================================================================
# Cost per token tests
# ===================================================================

class TestCostPerToken:
    def test_cost_per_token_formula(self):
        """cost_per_token = hourly_cost / (throughput * 3600)."""
        for cfg in ALL_CONFIGS:
            throughput = MEASURED_THROUGHPUT[cfg.config_id()]
            expected = cfg.hourly_cost / (throughput * 3600)
            assert get_cost_per_token(TEST_MODEL, cfg) == pytest.approx(expected)

    def test_unknown_config_returns_hourly_cost_over_3600(self):
        """A config not in MEASURED_THROUGHPUT uses fallback throughput=1.0."""
        weird = HardwareConfig(cpu_cores=99, memory="1g", hourly_cost=0.01)
        assert get_cost_per_token(TEST_MODEL, weird) == pytest.approx(0.01 / 3600)


# ===================================================================
# ThroughputTracker tests
# ===================================================================

class TestThroughputTracker:
    def test_ema_starts_at_zero(self):
        tracker = ThroughputTracker()
        assert tracker.get_ema(TEST_MODEL) == 0.0

    def test_ema_seeds_with_first_observation(self):
        tracker = ThroughputTracker()
        tracker.update_ema(TEST_MODEL, 25.0, now=0.0)
        assert tracker.get_ema(TEST_MODEL) == 25.0

    def test_ema_decays_toward_new_value(self):
        tracker = ThroughputTracker()
        tracker.update_ema(TEST_MODEL, 100.0, now=0.0)
        tracker.update_ema(TEST_MODEL, 0.0, now=10.0)
        assert tracker.get_ema(TEST_MODEL) < 100.0

    def test_streaming_token_counter(self):
        tracker = ThroughputTracker()
        tracker.record_streaming_tokens(TEST_MODEL, 5)
        tracker.record_streaming_tokens(TEST_MODEL, 3)
        assert tracker._streaming_count[TEST_MODEL] == 8

    def test_reset_model_clears_state(self):
        tracker = ThroughputTracker()
        tracker.update_ema(TEST_MODEL, 50.0, now=0.0)
        tracker.record_streaming_tokens(TEST_MODEL, 10)
        tracker.reset_model(TEST_MODEL)
        assert tracker.get_ema(TEST_MODEL) == 0.0
        assert TEST_MODEL not in tracker._streaming_count
        assert tracker._waiting_for_first_token[TEST_MODEL] is True


# ===================================================================
# select_config_per_request tests
# ===================================================================

class TestSelectConfigPerRequest:
    def test_stays_on_current_when_tps_ok_and_high_concurrency(self):
        """TPS above threshold but concurrency high → no change."""
        result = select_config_per_request(CPU_4, MIN_TPS_THRESHOLD + 1, 5.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_4"

    def test_scales_up_when_tps_below_threshold(self):
        """Per-request TPS below MIN_TPS_THRESHOLD → scale up."""
        result = select_config_per_request(CPU_4, MIN_TPS_THRESHOLD - 1, 5.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_12"

    def test_scales_up_from_cpu12_when_tps_low(self):
        result = select_config_per_request(CPU_12, MIN_TPS_THRESHOLD - 1, 5.0, CONFIGS_BY_COST)
        assert result.config_id() == "gpu_25"

    def test_scales_up_from_gpu25_when_tps_low(self):
        result = select_config_per_request(GPU_25, MIN_TPS_THRESHOLD - 1, 5.0, CONFIGS_BY_COST)
        assert result.config_id() == "gpu_100"

    def test_stays_on_gpu100_at_max(self):
        """Already on most expensive config, can't scale up further."""
        result = select_config_per_request(GPU_100, MIN_TPS_THRESHOLD - 1, 5.0, CONFIGS_BY_COST)
        assert result.config_id() == "gpu_100"

    def test_scales_down_when_tps_ok_and_low_concurrency(self):
        """TPS above threshold AND concurrency low AND lower config can handle it → scale down."""
        # cpu_4 capacity=32, concurrency=1.0 → 32/1=32 >= 15 (10*1.5) → ok
        result = select_config_per_request(
            CPU_12, MIN_TPS_THRESHOLD + 1, 1.0, CONFIGS_BY_COST,
        )
        assert result.config_id() == "cpu_4"

    def test_no_scale_down_when_lower_config_cant_handle(self):
        """TPS above threshold, concurrency low, but lower config can't handle load → stay."""
        # cpu_12 → cpu_4: capacity=32, concurrency=3.0 → 32/3=10.7 < 15 (10*1.5) → no scale-down
        result = select_config_per_request(
            CPU_12, MIN_TPS_THRESHOLD + 1, 3.0, CONFIGS_BY_COST,
        )
        assert result.config_id() == "cpu_12"

    def test_no_scale_down_when_concurrency_above_threshold(self):
        """TPS above threshold but concurrency above SCALE_DOWN_CONCURRENCY → stay."""
        result = select_config_per_request(
            GPU_100, MIN_TPS_THRESHOLD + 1, SCALE_DOWN_CONCURRENCY + 1, CONFIGS_BY_COST,
        )
        assert result.config_id() == "gpu_100"

    def test_no_scale_down_when_tps_below_threshold(self):
        """TPS below threshold even with low concurrency → don't scale down (scale up instead)."""
        result = select_config_per_request(
            CPU_12, MIN_TPS_THRESHOLD - 1, SCALE_DOWN_CONCURRENCY - 0.5, CONFIGS_BY_COST,
        )
        # Should scale UP, not down
        assert result.config_id() == "gpu_25"

    def test_stays_on_cpu4_at_min(self):
        """Already on cheapest config, can't scale down further."""
        result = select_config_per_request(
            CPU_4, MIN_TPS_THRESHOLD + 1, 1.0, CONFIGS_BY_COST,
        )
        assert result.config_id() == "cpu_4"

    def test_scales_down_from_gpu100(self):
        # gpu_25 capacity=147, concurrency=1.0 → 147/1=147 >= 15 (10*1.5) → ok
        result = select_config_per_request(
            GPU_100, MIN_TPS_THRESHOLD + 1, 1.0, CONFIGS_BY_COST,
        )
        assert result.config_id() == "gpu_25"


# ===================================================================
# CostAwareAutoscaler.check_scaling tests
# ===================================================================

class TestCheckScaling:
    def test_returns_none_during_cooldown(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=300, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        fake_time[0] = 100.0  # within cooldown
        # Low TPS that would trigger scale-up
        assert scaler.check_scaling(
            TEST_MODEL, per_request_tps_ema=3.0, active_requests_ema=5.0,
        ) is None

    def test_returns_config_after_cooldown_scale_up(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=300, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        fake_time[0] = 301.0  # past cooldown
        result = scaler.check_scaling(
            TEST_MODEL, per_request_tps_ema=3.0, active_requests_ema=5.0,
        )
        assert result is not None
        assert result.config_id() == "cpu_12"

    def test_returns_config_after_cooldown_scale_down(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=300, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_12
        scaler.last_scale_time[TEST_MODEL] = 0.0

        fake_time[0] = 301.0
        # concurrency=1.0 so cpu_4 (32 tok/s) / 1 = 32 >= 15 (10*1.5) → scale down ok
        result = scaler.check_scaling(
            TEST_MODEL,
            per_request_tps_ema=MIN_TPS_THRESHOLD + 1,
            active_requests_ema=1.0,
        )
        assert result is not None
        assert result.config_id() == "cpu_4"

    def test_returns_none_when_optimal_equals_current(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=10, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        fake_time[0] = 20.0
        # TPS above threshold, concurrency high → stay on cpu_4
        assert scaler.check_scaling(
            TEST_MODEL,
            per_request_tps_ema=MIN_TPS_THRESHOLD + 1,
            active_requests_ema=SCALE_DOWN_CONCURRENCY + 1,
        ) is None
