"""
Tests for cost-aware autoscaler (plain pytest, mocked inference).
No Docker containers or real model inference needed.
ThroughputTracker and CostAwareAutoscaler use injectable clock for deterministic time control.
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
    SCALE_UP_MULT,
    SCALE_DOWN_MULT,
    get_cost_per_token,
    select_config,
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
# select_config tests (threshold-based scaling)
# ===================================================================

class TestSelectConfig:
    def test_stays_on_current_when_within_thresholds(self):
        """EMA between scale-down and scale-up thresholds → no change."""
        result = select_config(CPU_4, 15.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_4"

    def test_scales_up_at_80_pct_capacity(self):
        """EMA >= 80% of cpu_4 capacity (32*0.8=25.6) → scale to cpu_12."""
        result = select_config(CPU_4, 26.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_12"

    def test_scales_down_when_underusing(self):
        """EMA <= 30% of cpu_12 capacity AND <= 75% of cpu_4 capacity → scale down."""
        # cpu_12 capacity=47, 30% = 14.1; cpu_4 capacity=32, 75% = 24
        result = select_config(CPU_12, 10.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_4"

    def test_no_scale_down_if_cheaper_cant_handle(self):
        """EMA low for current but too high for cheaper → stay."""
        # cpu_12 capacity=47, 30% = 14.1; cpu_4 capacity=32, 75% = 24
        # EMA=14.0 is below 30% of cpu_12 but also below 75% of cpu_4 → scale down
        # EMA=25.0 is above 75% of cpu_4 (24) → don't scale down
        result = select_config(CPU_12, 25.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_12"

    def test_scales_up_from_gpu_25_to_gpu_100(self):
        """EMA >= 80% of gpu_25 capacity (147*0.8=117.6) → gpu_100."""
        result = select_config(GPU_25, 120.0, CONFIGS_BY_COST)
        assert result.config_id() == "gpu_100"

    def test_stays_on_gpu_100_at_max(self):
        """Already on most expensive config, can't scale up further."""
        result = select_config(GPU_100, 900.0, CONFIGS_BY_COST)
        assert result.config_id() == "gpu_100"

    def test_stays_on_cpu_4_at_min(self):
        """Already on cheapest config, can't scale down further."""
        result = select_config(CPU_4, 0.0, CONFIGS_BY_COST)
        assert result.config_id() == "cpu_4"


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
        # High EMA that would trigger scale-up
        scaler.throughput_tracker.update_ema(TEST_MODEL, 30.0, now=0.0)

        fake_time[0] = 100.0  # within cooldown
        assert scaler.check_scaling(TEST_MODEL) is None

    def test_returns_config_after_cooldown(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=300, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0
        scaler.throughput_tracker.update_ema(TEST_MODEL, 30.0, now=0.0)

        fake_time[0] = 301.0  # past cooldown
        result = scaler.check_scaling(TEST_MODEL)
        assert result is not None
        assert result.config_id() == "cpu_12"

    def test_returns_none_when_waiting_for_first_token(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=10, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0
        scaler.throughput_tracker._waiting_for_first_token[TEST_MODEL] = True

        fake_time[0] = 20.0
        assert scaler.check_scaling(TEST_MODEL) is None

    def test_returns_none_when_optimal_equals_current(self):
        fake_time = [0.0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=10, clock=lambda: fake_time[0],
        )
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0
        # EMA in the middle of cpu_4 range → no scaling
        scaler.throughput_tracker.update_ema(TEST_MODEL, 15.0, now=0.0)

        fake_time[0] = 20.0
        assert scaler.check_scaling(TEST_MODEL) is None
