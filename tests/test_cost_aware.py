"""
Tests for cost-aware autoscaler (plain pytest, mocked inference).
No Docker containers or real model inference needed.
DemandTracker and CostAwareAutoscaler use injectable clock for deterministic time control.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from main_cost_aware import (
    HardwareConfig,
    DemandTracker,
    CostAwareAutoscaler,
    DEFAULT_THROUGHPUT,
    get_cost_per_token,
    get_throughput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reusable test configs (matching HARDWARE_CONFIGS in main_cost_aware.py)
CPU_4 = HardwareConfig(cpu_cores=4, memory="8g", hourly_cost=0.40)
CPU_8 = HardwareConfig(cpu_cores=8, memory="16g", hourly_cost=0.80)
CPU_12 = HardwareConfig(cpu_cores=12, memory="24g", hourly_cost=1.20)
GPU_50 = HardwareConfig(cpu_cores=2, memory="8g", gpu_percentage=50, hourly_cost=1.00)
GPU_100 = HardwareConfig(cpu_cores=2, memory="16g", gpu_percentage=100, hourly_cost=2.00)

ALL_CONFIGS = [CPU_4, CPU_8, CPU_12, GPU_50, GPU_100]

TEST_MODEL = "test-model"


# ===================================================================
# Task 6: Unit Tests
# ===================================================================


class TestHardwareConfigImage:
    """6.1 – Property 1: Docker image selection based on GPU presence."""

    def test_cpu_config_returns_cpu_image(self):
        """CPU configs (no gpu_percentage) return the CPU image."""
        for cfg in [CPU_4, CPU_8, CPU_12]:
            assert cfg.image == "ghcr.io/ggml-org/llama.cpp:full"

    def test_gpu_config_returns_cuda_image(self):
        """GPU configs (gpu_percentage set) return the CUDA image."""
        for cfg in [GPU_50, GPU_100]:
            assert cfg.image == "ghcr.io/ggml-org/llama.cpp:full-cuda"

    def test_container_type_cpu(self):
        assert CPU_4.container_type == "cpu"

    def test_container_type_gpu(self):
        assert GPU_50.container_type == "gpu"


class TestCostPerToken:
    """6.2 – Property 2: cost_per_token = hourly_cost / (throughput * 3600)."""

    def test_cpu_4_cost_per_token(self):
        expected = 0.40 / (12.0 * 3600)
        assert get_cost_per_token(TEST_MODEL, CPU_4) == pytest.approx(expected)

    def test_cpu_8_cost_per_token(self):
        expected = 0.80 / (18.0 * 3600)
        assert get_cost_per_token(TEST_MODEL, CPU_8) == pytest.approx(expected)

    def test_gpu_100_cost_per_token(self):
        expected = 2.00 / (100.0 * 3600)
        assert get_cost_per_token(TEST_MODEL, GPU_100) == pytest.approx(expected)

    def test_all_configs_match_formula(self):
        """Verify formula for every config in the standard set."""
        for cfg in ALL_CONFIGS:
            throughput = get_throughput(TEST_MODEL, cfg)
            expected = cfg.hourly_cost / (throughput * 3600)
            assert get_cost_per_token(TEST_MODEL, cfg) == pytest.approx(expected)


class TestDemandTracker:
    """6.3 – DemandTracker uses an exponential moving average (EMA).

    With window_seconds=W, alpha = 2/(W+1).
    record_tokens adds alpha * token_count to the decayed EMA.
    get_demand decays the EMA to the current time.
    """

    def test_demand_increases_after_recording(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        tracker = DemandTracker(window_seconds=60, clock=clock)

        assert tracker.get_demand(TEST_MODEL) == 0.0
        tracker.record_tokens(TEST_MODEL, 120)
        demand = tracker.get_demand(TEST_MODEL)
        assert demand > 0.0

    def test_demand_grows_with_more_events(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        tracker = DemandTracker(window_seconds=60, clock=clock)

        tracker.record_tokens(TEST_MODEL, 100)
        d1 = tracker.get_demand(TEST_MODEL)

        fake_time[0] = 10.0
        tracker.record_tokens(TEST_MODEL, 200)
        d2 = tracker.get_demand(TEST_MODEL)
        assert d2 > d1

    def test_demand_decays_over_time(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        tracker = DemandTracker(window_seconds=60, clock=clock)

        tracker.record_tokens(TEST_MODEL, 100)
        d_now = tracker.get_demand(TEST_MODEL)

        fake_time[0] = 120.0  # well past the EMA span
        d_later = tracker.get_demand(TEST_MODEL)
        assert d_later < d_now * 0.1  # should have decayed significantly

    def test_demand_returns_zero_with_no_events(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        tracker = DemandTracker(window_seconds=60, clock=clock)
        assert tracker.get_demand(TEST_MODEL) == 0.0

    def test_ema_alpha_matches_window(self):
        """alpha = 2 / (window_seconds + 1)."""
        tracker = DemandTracker(window_seconds=60)
        assert tracker.alpha == pytest.approx(2.0 / 61)


class TestSelectOptimalConfig:
    """6.4 – Property 4: cheapest viable config for various demand levels.

    With default pricing/throughput the cost_per_token ordering is:
        gpu_50 = gpu_100 < cpu_1 < cpu_4 < cpu_8
    So when GPU configs are viable they are always preferred.
    We test both with the full config set AND with a CPU-only subset
    to verify the selection logic across different scenarios.
    """

    def _make_autoscaler(self, configs=None):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        return CostAwareAutoscaler(
            configs=configs or ALL_CONFIGS, cooldown_seconds=10, clock=clock,
        )

    def test_low_demand_selects_cheapest_cost_per_token(self):
        """At demand=1 tok/s all configs are viable; gpu_50 has lowest cost_per_token."""
        scaler = self._make_autoscaler()
        result = scaler.select_optimal_config(TEST_MODEL, demand=1.0)
        # gpu_50 and gpu_100 tie on cost_per_token; min() picks gpu_50 (first in list)
        assert result.config_id() == "gpu_50"

    def test_cpu_only_low_demand_selects_cpu_4(self):
        """With CPU-only configs at demand=1, cpu_4 is cheapest viable (lowest cost_per_token)."""
        cpu_configs = [CPU_4, CPU_8, CPU_12]
        scaler = self._make_autoscaler(configs=cpu_configs)
        result = scaler.select_optimal_config(TEST_MODEL, demand=1.0)
        assert result.config_id() == "cpu_4"

    def test_cpu_only_medium_demand_selects_cpu_8(self):
        """With CPU-only configs at demand=13, cpu_4 can't handle it; cpu_8 is cheapest viable."""
        cpu_configs = [CPU_4, CPU_8, CPU_12]
        scaler = self._make_autoscaler(configs=cpu_configs)
        result = scaler.select_optimal_config(TEST_MODEL, demand=13.0)
        assert result.config_id() == "cpu_8"

    def test_high_demand_selects_gpu(self):
        """At demand=20 tok/s, only gpu_50 and gpu_100 are viable; gpu_50 is cheaper per token."""
        scaler = self._make_autoscaler()
        result = scaler.select_optimal_config(TEST_MODEL, demand=20.0)
        assert result.config_id() == "gpu_50"

    def test_very_high_demand_selects_gpu_100(self):
        """At demand=60 tok/s, only gpu_100 is viable."""
        scaler = self._make_autoscaler()
        result = scaler.select_optimal_config(TEST_MODEL, demand=60.0)
        assert result.config_id() == "gpu_100"

    def test_exceeds_all_configs_falls_back_to_highest_throughput(self):
        """At demand=200 tok/s, no config is viable → fall back to gpu_100 (highest throughput)."""
        scaler = self._make_autoscaler()
        result = scaler.select_optimal_config(TEST_MODEL, demand=200.0)
        assert result.config_id() == "gpu_100"

    def test_zero_demand_selects_cheapest_cost_per_token(self):
        """At demand=0, all configs are viable; cheapest cost_per_token wins (gpu_50)."""
        scaler = self._make_autoscaler()
        result = scaler.select_optimal_config(TEST_MODEL, demand=0.0)
        assert result.config_id() == "gpu_50"


class TestCheckScalingCooldown:
    """6.5 – Property 5: check_scaling() returns None during cooldown."""

    def test_returns_none_during_cooldown(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        scaler = CostAwareAutoscaler(configs=ALL_CONFIGS, cooldown_seconds=10, clock=clock)

        # Set current config and last_scale_time to simulate a recent scaling event
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        # Record high demand so optimal != current
        scaler.demand_tracker.record_tokens(TEST_MODEL, 6000)

        # Still within cooldown (time=5, cooldown=10)
        fake_time[0] = 5.0
        assert scaler.check_scaling(TEST_MODEL) is None

    def test_returns_config_after_cooldown(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        scaler = CostAwareAutoscaler(configs=ALL_CONFIGS, cooldown_seconds=10, clock=clock)

        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        # Record high demand so optimal != current
        scaler.demand_tracker.record_tokens(TEST_MODEL, 6000)

        # After cooldown (time=10, cooldown=10)
        fake_time[0] = 10.0
        result = scaler.check_scaling(TEST_MODEL)
        assert result is not None
        assert result.config_id() != "cpu_4"

    def test_returns_none_when_optimal_equals_current(self):
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        scaler = CostAwareAutoscaler(configs=ALL_CONFIGS, cooldown_seconds=10, clock=clock)

        # At zero demand, optimal is gpu_50 (lowest cost_per_token)
        scaler.current_config[TEST_MODEL] = GPU_50
        scaler.last_scale_time[TEST_MODEL] = 0.0

        # No demand → optimal is gpu_50, same as current → None
        fake_time[0] = 20.0  # well past cooldown
        assert scaler.check_scaling(TEST_MODEL) is None


class TestGetThroughputFallback:
    """6.6 – get_throughput() falls back to DEFAULT_THROUGHPUT for unknown models."""

    def test_unknown_model_uses_defaults(self):
        """An unknown model should still get default throughput values."""
        for cfg in ALL_CONFIGS:
            expected = DEFAULT_THROUGHPUT[cfg.config_id()]
            assert get_throughput("totally-unknown-model", cfg) == expected

    def test_unknown_config_id_returns_1(self):
        """A config whose config_id is not in DEFAULT_THROUGHPUT returns 1.0."""
        weird_cfg = HardwareConfig(cpu_cores=99, memory="1g", hourly_cost=0.01)
        assert get_throughput(TEST_MODEL, weird_cfg) == 1.0


# ===================================================================
# Task 7: Integration Tests (mocked containers)
# ===================================================================


class TestAutoscalerIntegration:
    """Integration tests exercising DemandTracker + CostAwareAutoscaler together."""

    def _make_system(self, cooldown: float = 10.0, window: int = 60):
        """Create an autoscaler with a fake clock for deterministic testing."""
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        scaler = CostAwareAutoscaler(
            configs=ALL_CONFIGS, cooldown_seconds=cooldown, clock=clock,
        )
        return scaler, fake_time

    # 7.1 ---------------------------------------------------------------
    def test_selects_cheapest_config_at_low_demand(self):
        """At low demand the autoscaler should pick the cheapest cost_per_token config.

        With default pricing, gpu_50 has the lowest cost_per_token even at low demand.
        We also verify with CPU-only configs that cpu_1 is selected.
        """
        scaler, fake_time = self._make_system()

        # Record a small number of tokens
        scaler.demand_tracker.record_tokens(TEST_MODEL, 60)  # 60/60 = 1 tok/s
        demand = scaler.demand_tracker.get_demand(TEST_MODEL)
        result = scaler.select_optimal_config(TEST_MODEL, demand)
        # gpu_50 has lowest cost_per_token among all viable configs
        assert result.config_id() == "gpu_50"

    def test_selects_cpu_4_at_low_demand_cpu_only(self):
        """With CPU-only configs, low demand selects cpu_4 (cheapest per token)."""
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        cpu_configs = [CPU_4, CPU_8, CPU_12]
        scaler = CostAwareAutoscaler(
            configs=cpu_configs, cooldown_seconds=10, clock=clock,
        )
        scaler.demand_tracker.record_tokens(TEST_MODEL, 60)
        demand = scaler.demand_tracker.get_demand(TEST_MODEL)
        result = scaler.select_optimal_config(TEST_MODEL, demand)
        assert result.config_id() == "cpu_4"

    # 7.2 ---------------------------------------------------------------
    def test_scales_up_when_demand_increases(self):
        """When demand exceeds cpu_4 capacity, autoscaler should recommend a higher config."""
        scaler, fake_time = self._make_system()

        # Set initial state: running on cpu_4
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        # Simulate high token load — many events to build up EMA
        for i in range(6):
            fake_time[0] = float(i * 5)
            scaler.demand_tracker.record_tokens(TEST_MODEL, 100)

        # Advance past cooldown
        fake_time[0] = 30.0
        new_config = scaler.check_scaling(TEST_MODEL)
        assert new_config is not None
        # Should pick something bigger than cpu_4
        throughput = get_throughput(TEST_MODEL, new_config)
        assert throughput > get_throughput(TEST_MODEL, CPU_4)

    # 7.3 ---------------------------------------------------------------
    def test_scales_down_when_demand_drops(self):
        """After demand drops, autoscaler should recommend scaling back to a cheaper config.

        Uses CPU-only configs so the cost_per_token ordering is cpu_4 < cpu_8 < cpu_12,
        making scale-down behavior clear.
        """
        fake_time = [0.0]
        clock = lambda: fake_time[0]
        cpu_configs = [CPU_4, CPU_8, CPU_12]
        scaler = CostAwareAutoscaler(
            configs=cpu_configs, cooldown_seconds=10, clock=clock,
        )

        # Start on cpu_12
        scaler.current_config[TEST_MODEL] = CPU_12
        scaler.last_scale_time[TEST_MODEL] = 0.0

        # Record very low demand
        scaler.demand_tracker.record_tokens(TEST_MODEL, 30)

        # Advance past cooldown
        fake_time[0] = 15.0
        new_config = scaler.check_scaling(TEST_MODEL)
        assert new_config is not None
        assert new_config.config_id() == "cpu_4"

    # 7.4 ---------------------------------------------------------------
    def test_cooldown_prevents_rapid_oscillation(self):
        """Even if demand changes, scaling should not happen during cooldown."""
        scaler, fake_time = self._make_system(cooldown=10.0)

        # Initial state: cpu_4, just scaled at t=0
        scaler.current_config[TEST_MODEL] = CPU_4
        scaler.last_scale_time[TEST_MODEL] = 0.0

        # High demand that would normally trigger scale-up
        scaler.demand_tracker.record_tokens(TEST_MODEL, 6000)

        # Check at t=3 → within cooldown → None
        fake_time[0] = 3.0
        assert scaler.check_scaling(TEST_MODEL) is None

        # Check at t=7 → still within cooldown → None
        fake_time[0] = 7.0
        assert scaler.check_scaling(TEST_MODEL) is None

        # Check at t=10 → cooldown elapsed → should recommend scale-up
        fake_time[0] = 10.0
        result = scaler.check_scaling(TEST_MODEL)
        assert result is not None
        assert result.config_id() != "cpu_4"
