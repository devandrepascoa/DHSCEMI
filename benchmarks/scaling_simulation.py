#!/usr/bin/env python3
"""
Simulate the scaling benchmark using the real DemandTracker (EMA)
and CostAwareAutoscaler with a fake clock. No Docker, no inference —
just models the demand curve and scaling decisions to validate parameters
before running the real benchmark.

Runs multiple scenarios with varying noise levels, request duration
jitter, and token count variance to ensure the staircase pattern
(cpu_4 → gpu_25 → gpu_100 → gpu_25 → cpu_4) is robust.

Usage:
    uv run python benchmarks/scaling_simulation.py
"""
from __future__ import annotations

import random
import sys
sys.path.insert(0, ".")

from main_cost_aware import (
    DemandTracker,
    HardwareConfig,
    CostAwareAutoscaler,
    DEFAULT_THROUGHPUT,
)

# ---------------------------------------------------------------------------
# Hardware configs: cpu_4, cpu_12, gpu_25, gpu_100
# Measured throughput from benchmarks (--parallel 32, max_tokens=256)
# ---------------------------------------------------------------------------
# Hourly costs chosen so cost-per-token ordering is:
#   cpu_4 < cpu_12 < gpu_25 < gpu_100
# This ensures the autoscaler picks the cheapest viable config at each demand
# level, producing the staircase:
#   cpu_4 → cpu_12 → gpu_25 → gpu_100 → gpu_25 → cpu_12 → cpu_4
#
# cost/tok: cpu_4=0.43μ$, cpu_12=0.71μ$, gpu_25=0.94μ$, gpu_100=1.04μ$
CONFIGS = [
    HardwareConfig(cpu_cores=4,  memory="8g",  hourly_cost=0.05),
    HardwareConfig(cpu_cores=12, memory="8g",  hourly_cost=0.12),
    HardwareConfig(cpu_cores=2,  memory="8g",  gpu_percentage=25,  hourly_cost=0.50),
    HardwareConfig(cpu_cores=2,  memory="16g", gpu_percentage=100, hourly_cost=4.00),
]

MEASURED_THROUGHPUT = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}

for k, v in MEASURED_THROUGHPUT.items():
    DEFAULT_THROUGHPUT[k] = v

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
COOLDOWN = 300        # 5 min cooldown between scaling events
DEMAND_WINDOW = 180   # 3 min EMA window
TOKENS_PER_REQUEST = 140
HEADROOM = 0.25       # 25% headroom multiplier to widen threshold gaps

# ---------------------------------------------------------------------------
# Workload phases: (name, duration_s, concurrency, rpm)
#
# Each phase is long enough (~15 min) for the 3-min EMA to fully converge
# and the 5-min cooldown to expire before the next scaling decision.
#
# Single-request throughput (batch=1):
#   cpu_4=9.0, cpu_12=15.4, gpu_25=13.3, gpu_100=152.9 tok/s
# So N saturated workers on config X → N × single_tps tok/s demand.
#
# Thresholds (max aggregate throughput):
#   cpu_4=32, cpu_12=47, gpu_25=147, gpu_100=1064
#
# rpm=0 means saturated (workers fire as fast as they finish).
#
# Key challenge: gpu_100 is so fast (152.9 tok/s per request) that even
# 1 saturated worker produces demand > 147. Ramp-down phases use RPM
# limiting to control demand precisely.
#
# RPM-limited demand: rpm × TOKENS_PER_REQUEST / 60 = rpm × 2.33 tok/s
#   For demand ~100 tok/s: rpm ≈ 43
#   For demand ~35 tok/s:  rpm ≈ 15
#   For demand ~7 tok/s:   rpm ≈ 3
#
# With HEADROOM=0.25 + stickiness (asymmetric hysteresis):
#   Scale UP when demand × (1+headroom) exceeds current throughput
#   Stay on current config as long as raw demand < current throughput
#   Effective thresholds for scale-up:
#     cpu_4:  32/1.25 = 25.6 tok/s → triggers cpu_12
#     cpu_12: 47/1.25 = 37.6 tok/s → triggers gpu_25
#     gpu_25: 147/1.25 = 117.6 tok/s → triggers gpu_100
#   Scale-down only when demand drops enough for a cheaper config to be
#   viable WITH headroom (i.e., cheaper config throughput >= demand × 1.25)
# ---------------------------------------------------------------------------
PHASES = [
    ("low load",       900,  1,   3),    # ~7 tok/s EMA, stays on cpu_4
    ("medium load",    900,  4,  15),    # ~35 tok/s (rpm-limited), triggers cpu_12
    ("high load",      900,  8,   0),    # ~123 tok/s on cpu_12→gpu_25, stays gpu_25
    ("peak load",      900, 30,   0),    # ~400 tok/s on gpu_25, triggers gpu_100
    ("sustain gpu",    600, 30,   0),    # sustain on gpu_100
    ("ramp-down 1",    900,  4,  43),    # ~100 tok/s (rpm-limited), triggers gpu_25
    ("ramp-down 2",    900,  4,  15),    # ~35 tok/s (rpm-limited), triggers cpu_12
    ("ramp-down 3",    900,  1,   3),    # ~7 tok/s, triggers cpu_4
    ("low load",       600,  1,   3),    # settle on cpu_4
]

# Single-request duration = TOKENS_PER_REQUEST / single_request_throughput (batch=1)
BASE_DURATION = {
    "cpu_4":   15.6,   # 140 / 9.0 tok/s
    "cpu_12":  9.1,    # 140 / 15.4 tok/s
    "gpu_25":  10.5,   # 140 / 13.3 tok/s
    "gpu_100": 0.9,    # 140 / 152.9 tok/s
}

EXPECTED_SEQUENCE = ["cpu_4", "cpu_12", "gpu_25", "gpu_100", "gpu_25", "cpu_12", "cpu_4"]


def simulate(
    seed: int = 42,
    duration_jitter: float = 0.0,
    token_jitter: float = 0.0,
    rpm_jitter: float = 0.0,
    verbose: bool = False,
) -> tuple[list[str], list[tuple]]:
    """Run one simulation.

    Args:
        seed: Random seed for reproducibility.
        duration_jitter: Fraction of base duration to vary (0.0 = none, 0.3 = ±30%).
        token_jitter: Fraction of token count to vary (0.0 = none, 0.2 = ±20%).
        rpm_jitter: Fraction of rpm to vary per-phase (0.0 = none, 0.15 = ±15%).
        verbose: Print per-sample output.

    Returns:
        (config_sequence, scaling_events)
    """
    rng = random.Random(seed)
    fake_time = [0.0]
    clock = lambda: fake_time[0]

    tracker = DemandTracker(window_seconds=DEMAND_WINDOW, clock=clock)
    scaler = CostAwareAutoscaler(
        configs=CONFIGS, cooldown_seconds=COOLDOWN, clock=clock,
        headroom=HEADROOM,
    )
    scaler.demand_tracker = tracker

    current_config = CONFIGS[0]
    scaler.current_config[MODEL] = current_config
    scaler.last_scale_time[MODEL] = 0.0

    scaling_events = []
    total_elapsed = 0.0

    for phase_name, duration, concurrency, rpm in PHASES:
        effective_rpm = rpm
        if rpm_jitter > 0 and rpm > 0:
            effective_rpm = rpm * (1.0 + rng.uniform(-rpm_jitter, rpm_jitter))
            effective_rpm = max(1.0, effective_rpm)

        # rpm=0 means saturated: workers fire as fast as they finish (no spacing)
        if concurrency > 0 and effective_rpm > 0:
            worker_interval = 60.0 * concurrency / effective_rpm
        else:
            worker_interval = 0  # no delay between requests

        workers = [{"next_start": total_elapsed, "busy_until": 0}
                   for _ in range(concurrency)]

        t = total_elapsed
        end_t = total_elapsed + duration

        while t < end_t:
            # Check for completed requests and record their tokens at current time
            fake_time[0] = t
            for w in workers:
                if w["busy_until"] > 0 and t >= w["busy_until"]:
                    # Request completed — record tokens at current simulation time
                    tokens = TOKENS_PER_REQUEST
                    if token_jitter > 0:
                        tokens = int(tokens * (1.0 + rng.uniform(-token_jitter, token_jitter)))
                        tokens = max(10, tokens)
                    tracker.record_tokens(MODEL, tokens)
                    w["busy_until"] = 0  # mark as idle

            # Start new requests for idle workers
            for w in workers:
                if w["busy_until"] == 0 and t >= w["next_start"]:
                    config_id = current_config.config_id()
                    base_dur = BASE_DURATION.get(config_id, 10.0)
                    if duration_jitter > 0:
                        dur = base_dur * (1.0 + rng.uniform(-duration_jitter, duration_jitter))
                    else:
                        dur = base_dur
                    w["busy_until"] = t + dur
                    w["next_start"] = t + worker_interval

            if int(t) % 5 == 0:
                fake_time[0] = t
                demand = tracker.get_demand(MODEL)

                action = ""
                new_config = scaler.check_scaling(MODEL)
                if new_config and new_config.config_id() != current_config.config_id():
                    old_id = current_config.config_id()
                    current_config = new_config
                    scaler.current_config[MODEL] = new_config
                    scaler.last_scale_time[MODEL] = t
                    action = f"SCALE {old_id} → {new_config.config_id()}"
                    scaling_events.append((t, phase_name, demand, action))

                if verbose and int(t) % 30 == 0:
                    print(f"  {t/60:5.1f}m {phase_name:<14} {current_config.config_id():<8} "
                          f"demand={demand:6.1f} {action}")

            t += 1.0

        total_elapsed = end_t

    configs_seen = [CONFIGS[0].config_id()]
    for _, _, _, action in scaling_events:
        new_cfg = action.split("→")[-1].strip()
        configs_seen.append(new_cfg)

    return configs_seen, scaling_events


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("Baseline (no noise)", dict(seed=42)),
    ("Seed 123", dict(seed=123)),
    ("Seed 999", dict(seed=999)),
    ("Seed 7", dict(seed=7)),
    ("Duration jitter ±20%", dict(seed=42, duration_jitter=0.20)),
    ("Duration jitter ±30%", dict(seed=42, duration_jitter=0.30)),
    ("Duration jitter ±40%", dict(seed=42, duration_jitter=0.40)),
    ("Token jitter ±20%", dict(seed=42, token_jitter=0.20)),
    ("Token jitter ±30%", dict(seed=42, token_jitter=0.30)),
    ("RPM jitter ±15%", dict(seed=42, rpm_jitter=0.15)),
    ("RPM jitter ±25%", dict(seed=42, rpm_jitter=0.25)),
    ("All noise low (dur±20%, tok±15%, rpm±10%)",
     dict(seed=42, duration_jitter=0.20, token_jitter=0.15, rpm_jitter=0.10)),
    ("All noise medium (dur±30%, tok±20%, rpm±15%)",
     dict(seed=42, duration_jitter=0.30, token_jitter=0.20, rpm_jitter=0.15)),
    ("All noise high (dur±40%, tok±30%, rpm±25%)",
     dict(seed=42, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
    ("All noise high seed=123",
     dict(seed=123, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
    ("All noise high seed=999",
     dict(seed=999, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
    ("All noise high seed=7",
     dict(seed=7, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
    ("All noise high seed=314",
     dict(seed=314, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
    ("All noise high seed=2025",
     dict(seed=2025, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
    ("All noise high seed=55555",
     dict(seed=55555, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25)),
]


def main():
    print("=" * 72)
    print("SCALING SIMULATION — ROBUSTNESS TEST")
    print(f"Configs: {' → '.join(c.config_id() for c in CONFIGS)}")
    print(f"Throughput: {', '.join(f'{k}={v}' for k, v in MEASURED_THROUGHPUT.items())} tok/s")
    print(f"Expected sequence: {' → '.join(EXPECTED_SEQUENCE)}")
    print("=" * 72)

    passed = 0
    failed = 0
    failures = []

    for name, kwargs in SCENARIOS:
        seq, events = simulate(**kwargs)
        ok = seq == EXPECTED_SEQUENCE
        status = "✓" if ok else "✗"

        events_str = " | ".join(
            f"{t/60:.0f}m:{a.split('→')[-1].strip()}"
            for t, _, _, a in events
        )

        print(f"  {status} {name:<50} {' → '.join(seq)}")
        if events_str:
            print(f"    Events: {events_str}")

        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(name)

    print()
    print(f"Results: {passed}/{passed + failed} passed")
    if failures:
        print(f"Failures:")
        for f in failures:
            print(f"  - {f}")

    # Monte Carlo: realistic noise
    print()
    print("-" * 72)
    print("MONTE CARLO: 100 runs with REALISTIC noise (dur±20%, tok±15%)")
    print("-" * 72)
    mc_pass = 0
    mc_fail_seeds = []
    for i in range(100):
        seed = 10000 + i
        seq, _ = simulate(
            seed=seed, duration_jitter=0.20, token_jitter=0.15, rpm_jitter=0.0
        )
        if seq == EXPECTED_SEQUENCE:
            mc_pass += 1
        else:
            mc_fail_seeds.append((seed, seq))

    print(f"  Passed: {mc_pass}/100 ({mc_pass}%)")
    if mc_fail_seeds:
        print(f"  Failed seeds:")
        for seed, seq in mc_fail_seeds[:10]:
            print(f"    seed={seed}: {' → '.join(seq)}")
        if len(mc_fail_seeds) > 10:
            print(f"    ... and {len(mc_fail_seeds) - 10} more")

    # Monte Carlo: moderate noise
    print()
    print("-" * 72)
    print("MONTE CARLO: 100 runs with MODERATE noise (dur±30%, tok±20%, rpm±10%)")
    print("-" * 72)
    mc2_pass = 0
    mc2_fail_seeds = []
    for i in range(100):
        seed = 20000 + i
        seq, _ = simulate(
            seed=seed, duration_jitter=0.30, token_jitter=0.20, rpm_jitter=0.10
        )
        if seq == EXPECTED_SEQUENCE:
            mc2_pass += 1
        else:
            mc2_fail_seeds.append((seed, seq))

    print(f"  Passed: {mc2_pass}/100 ({mc2_pass}%)")
    if mc2_fail_seeds:
        print(f"  Failed seeds:")
        for seed, seq in mc2_fail_seeds[:10]:
            print(f"    seed={seed}: {' → '.join(seq)}")
        if len(mc2_fail_seeds) > 10:
            print(f"    ... and {len(mc2_fail_seeds) - 10} more")

    # Monte Carlo: extreme noise
    print()
    print("-" * 72)
    print("MONTE CARLO: 100 runs with EXTREME noise (dur±40%, tok±30%, rpm±25%)")
    print("-" * 72)
    mc3_pass = 0
    mc3_fail_seeds = []
    for i in range(100):
        seed = 30000 + i
        seq, _ = simulate(
            seed=seed, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25
        )
        if seq == EXPECTED_SEQUENCE:
            mc3_pass += 1
        else:
            mc3_fail_seeds.append((seed, seq))

    print(f"  Passed: {mc3_pass}/100 ({mc3_pass}%)")
    if mc3_fail_seeds:
        print(f"  Failed seeds:")
        for seed, seq in mc3_fail_seeds[:10]:
            print(f"    seed={seed}: {' → '.join(seq)}")
        if len(mc3_fail_seeds) > 10:
            print(f"    ... and {len(mc3_fail_seeds) - 10} more")

    # Overall verdict
    total_pass = passed + mc_pass + mc2_pass + mc3_pass
    total_run = passed + failed + 300
    print()
    print(f"OVERALL: {total_pass}/{total_run} simulations achieved perfect staircase")
    print(f"  Named scenarios: {passed}/{passed + failed}")
    print(f"  Realistic noise: {mc_pass}/100")
    print(f"  Moderate noise:  {mc2_pass}/100")
    print(f"  Extreme noise:   {mc3_pass}/100")

    return failed <= 3 and mc_pass >= 95 and mc2_pass >= 85


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
