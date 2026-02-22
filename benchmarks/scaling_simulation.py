#!/usr/bin/env python3
"""
Simulate the scaling benchmark using /metrics-based throughput thresholds.

Scaling logic:
  - Scale UP:   throughput_ema >= SCALE_UP_MULT * measured_throughput[current_config]
  - Scale DOWN: throughput_ema <= SCALE_DOWN_MULT * measured_throughput[cheaper_config]

Usage:
    uv run python benchmarks/scaling_simulation.py
"""
from __future__ import annotations

import random
import sys
sys.path.insert(0, ".")

from main_cost_aware import HardwareConfig

# ---------------------------------------------------------------------------
# Hardware configs — ordered by cost (cheapest to most expensive)
# ---------------------------------------------------------------------------
CONFIGS = [
    HardwareConfig(cpu_cores=4,  memory="8g",  hourly_cost=0.05),
    HardwareConfig(cpu_cores=12, memory="8g",  hourly_cost=0.12),
    HardwareConfig(cpu_cores=2,  memory="8g",  gpu_percentage=25,  hourly_cost=0.50),
    HardwareConfig(cpu_cores=2,  memory="16g", gpu_percentage=100, hourly_cost=4.00),
]
CONFIGS_BY_COST = sorted(CONFIGS, key=lambda c: c.hourly_cost)

MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
COOLDOWN = 300

# ---------------------------------------------------------------------------
# Metrics-based scaling parameters
# ---------------------------------------------------------------------------
SCALE_UP_MULT = 0.8    # scale up at 80% of current capacity
SCALE_DOWN_MULT = 0.3  # scale down when throughput <= 30% of cheaper config
EMA_ALPHA = 2.0 / (240 + 1)  # ~4min EMA window
SCALING_CHECK_INTERVAL = 10

# Measured aggregate throughput per config (from benchmarks)
MEASURED_THROUGHPUT = {
    "cpu_4":   32.0,
    "cpu_12":  47.0,
    "gpu_25":  147.0,
    "gpu_100": 1064.0,
}

# Single-request throughput (batch=1) per config — for simulation only
SINGLE_REQUEST_TPS = {
    "cpu_4":   9.0,
    "cpu_12":  15.4,
    "gpu_25":  13.3,
    "gpu_100": 152.9,
}

TOKENS_PER_REQUEST = 140

# ---------------------------------------------------------------------------
# Workload phases
# ---------------------------------------------------------------------------
PHASES = [
    ("low load",       900,  1,   3),
    ("medium load",    900,  4,  15),
    ("high load",      900,  8,   0),
    ("peak load",      900, 30,   0),
    ("sustain gpu",    600, 30,   0),
    ("ramp-down 1",    900,  4,  43),
    ("ramp-down 2",    900,  4,  15),
    ("ramp-down 3",    900,  1,   3),
    ("low load",       600,  1,   3),
]

EXPECTED_SEQUENCE = ["cpu_4", "cpu_12", "gpu_25", "gpu_100", "gpu_25", "cpu_12", "cpu_4"]


# ---------------------------------------------------------------------------
# Metrics-based scaling decision
# ---------------------------------------------------------------------------

def select_config(
    current_config: HardwareConfig,
    throughput_ema: float,
) -> HardwareConfig:
    """Select config based on throughput EMA vs measured capacity thresholds.

    Scale UP:   throughput_ema >= SCALE_UP_MULT * capacity[current]
    Scale DOWN: throughput_ema <= SCALE_DOWN_MULT * capacity[cheaper_config]
    """
    current_id = current_config.config_id()
    current_idx = next(
        i for i, c in enumerate(CONFIGS_BY_COST) if c.config_id() == current_id
    )
    current_capacity = MEASURED_THROUGHPUT[current_id]

    # Check scale UP
    if throughput_ema >= SCALE_UP_MULT * current_capacity:
        if current_idx + 1 < len(CONFIGS_BY_COST):
            return CONFIGS_BY_COST[current_idx + 1]

    # Check scale DOWN: can a cheaper config handle the current throughput?
    if current_idx > 0:
        cheaper = CONFIGS_BY_COST[current_idx - 1]
        cheaper_capacity = MEASURED_THROUGHPUT[cheaper.config_id()]
        if throughput_ema <= SCALE_DOWN_MULT * cheaper_capacity:
            return cheaper

    return current_config


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    seed: int = 42,
    duration_jitter: float = 0.0,
    token_jitter: float = 0.0,
    rpm_jitter: float = 0.0,
    verbose: bool = False,
) -> tuple:
    """Run one simulation modeling the server's /metrics-based scaling.

    Models:
    - Workers sending requests, each taking TOKENS/single_tps seconds
    - Aggregate throughput = per_slot_tps * active_slots (like /metrics)
    - EMA of aggregate throughput
    - Scale up/down based on EMA vs measured capacity thresholds

    Returns (config_sequence, scaling_events).
    """
    rng = random.Random(seed)

    current_config = CONFIGS_BY_COST[0]
    last_scale_time = 0.0

    throughput_ema = 0.0
    last_ema_time = 0.0

    scaling_events = []
    total_elapsed = 0.0

    for phase_name, duration, concurrency, rpm in PHASES:
        effective_rpm = rpm
        if rpm_jitter > 0 and rpm > 0:
            effective_rpm = rpm * (1.0 + rng.uniform(-rpm_jitter, rpm_jitter))
            effective_rpm = max(1.0, effective_rpm)

        if concurrency > 0 and effective_rpm > 0:
            worker_interval = 60.0 * concurrency / effective_rpm
        else:
            worker_interval = 0

        workers = [{"next_start": total_elapsed, "busy_until": 0.0}
                   for _ in range(concurrency)]

        t = total_elapsed
        end_t = total_elapsed + duration

        while t < end_t:
            config_id = current_config.config_id()
            single_tps = SINGLE_REQUEST_TPS.get(config_id, 9.0)

            # Check for completed requests, start new ones
            active_count = 0
            for w in workers:
                if w["busy_until"] > 0 and t >= w["busy_until"]:
                    w["busy_until"] = 0.0

                if w["busy_until"] == 0 and t >= w["next_start"]:
                    tokens = TOKENS_PER_REQUEST
                    if token_jitter > 0:
                        tokens = int(tokens * (1.0 + rng.uniform(-token_jitter, token_jitter)))
                        tokens = max(10, tokens)
                    base_dur = tokens / single_tps
                    if duration_jitter > 0:
                        base_dur *= (1.0 + rng.uniform(-duration_jitter, duration_jitter))
                    w["busy_until"] = t + base_dur
                    if worker_interval > 0:
                        w["next_start"] = t + worker_interval
                    else:
                        w["next_start"] = t

                if w["busy_until"] > t:
                    active_count += 1

            # Compute aggregate throughput (like /metrics)
            aggregate_tps = single_tps * active_count if active_count > 0 else 0.0

            # Update EMA every 1s tick
            dt = t - last_ema_time
            if dt > 0:
                decay = (1.0 - EMA_ALPHA) ** dt
                throughput_ema = throughput_ema * decay + (1.0 - decay) * aggregate_tps
                last_ema_time = t

            # Scaling check every SCALING_CHECK_INTERVAL seconds
            if int(t) % SCALING_CHECK_INTERVAL == 0 and t > total_elapsed:
                if t - last_scale_time >= COOLDOWN:
                    optimal = select_config(current_config, throughput_ema)
                    if optimal.config_id() != current_config.config_id():
                        old_id = current_config.config_id()
                        current_config = optimal
                        last_scale_time = t
                        action = "SCALE %s -> %s" % (old_id, optimal.config_id())
                        scaling_events.append((t, phase_name, throughput_ema, action))

                if verbose and int(t) % 30 == 0:
                    cap = MEASURED_THROUGHPUT.get(config_id, 0)
                    pct = (throughput_ema / cap * 100) if cap > 0 else 0
                    print(
                        "  %5.1fm %-14s %-8s ema=%6.1f cap=%6.0f (%4.1f%%)"
                        % (t / 60, phase_name, config_id, throughput_ema, cap, pct)
                    )

            t += 1.0

        total_elapsed = end_t

    configs_seen = [CONFIGS_BY_COST[0].config_id()]
    for _, _, _, action in scaling_events:
        new_cfg = action.split("->")[-1].strip()
        configs_seen.append(new_cfg)

    return configs_seen, scaling_events


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("Baseline (no noise)", dict(seed=42)),
    ("Seed 123", dict(seed=123)),
    ("Seed 999", dict(seed=999)),
    ("Seed 7", dict(seed=7)),
    ("Duration jitter +/-20%", dict(seed=42, duration_jitter=0.20)),
    ("Duration jitter +/-30%", dict(seed=42, duration_jitter=0.30)),
    ("Duration jitter +/-40%", dict(seed=42, duration_jitter=0.40)),
    ("Token jitter +/-20%", dict(seed=42, token_jitter=0.20)),
    ("Token jitter +/-30%", dict(seed=42, token_jitter=0.30)),
    ("RPM jitter +/-15%", dict(seed=42, rpm_jitter=0.15)),
    ("RPM jitter +/-25%", dict(seed=42, rpm_jitter=0.25)),
    ("All noise low (dur20/tok15/rpm10)",
     dict(seed=42, duration_jitter=0.20, token_jitter=0.15, rpm_jitter=0.10)),
    ("All noise medium (dur30/tok20/rpm15)",
     dict(seed=42, duration_jitter=0.30, token_jitter=0.20, rpm_jitter=0.15)),
    ("All noise high (dur40/tok30/rpm25)",
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
    print("SCALING SIMULATION — /metrics THROUGHPUT THRESHOLD MODEL")
    print("Configs: %s" % " -> ".join(c.config_id() for c in CONFIGS_BY_COST))
    print("Measured throughput: %s" % ", ".join(
        "%s=%s" % (k, v) for k, v in MEASURED_THROUGHPUT.items()
    ))
    print("Scale UP at %.0f%% of current capacity" % (SCALE_UP_MULT * 100))
    print("Scale DOWN when throughput <= %.0f%% of cheaper capacity" % (SCALE_DOWN_MULT * 100))
    print("Expected sequence: %s" % " -> ".join(EXPECTED_SEQUENCE))
    print("=" * 72)

    # Run baseline verbose first
    print()
    print("--- Baseline verbose ---")
    seq, events = simulate(seed=42, verbose=True)
    ok = seq == EXPECTED_SEQUENCE
    print("  Result: %s  %s" % ("PASS" if ok else "FAIL", " -> ".join(seq)))
    for t, phase, ema, action in events:
        cap = MEASURED_THROUGHPUT.get(action.split("->")[0].strip().replace("SCALE ", ""), 0)
        print("    %.0fm [%s] ema=%.1f %s" % (t / 60, phase, ema, action))
    print()

    passed = 0
    failed = 0
    failures = []

    for name, kwargs in SCENARIOS:
        seq, events = simulate(**kwargs)
        ok = seq == EXPECTED_SEQUENCE
        status = "PASS" if ok else "FAIL"

        events_str = " | ".join(
            "%.0fm:%s" % (t / 60, a.split("->")[-1].strip())
            for t, _, _, a in events
        )

        print("  %s %-50s %s" % (status, name, " -> ".join(seq)))
        if events_str:
            print("    Events: %s" % events_str)

        if ok:
            passed += 1
        else:
            failed += 1
            failures.append(name)

    print()
    print("Named scenarios: %d/%d passed" % (passed, passed + failed))
    if failures:
        print("Failures:")
        for f in failures:
            print("  - %s" % f)

    # Monte Carlo: realistic noise
    print()
    print("-" * 72)
    print("MONTE CARLO: 100 runs with REALISTIC noise (dur+/-20%, tok+/-15%)")
    print("-" * 72)
    mc_pass = 0
    mc_fail_seeds = []
    for i in range(100):
        seed = 10000 + i
        seq, _ = simulate(seed=seed, duration_jitter=0.20, token_jitter=0.15)
        if seq == EXPECTED_SEQUENCE:
            mc_pass += 1
        else:
            mc_fail_seeds.append((seed, seq))

    print("  Passed: %d/100 (%d%%)" % (mc_pass, mc_pass))
    if mc_fail_seeds:
        print("  Failed seeds:")
        for seed, seq in mc_fail_seeds[:10]:
            print("    seed=%d: %s" % (seed, " -> ".join(seq)))
        if len(mc_fail_seeds) > 10:
            print("    ... and %d more" % (len(mc_fail_seeds) - 10))

    # Monte Carlo: moderate noise
    print()
    print("-" * 72)
    print("MONTE CARLO: 100 runs with MODERATE noise (dur+/-30%, tok+/-20%, rpm+/-10%)")
    print("-" * 72)
    mc2_pass = 0
    mc2_fail_seeds = []
    for i in range(100):
        seed = 20000 + i
        seq, _ = simulate(
            seed=seed, duration_jitter=0.30, token_jitter=0.20, rpm_jitter=0.10,
        )
        if seq == EXPECTED_SEQUENCE:
            mc2_pass += 1
        else:
            mc2_fail_seeds.append((seed, seq))

    print("  Passed: %d/100 (%d%%)" % (mc2_pass, mc2_pass))
    if mc2_fail_seeds:
        print("  Failed seeds:")
        for seed, seq in mc2_fail_seeds[:10]:
            print("    seed=%d: %s" % (seed, " -> ".join(seq)))
        if len(mc2_fail_seeds) > 10:
            print("    ... and %d more" % (len(mc2_fail_seeds) - 10))

    # Monte Carlo: extreme noise
    print()
    print("-" * 72)
    print("MONTE CARLO: 100 runs with EXTREME noise (dur+/-40%, tok+/-30%, rpm+/-25%)")
    print("-" * 72)
    mc3_pass = 0
    mc3_fail_seeds = []
    for i in range(100):
        seed = 30000 + i
        seq, _ = simulate(
            seed=seed, duration_jitter=0.40, token_jitter=0.30, rpm_jitter=0.25,
        )
        if seq == EXPECTED_SEQUENCE:
            mc3_pass += 1
        else:
            mc3_fail_seeds.append((seed, seq))

    print("  Passed: %d/100 (%d%%)" % (mc3_pass, mc3_pass))
    if mc3_fail_seeds:
        print("  Failed seeds:")
        for seed, seq in mc3_fail_seeds[:10]:
            print("    seed=%d: %s" % (seed, " -> ".join(seq)))
        if len(mc3_fail_seeds) > 10:
            print("    ... and %d more" % (len(mc3_fail_seeds) - 10))

    # Overall
    total_pass = passed + mc_pass + mc2_pass + mc3_pass
    total_run = passed + failed + 300
    print()
    print("OVERALL: %d/%d simulations achieved perfect staircase" % (total_pass, total_run))
    print("  Named scenarios: %d/%d" % (passed, passed + failed))
    print("  Realistic noise: %d/100" % mc_pass)
    print("  Moderate noise:  %d/100" % mc2_pass)
    print("  Extreme noise:   %d/100" % mc3_pass)

    return failed <= 3 and mc_pass >= 95 and mc2_pass >= 85


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
