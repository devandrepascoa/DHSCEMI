#!/usr/bin/env python3
"""
Load test to demonstrate autoscaling behavior with visualization.
Triggers: scale up (1->4->8) -> stay high -> scale down (8->4->1)

Timeline (with 5-min cooldowns):
- Phase 1: Ramp up to trigger 1->4 (rpm > 0.4)
- Phase 2: Wait for cooldown, then ramp to trigger 4->8 (rpm > 1.5)  
- Phase 3: Maintain high load
- Phase 4: Reduce load to trigger 8->4 (rpm < 0.8)
- Phase 5: Reduce further to trigger 4->1 (rpm < 0.2)
"""
import asyncio
import aiohttp
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROXY_URL = "http://localhost:8000"
MODEL = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
MAX_TOKENS = 50  # Keep requests short for faster testing

# Thresholds from main_simple.py (updated with wider hysteresis)
SCALE_UP_1_TO_4 = 0.4   # rpm
SCALE_UP_4_TO_8 = 1.5   # rpm
SCALE_DOWN_8_TO_4 = 0.8 # rpm (was 1.2)
SCALE_DOWN_4_TO_1 = 0.2 # rpm (was 0.3)
COOLDOWN = 300  # 5 minutes


@dataclass
class MetricsCollector:
    timestamps: List[float] = field(default_factory=list)
    cpu_cores: List[int] = field(default_factory=list)
    rpm: List[float] = field(default_factory=list)
    active_requests: List[int] = field(default_factory=list)
    total_requests: List[int] = field(default_factory=list)
    phases: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def record(self, cores: int, rpm_val: float, active: int, total: int, phase: str):
        elapsed = time.time() - self.start_time
        self.timestamps.append(elapsed)
        self.cpu_cores.append(cores)
        self.rpm.append(rpm_val)
        self.active_requests.append(active)
        self.total_requests.append(total)
        self.phases.append(phase)
    
    def save_visualization(self, filename: str = "load_test_results.html"):
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("CPU Cores Over Time", "Requests Per Minute (5-min avg)", "Active Requests"),
            vertical_spacing=0.08
        )
        
        # Convert timestamps to minutes
        times_min = [t / 60 for t in self.timestamps]
        
        # CPU Cores
        fig.add_trace(
            go.Scatter(x=times_min, y=self.cpu_cores, mode='lines+markers', 
                      name='CPU Cores', line=dict(color='blue', width=2),
                      fill='tozeroy', fillcolor='rgba(0,100,255,0.2)'),
            row=1, col=1
        )
        
        # RPM with thresholds
        fig.add_trace(
            go.Scatter(x=times_min, y=self.rpm, mode='lines+markers',
                      name='RPM (5-min avg)', line=dict(color='green', width=2)),
            row=2, col=1
        )
        # Add threshold lines
        fig.add_hline(y=SCALE_UP_1_TO_4, line_dash="dash", line_color="orange", 
                     annotation_text="Scale up 1→4", row=2, col=1)
        fig.add_hline(y=SCALE_UP_4_TO_8, line_dash="dash", line_color="red",
                     annotation_text="Scale up 4→8", row=2, col=1)
        fig.add_hline(y=SCALE_DOWN_8_TO_4, line_dash="dot", line_color="purple",
                     annotation_text="Scale down 8→4", row=2, col=1)
        fig.add_hline(y=SCALE_DOWN_4_TO_1, line_dash="dot", line_color="brown",
                     annotation_text="Scale down 4→1", row=2, col=1)
        
        # Active Requests
        fig.add_trace(
            go.Scatter(x=times_min, y=self.active_requests, mode='lines+markers',
                      name='Active Requests', line=dict(color='red', width=2),
                      fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Autoscaler Load Test Results",
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
        fig.update_yaxes(title_text="Cores", row=1, col=1)
        fig.update_yaxes(title_text="RPM", row=2, col=1)
        fig.update_yaxes(title_text="Requests", row=3, col=1)
        
        fig.write_html(filename)
        print(f"\nVisualization saved to: {filename}")


metrics = MetricsCollector()
current_phase = "init"


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


async def get_status() -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{PROXY_URL}/status") as resp:
            return await resp.json()


async def send_request(session: aiohttp.ClientSession, request_id: int) -> bool:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello briefly."}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    try:
        async with session.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            if resp.status == 200:
                await resp.json()
                return True
            return False
    except Exception as e:
        log(f"Request {request_id} failed: {e}")
        return False


async def print_status():
    global current_phase
    status = await get_status()
    for name, info in status.get("containers", {}).items():
        cores = int(info['config'].split()[0])
        rpm_val = info['rpm_5min_avg']
        active = info['active_requests']
        total = info['total_requests']
        
        log(f"  Config: {info['config']}, RPM: {rpm_val}, "
            f"Active: {active}, Total: {total}")
        
        # Record metrics for visualization
        metrics.record(cores, rpm_val, active, total, current_phase)


async def wait_for_scale(target_cores: int, timeout: int = 400) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        status = await get_status()
        for name, info in status.get("containers", {}).items():
            if f"{target_cores} cores" in info["config"]:
                return True
        await asyncio.sleep(5)
    return False


async def send_requests_at_rate(target_rpm: float, duration_seconds: int):
    interval = 60 / target_rpm if target_rpm > 0 else float('inf')
    end_time = time.time() + duration_seconds
    request_id = 0
    
    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            start = time.time()
            request_id += 1
            log(f"Sending request {request_id}...")
            success = await send_request(session, request_id)
            log(f"Request {request_id}: {'OK' if success else 'FAILED'}")
            
            await print_status()
            
            # Calculate sleep time to maintain target rate
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0 and time.time() + sleep_time < end_time:
                log(f"Sleeping {sleep_time:.0f}s to maintain {target_rpm} rpm...")
                await asyncio.sleep(sleep_time)


async def main():
    log("=" * 60)
    log("AUTOSCALER LOAD TEST")
    log("=" * 60)
    log(f"Thresholds: 1->4 at >{SCALE_UP_1_TO_4} rpm, 4->8 at >{SCALE_UP_4_TO_8} rpm")
    log(f"            8->4 at <{SCALE_DOWN_8_TO_4} rpm, 4->1 at <{SCALE_DOWN_4_TO_1} rpm")
    log(f"Cooldown: {COOLDOWN}s (5 min)")
    log("")
    
    # Check initial status
    log("Initial status:")
    await print_status()
    log("")
    
    # PHASE 1: Scale up 1 -> 4 cores (need rpm > 0.4)
    current_phase = "Phase 1: 1→4"
    log("=" * 60)
    log("PHASE 1: Triggering scale up 1 -> 4 cores (target rpm: 0.6)")
    log("=" * 60)
    
    # Send at ~0.6 rpm for 6 minutes to exceed 0.4 threshold
    await send_requests_at_rate(target_rpm=0.6, duration_seconds=360)
    
    log("Waiting for scale up to 4 cores...")
    if await wait_for_scale(4):
        log("SUCCESS: Scaled to 4 cores!")
    else:
        log("WARNING: Scale to 4 cores not detected within timeout")
    
    await print_status()
    log("")
    
    # PHASE 2: Scale up 4 -> 8 cores (need rpm > 1.5)
    current_phase = "Phase 2: 4→8"
    log("=" * 60)
    log("PHASE 2: Triggering scale up 4 -> 8 cores (target rpm: 2.0)")
    log("=" * 60)
    
    # Send at ~2.0 rpm for 6 minutes to exceed 1.5 threshold
    await send_requests_at_rate(target_rpm=2.0, duration_seconds=360)
    
    log("Waiting for scale up to 8 cores...")
    if await wait_for_scale(8):
        log("SUCCESS: Scaled to 8 cores!")
    else:
        log("WARNING: Scale to 8 cores not detected within timeout")
    
    await print_status()
    log("")
    
    # PHASE 3: Maintain high load
    current_phase = "Phase 3: High load"
    log("=" * 60)
    log("PHASE 3: Maintaining high load at 8 cores for 3 minutes")
    log("=" * 60)
    
    await send_requests_at_rate(target_rpm=2.0, duration_seconds=180)
    await print_status()
    log("")
    
    # PHASE 4: Scale down 8 -> 4 cores (need rpm < 0.8)
    current_phase = "Phase 4: 8→4"
    log("=" * 60)
    log("PHASE 4: Reducing load for scale down 8 -> 4 cores (target rpm: 0.5)")
    log("=" * 60)
    
    # Send at ~0.5 rpm (below 0.8 threshold) for 6 minutes
    await send_requests_at_rate(target_rpm=0.5, duration_seconds=360)
    
    log("Waiting for scale down to 4 cores...")
    if await wait_for_scale(4):
        log("SUCCESS: Scaled down to 4 cores!")
    else:
        log("WARNING: Scale down to 4 cores not detected within timeout")
    
    await print_status()
    log("")
    
    # PHASE 5: Scale down 4 -> 1 core (need rpm < 0.2)
    current_phase = "Phase 5: 4→1"
    log("=" * 60)
    log("PHASE 5: Reducing load for scale down 4 -> 1 core (target rpm: 0.1)")
    log("=" * 60)
    
    # Send at ~0.1 rpm (below 0.2 threshold) for 6 minutes
    await send_requests_at_rate(target_rpm=0.1, duration_seconds=360)
    
    log("Waiting for scale down to 1 core...")
    if await wait_for_scale(1):
        log("SUCCESS: Scaled down to 1 core!")
    else:
        log("WARNING: Scale down to 1 core not detected within timeout")
    
    await print_status()
    log("")
    
    current_phase = "Complete"
    log("=" * 60)
    log("LOAD TEST COMPLETE")
    log("=" * 60)
    log("Final status:")
    await print_status()
    
    # Save visualization
    metrics.save_visualization("load_test_results.html")
    log(f"Collected {len(metrics.timestamps)} data points")


if __name__ == "__main__":
    asyncio.run(main())
