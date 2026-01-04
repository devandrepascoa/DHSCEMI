#!/usr/bin/env python3
"""
Dynamic Container Scaling Benchmark

This benchmark tests the dynamic hardware cost optimization system by:
1. Using the 01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf model
2. Gradually increasing workload to trigger container scaling
3. Tracking throughput (requests/second) and total system cost
4. Observing container scaling behavior and cost optimization
"""

import asyncio
import aiohttp
import time
import json
import csv
import threading
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class BenchmarkMetrics:
    timestamp: float
    workload_level: int
    requests_per_second: float
    total_throughput: float
    total_system_cost: float
    active_containers: int
    container_configs: List[str]
    scaling_events: int
    avg_response_time: float
    tokens_per_hour: float
    cost_per_token: float

class DynamicScalingBenchmark:
    def __init__(self,
                 proxy_url: str = "http://localhost:8000",
                 model_name: str = "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M",
                 max_concurrent: int = 100,
                 benchmark_duration: int = 1800):  # 30 minutes
        self.proxy_url = proxy_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.benchmark_duration = benchmark_duration

        # Tracking metrics
        self.metrics_history: List[BenchmarkMetrics] = []
        self.response_times: List[float] = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.scaling_events = 0
        self.last_container_configs = []

        # Workload progression
        self.workload_levels = [
            100,  1000, 10000, 15000, 20000, 50000
        ]  # requests per minute progression

        self.current_workload_index = 0
        self.benchmark_start_time = None
        self.is_running = False

    async def send_chat_request(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Send a chat completion request and measure response time"""
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short explanation of dynamic hardware scaling in AI systems."
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }

        try:
            async with session.post(
                f"{self.proxy_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time

                if response.status == 200:
                    result = await response.json()
                    self.successful_requests += 1
                    self.response_times.append(response_time)

                    # Extract token usage if available
                    usage = result.get("usage", {})
                    total_tokens = usage.get("total_tokens", 150)  # fallback estimate

                    return {
                        "success": True,
                        "response_time": response_time,
                        "tokens": total_tokens,
                        "status": response.status
                    }
                else:
                    self.failed_requests += 1
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status": response.status,
                        "error": await response.text()
                    }

        except Exception as e:
            end_time = time.time()
            self.failed_requests += 1
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }

    async def get_system_metrics(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Fetch current system metrics from the proxy"""
        try:
            # Get workload stats
            async with session.get(f"{self.proxy_url}/v1/workload-stats/{self.model_name}") as response:
                if response.status == 200:
                    workload_stats = await response.json()
                else:
                    workload_stats = {}

            # Get cost analysis
            async with session.get(f"{self.proxy_url}/v1/cost-analysis/{self.model_name}") as response:
                if response.status == 200:
                    cost_analysis = await response.json()
                else:
                    cost_analysis = {}

            # Get general metrics
            async with session.get(f"{self.proxy_url}/v1/metrics") as response:
                if response.status == 200:
                    general_metrics = await response.json()
                else:
                    general_metrics = {}

            return {
                "workload_stats": workload_stats,
                "cost_analysis": cost_analysis,
                "general_metrics": general_metrics
            }

        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {}

    async def workload_generator(self, session: aiohttp.ClientSession):
        """Generate requests according to current workload level"""
        while self.is_running:
            if self.current_workload_index >= len(self.workload_levels):
                break

            current_workload = self.workload_levels[self.current_workload_index]
            requests_per_second = current_workload / 60.0

            if requests_per_second > 0:
                interval = 1.0 / requests_per_second

                # Send request
                await self.send_chat_request(session)

                # Wait for next request
                await asyncio.sleep(interval)
            else:
                await asyncio.sleep(1)

    async def metrics_collector(self, session: aiohttp.ClientSession):
        """Collect system metrics periodically"""
        while self.is_running:
            try:
                current_time = time.time()
                elapsed_time = current_time - self.benchmark_start_time

                # Get system metrics
                system_metrics = await self.get_system_metrics(session)
                workload_stats = system_metrics.get("workload_stats", {})
                cost_analysis = system_metrics.get("cost_analysis", {})

                # Calculate current throughput
                if len(self.response_times) > 0:
                    # Calculate requests per second over last 10 seconds
                    recent_responses = [t for t in self.response_times if current_time - t < 10]
                    requests_per_second = len(recent_responses) / min(10, elapsed_time)
                    avg_response_time = statistics.mean(self.response_times[-100:]) if self.response_times else 0
                else:
                    requests_per_second = 0
                    avg_response_time = 0

                # Extract key metrics
                tokens_per_hour = workload_stats.get("tokens_per_hour", 0)
                current_hardware = workload_stats.get("current_hardware", "cpu")

                # Calculate total system cost
                total_cost = 0
                container_configs = []

                # Use the new cost analysis format with actual running containers
                total_cost = cost_analysis.get("total_hourly_cost", 0)

                if "containers" in cost_analysis:
                    print(f"Cost analysis: {cost_analysis}")
                    for container in cost_analysis["containers"]:
                        config_name = f"{container.get('config_type', 'unknown')}_{container.get('cpu_cores', 0)}cores" if container.get('config_type') == 'cpu' else f"{container.get('config_type', 'unknown')}_{container.get('gpu_percentage', 0)}%"
                        container_configs.append(config_name)

                # Detect scaling events
                if container_configs != self.last_container_configs:
                    self.scaling_events += 1
                    self.last_container_configs = container_configs.copy()
                    print(f"🔄 Scaling event detected! New configs: {container_configs}")

                # Calculate cost per token
                cost_per_token = total_cost / max(tokens_per_hour, 1) if tokens_per_hour > 0 else 0

                # Record metrics
                current_workload = (self.workload_levels[self.current_workload_index]
                                  if self.current_workload_index < len(self.workload_levels) else 0)

                metrics = BenchmarkMetrics(
                    timestamp=current_time,
                    workload_level=current_workload,
                    requests_per_second=requests_per_second,
                    total_throughput=self.successful_requests / max(elapsed_time, 1),
                    total_system_cost=total_cost,
                    active_containers=len(container_configs),
                    container_configs=container_configs,
                    scaling_events=self.scaling_events,
                    avg_response_time=avg_response_time,
                    tokens_per_hour=tokens_per_hour,
                    cost_per_token=cost_per_token
                )

                self.metrics_history.append(metrics)

                # Print progress
                print(f"⏱️  {elapsed_time:.0f}s | "
                      f"Workload: {current_workload} req/min | "
                      f"RPS: {requests_per_second:.2f} | "
                      f"Cost: ${total_cost:.4f}/hour | "
                      f"Configs: {len(container_configs)} | "
                      f"Scaling events: {self.scaling_events}")

            except Exception as e:
                print(f"Error collecting metrics: {e}")
                raise e

            await asyncio.sleep(5)  # Collect metrics every 5 seconds

    def workload_controller(self):
        """Control workload progression over time"""
        workload_step_duration = self.benchmark_duration / len(self.workload_levels)

        while self.is_running:
            time.sleep(workload_step_duration)

            if self.current_workload_index < len(self.workload_levels) - 1:
                self.current_workload_index += 1
                current_workload = self.workload_levels[self.current_workload_index]
                print(f"📈 Increasing workload to {current_workload} requests/minute")
            else:
                print("🏁 Maximum workload reached")
                break

    async def run_benchmark(self):
        """Run the complete benchmark"""
        print("🚀 Starting Dynamic Container Scaling Benchmark")
        print(f"📊 Model: {self.model_name}")
        print(f"🎯 Duration: {self.benchmark_duration} seconds")
        print(f"📈 Workload levels: {len(self.workload_levels)} steps")
        print(f"🔗 Proxy URL: {self.proxy_url}")
        print()

        self.benchmark_start_time = time.time()
        self.is_running = True

        # Start workload controller in separate thread
        controller_thread = threading.Thread(target=self.workload_controller)
        controller_thread.start()

        async with aiohttp.ClientSession() as session:
            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self.metrics_collector(session)),
                *[asyncio.create_task(self.workload_generator(session))
                  for _ in range(min(10, self.max_concurrent))]  # Multiple workers
            ]

            # Wait for benchmark duration or manual stop
            try:
                await asyncio.sleep(self.benchmark_duration)
            except KeyboardInterrupt:
                print("\n⚠️  Benchmark interrupted by user")
            finally:
                self.is_running = False

                # Cancel tasks
                for task in tasks:
                    task.cancel()

                # Wait for tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

                # Stop controller thread
                controller_thread.join(timeout=5)

        print("\n✅ Benchmark completed!")
        self.save_results()
        self.generate_report()

    def save_results(self):
        """Save benchmark results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dynamic_scaling_benchmark_{timestamp}.csv"
        filepath = Path(__file__).parent / filename

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'workload_level', 'requests_per_second', 'total_throughput',
                'total_system_cost', 'active_containers', 'container_configs',
                'scaling_events', 'avg_response_time', 'tokens_per_hour', 'cost_per_token'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for metrics in self.metrics_history:
                writer.writerow({
                    'timestamp': metrics.timestamp,
                    'workload_level': metrics.workload_level,
                    'requests_per_second': metrics.requests_per_second,
                    'total_throughput': metrics.total_throughput,
                    'total_system_cost': metrics.total_system_cost,
                    'active_containers': metrics.active_containers,
                    'container_configs': ';'.join(metrics.container_configs),
                    'scaling_events': metrics.scaling_events,
                    'avg_response_time': metrics.avg_response_time,
                    'tokens_per_hour': metrics.tokens_per_hour,
                    'cost_per_token': metrics.cost_per_token
                })

        print(f"📁 Results saved to: {filepath}")
        return filepath

    def generate_report(self):
        """Generate benchmark report with visualizations"""
        if not self.metrics_history:
            print("No metrics data to generate report")
            return

        # Create DataFrame for analysis
        data = []
        for m in self.metrics_history:
            data.append({
                'elapsed_time': m.timestamp - self.benchmark_start_time,
                'workload_level': m.workload_level,
                'requests_per_second': m.requests_per_second,
                'total_system_cost': m.total_system_cost,
                'active_containers': m.active_containers,
                'scaling_events': m.scaling_events,
                'tokens_per_hour': m.tokens_per_hour,
                'cost_per_token': m.cost_per_token
            })

        df = pd.DataFrame(data)

        # Generate plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dynamic Container Scaling Benchmark Results', fontsize=16)

        # Plot 1: Throughput vs Time
        ax1.plot(df['elapsed_time'], df['requests_per_second'], 'b-', linewidth=2, label='Actual RPS')
        ax1.plot(df['elapsed_time'], df['workload_level']/60, 'r--', alpha=0.7, label='Target RPS')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Requests per Second')
        ax1.set_title('Throughput Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cost vs Time
        ax2.plot(df['elapsed_time'], df['total_system_cost'], 'g-', linewidth=2)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Total System Cost ($/hour)')
        ax2.set_title('System Cost Over Time')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Container Scaling
        ax3.plot(df['elapsed_time'], df['active_containers'], 'orange', linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Active Containers')
        ax3.set_title('Container Scaling Events')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Cost Efficiency
        ax4.plot(df['elapsed_time'], df['cost_per_token'], 'purple', linewidth=2)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Cost per Token ($)')
        ax4.set_title('Cost Efficiency Over Time')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"dynamic_scaling_benchmark_plot_{timestamp}.png"
        plot_filepath = Path(__file__).parent / plot_filename
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"📈 Plots saved to: {plot_filepath}")

        plt.show()

        # Print summary
        print(f"\n📊 BENCHMARK SUMMARY")
        print(f"=" * 50)
        print(f"Duration: {df['elapsed_time'].max():.1f} seconds")
        print(f"Total requests: {self.successful_requests + self.failed_requests}")
        print(f"Successful requests: {self.successful_requests}")
        print(f"Failed requests: {self.failed_requests}")
        print(f"Success rate: {100 * self.successful_requests / max(self.successful_requests + self.failed_requests, 1):.1f}%")
        print(f"Max throughput: {df['requests_per_second'].max():.2f} RPS")
        print(f"Max system cost: ${df['total_system_cost'].max():.4f}/hour")
        print(f"Scaling events: {df['scaling_events'].max()}")
        print(f"Final container count: {df['active_containers'].iloc[-1]}")
        print(f"Best cost per token: ${df['cost_per_token'].min():.6f}")

async def main():
    """Main benchmark execution"""
    benchmark = DynamicScalingBenchmark(
        proxy_url="http://localhost:8000",
        model_name="01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M",
        benchmark_duration=900,  # 15 minutes
        max_concurrent=50
    )

    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main())
