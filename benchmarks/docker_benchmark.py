#!/usr/bin/env python3
"""
Docker Benchmark Script
Creates containers with increasing CPU cores (1,2,4,8) and GPU partitions,
then runs nvidia-smi and lscpu in each.
"""

import subprocess
import json
import time
import os
from typing import List, Dict, Any


class DockerBenchmark:
    def __init__(self):
        self.cuda_image = "ubuntu:22.04"
        self.cpu_image = "ubuntu:22.04"  # CPU-only image for llama.cpp

    def check_docker(self) -> bool:
        """Check if Docker is available and running"""
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            subprocess.run(["docker", "info"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def check_nvidia_docker(self) -> bool:
        """Check if nvidia-docker runtime is available"""
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .Runtimes}}"],
                capture_output=True, text=True, check=True
            )
            runtimes = json.loads(result.stdout)
            return "nvidia" in str(runtimes)
        except subprocess.CalledProcessError:
            return False

    def create_container(self, container_name: str, cpu_cores: int, gpu_percentage: int, limit_cpu: bool = False,
                         use_gpu: bool = True) -> str:
        """Create a Docker container with specified configuration"""

        # Calculate CPU set - use first N cores
        cpu_set = ",".join(str(i) for i in range(cpu_cores))

        # Choose appropriate image
        image = self.cuda_image if use_gpu else self.cpu_image

        # Build Docker command with proper resource constraints
        cmd = [
            "docker", "run", "-d",
            "--privileged",
            "--name", container_name,
            "--rm",
        ]
        if limit_cpu:
            cmd.extend([
                f"--cpuset-cpus={cpu_set}",  # Pin to specific CPU cores
            ])

        # Add GPU configuration using MPS if enabled
        if use_gpu and gpu_percentage > 0:
            cmd.extend([
                "--gpus", "device=0",
                "-e", f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={gpu_percentage}"
            ])
        elif use_gpu:
            # GPU enabled but 0% (edge case)
            cmd.extend(["--gpus", "0"])

        cmd.extend([image, "sleep", "3600"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            container_id = result.stdout.strip()
            return container_id
        except subprocess.CalledProcessError as e:
            print(f"Error creating container: {e.stderr}")
            return None

    def run_command_in_container(self, container_id: str, command: str) -> Dict[str, Any]:
        """Run a command inside a container and return results"""
        try:
            cmd = ["docker", "exec", container_id] + command.split()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "return_code": -1
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": e.returncode
            }

    def benchmark_gpu_scenario(self, gpu_percentage: int) -> Dict[str, Any]:
        """Benchmark GPU scenario: maximum CPU with increasing GPU percentage"""

        container_name = f"benchmark-gpu-gpu{gpu_percentage}pct"
        print(f"\n=== GPU Scenario: GPU {gpu_percentage}% (MPS) ===")

        # Create container
        container_id = self.create_container(container_name, 0, gpu_percentage, limit_cpu=False, use_gpu=True)

        if not container_id:
            return {
                "scenario": "gpu",
                "gpu_percentage": gpu_percentage,
                "success": False,
                "error": "Failed to create container"
            }

        # Wait for container to be ready
        time.sleep(2)

        # Run commands
        results = {
            "scenario": "gpu",
            "gpu_percentage": gpu_percentage,
            "container_id": container_id,
            "container_name": container_name,
            "success": True,
            "nvidia_smi": self.run_command_in_container(container_id, "nvidia-smi"),
            "lscpu": self.run_command_in_container(container_id, "cat /sys/fs/cgroup/cpuset.cpus.effective")
        }

        self.print_result(results)
        
        # Immediately delete container after running commands
        try:
            subprocess.run(["docker", "stop", container_id], capture_output=True, check=False)
            subprocess.run(["docker", "rm", container_id], capture_output=True, check=False)
            print(f"Deleted container: {container_name}")
        except subprocess.CalledProcessError:
            pass

        return results

    def benchmark_cpu_scenario(self, cpu_cores: int) -> Dict[str, Any]:
        """Benchmark CPU scenario: 0 GPU with increasing CPU core count"""

        container_name = f"benchmark-cpu-cpu{cpu_cores}-gpu0"
        print(f"\n=== CPU Scenario: CPU {cpu_cores} cores, GPU disabled ===")

        # Create container with CPU-only image
        container_id = self.create_container(container_name, cpu_cores, 0, limit_cpu=True, use_gpu=False)
        if not container_id:
            return {
                "scenario": "cpu",
                "cpu_cores": cpu_cores,
                "gpu_percentage": 0,
                "success": False,
                "error": "Failed to create container"
            }

        # Wait for container to be ready
        time.sleep(2)

        # Run commands
        results = {
            "scenario": "cpu",
            "cpu_cores": cpu_cores,
            "gpu_percentage": 0,
            "container_id": container_id,
            "container_name": container_name,
            "success": True,
            "nvidia_smi": self.run_command_in_container(container_id, "nvidia-smi"),
            "lscpu": self.run_command_in_container(container_id, "cat /sys/fs/cgroup/cpuset.cpus.effective")
        }

        self.print_result(results)
        
        # Immediately delete container after running commands
        try:
            subprocess.run(["docker", "stop", container_id], capture_output=True, check=False)
            subprocess.run(["docker", "rm", container_id], capture_output=True, check=False)
            print(f"Deleted container: {container_name}")
        except subprocess.CalledProcessError:
            pass

        return results

    def run_benchmark(self) -> None:
        """Run the complete benchmark suite with two scenarios"""

        print("Starting Docker Benchmark with Two Scenarios...")
        print("=" * 70)

        # Check prerequisites
        if not self.check_docker():
            print("ERROR: Docker is not available or not running")
            return

        if not self.check_nvidia_docker():
            print("WARNING: nvidia-docker runtime not found. GPU tests will likely fail.")

        # Scenario 1: Maximum CPU with increasing GPU percentage
        max_cpu_cores = 8  # Maximum CPU cores
        gpu_percentages = [10, 25, 50, 75, 100]

        print("\nSCENARIO 1: Maximum CPU cores with increasing GPU percentage")
        print("-" * 50)

        for gpu_pct in gpu_percentages:
            result = self.benchmark_gpu_scenario(gpu_pct)

            # Print summary for this test
            if result["success"]:
                print(f"✓ Max CPU {max_cpu_cores}, GPU {gpu_pct}%: Success")
            else:
                print(f"✗ Max CPU {max_cpu_cores}, GPU {gpu_pct}%: {result.get('error', 'Failed')}")

        # Scenario 2: 0 GPU with increasing CPU core count
        cpu_configs = [1, 2, 4, 8, 16]

        print("\nSCENARIO 2: 0 GPU (disabled) with increasing CPU core count")
        print("-" * 50)

        for cpu_cores in cpu_configs:
            result = self.benchmark_cpu_scenario(cpu_cores)

            # Print summary for this test
            if result["success"]:
                print(f"✓ CPU {cpu_cores}, GPU disabled: Success")
            else:
                print(f"✗ CPU {cpu_cores}, GPU disabled: {result.get('error', 'Failed')}")

    def cleanup(self):
        """Clean up any remaining containers - now mostly a no-op since containers are deleted immediately"""
        print("\nCleaning up any remaining containers...")
        # Containers are now deleted immediately after each benchmark run
        # This method is kept for compatibility and to handle any edge cases
        print("No containers to clean up")

    def print_result(self, result: Dict[str, Any]):
        """Print a result"""
        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print("=" * 70)


def main():
    """Main function to run the benchmark"""
    benchmark = DockerBenchmark()

    try:
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
