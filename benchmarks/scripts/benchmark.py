#!/usr/bin/env python3
import os
import json
import subprocess
import time
import argparse
import asyncio
import aiohttp
import statistics
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import threading
import json
from tqdm import tqdm
import tiktoken


class LlamaServerBenchmark:

    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url

        connector = aiohttp.TCPConnector(limit=0)
        self.session = aiohttp.ClientSession(connector=connector)
        self.session.headers.update({"Content-Type": "application/json"})

    async def wait_for_server(self, timeout: int = 60) -> bool:

        print("Waiting for llama-server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"{self.server_url}/health", timeout=5) as response:
                    if response.status == 200:
                        print("✅ llama-server is ready")
                        return True
            except aiohttp.ClientError:
                pass

            await asyncio.sleep(2)

        print("❌ llama-server failed to start within timeout")
        return False

    async def get_model_info(self) -> Dict:

        try:
            async with self.session.get(f"{self.server_url}/models") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {}

    async def run_concurrent_completion_benchmark(self, prompt: str, max_tokens: int = 100,
                                                  temperature: float = 0.7,
                                                  concurrent_requests: int = 1,
                                                  variant: str = None,
                                                  cpu_cores: int = None,
                                                  gpu_percentage: int = None,
                                                  model_name: str = None) -> Dict:

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": 1,
            "cache_prompt": False,
            "stream": False
        }

        start_time = time.time()

        async def single_request():
            try:
                async with self.session.post(f"{self.server_url}/completion", json=payload, timeout=None) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                print(f"Request failed: {e}")
                return None

        try:

            async def tracked_request():
                result = await single_request()

                return result

            tasks = [tracked_request() for _ in range(concurrent_requests)]
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1200)

            successful_results = [r for r in results if r and not isinstance(r, Exception)]

            print("DATA_DUMP")
            for result in successful_results:
                result_copy = result.copy()

                result_copy["content"] = None
                print(json.dumps(result, indent=4))

            if not successful_results:
                return {}

            end_time = time.time()
            total_time = end_time - start_time

            total_tokens = 0
            total_prompt_tokens = 0
            total_predicted_tokens = 0
            total_prompt_ms = 0
            total_predicted_ms = 0

            for result in successful_results:
                timings = result.get("timings", {})

                prompt_tokens = timings.get("prompt_n", 0)
                predicted_tokens = timings.get("predicted_n", 0)

                total_prompt_tokens += prompt_tokens
                total_predicted_tokens += predicted_tokens
                total_tokens += predicted_tokens

                prompt_ms = timings.get("prompt_ms", 0)
                predicted_ms = timings.get("predicted_ms", 0)

                total_prompt_ms += prompt_ms
                total_predicted_ms += predicted_ms

            avg_time_per_request = total_time / len(successful_results)

            throughput = total_predicted_tokens / total_time if total_time > 0 else 0

            prompt_processing_throughput = total_prompt_tokens / (
                    total_prompt_ms / 1000.0) if total_prompt_ms > 0 else 0

            token_generation_throughput = total_predicted_tokens / (
                    total_predicted_ms / 1000.0) if total_predicted_ms > 0 else 0

            metrics = {

                "total_time": total_time,
                "successful_requests": len(successful_results),
                "total_requests": concurrent_requests,
                "throughput": throughput,
                "avg_time_per_request": avg_time_per_request,
                "prompt_processing_throughput": prompt_processing_throughput,
                "token_generation_throughput": token_generation_throughput,
                "time_to_first_token": total_prompt_ms / 1000.0 / len(successful_results) if successful_results else 0,
                "total_prompt_tokens": total_prompt_tokens,
                "total_predicted_tokens": total_predicted_tokens,
                "prompt_ms_per_token": total_prompt_ms / total_prompt_tokens if total_prompt_tokens > 0 else 0,
                "predicted_ms_per_token": total_predicted_ms / total_predicted_tokens if total_predicted_tokens > 0 else 0,

                "variant": variant,
                "cpu_cores": cpu_cores,
                "gpu_percentage": gpu_percentage,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "concurrent_requests": concurrent_requests,
                "prompt": prompt,
                "model_name": model_name,

                "total_prompt_ms": total_prompt_ms,
                "total_predicted_ms": total_predicted_ms,

                "request_failure_rate": (concurrent_requests - len(
                    successful_results)) / concurrent_requests if concurrent_requests > 0 else 0
            }

            print(json.dumps(metrics, indent=2))

            return metrics

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error running concurrent completion benchmark: {e}")
            return {}


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {config_path}")
    return config


def get_models(models_dir):
    models_dir = Path(models_dir)
    models = []
    for ext in ['.gguf', '.bin']:
        models.extend(models_dir.glob(f'*{ext}'))
    return sorted(models)


def verify_docker_images(cpu_only: bool = True):
    print("Verifying Docker images...")

    images = [
        'ghcr.io/ggml-org/llama.cpp:full',
    ]

    if not cpu_only:
        images.append('ghcr.io/ggml-org/llama.cpp:full-cuda')


    missing_images = []
    for image in images:
        try:
            subprocess.run(['docker', 'image', 'inspect', image],
                           check=True, capture_output=True)
            print(f"✅ Found prebuilt image: {image}")
        except subprocess.CalledProcessError:
            print(f"❌ Missing prebuilt image: {image}")
            missing_images.append(image)

    if missing_images:
        print(f"❌ Missing Docker images: {missing_images}")
        print("Please run: ./scripts/setup.sh to pull the required images")
        return False

    print("✅ All required Docker images are available")
    return True


def get_start_llama_server_cmd(model_path, variant, config, models_dir, cpu_cores=None, gpu_percentage=None, unique_id=None, token_size=None, concurrent_level=None, current_prompt=None):
    config_id = f"{variant}"
    if cpu_cores is not None:
        config_id += f"_cpu{cpu_cores}"
    if gpu_percentage is not None:
        config_id += f"_gpu{gpu_percentage}"
    if unique_id is not None:
        config_id += f"_{unique_id}"

    container_name = f"llama-server-{config_id}"
    port = 8080 + hash(config_id) % 1000

    config_desc = f"{model_path.name} - {variant}"
    if cpu_cores is not None:
        config_desc += f" - CPU {cpu_cores} cores"
    if gpu_percentage is not None:
        config_desc += f" - GPU {gpu_percentage}%"

    print(f"Starting llama-server: {config_desc}")

    image_map = {
        'cpu': 'ghcr.io/ggml-org/llama.cpp:full',
        'cuda': 'ghcr.io/ggml-org/llama.cpp:full-cuda'
    }
    image = image_map[variant]


    # list all files in current dir

    import os

    files = os.listdir()
    print(f"Files in current directory: {files}")

    cmd = [
        'docker', 'run', '-d', '--rm', '--name', container_name,
        '--privileged',
        '-p', f'{port}:8080',
        '-v', f'{models_dir}:/models',
    ]

    if variant == 'cuda':
        cmd.extend(['--gpus', 'all'])

    if cpu_cores is not None:
        cmd.extend(['--cpus', str(cpu_cores)])

    if gpu_percentage is not None and gpu_percentage < 100:
        cmd.extend([
            "-e", f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={gpu_percentage}"
        ])

    cmd.extend([image, '--server'])

    cmd.extend(['-m', str(Path("/models") / model_path.name)])

    server_params = config.get("server_params", [])
    cmd.extend(server_params)

    if gpu_percentage is not None and cpu_cores is not None:
        raise ValueError("Cannot specify both GPU percentage and CPU cores - they are mutually exclusive")

    if cpu_cores is not None:
        cmd.extend(['-t', str(cpu_cores)])
    else:
        cmd.extend(['-t', str(config.get("default_threads", 4))])

    if gpu_percentage is not None and variant == 'cuda':
        gpu_layers = max(1, int(gpu_percentage))
        cmd.extend(['-ngl', str(gpu_layers)])
    elif variant == 'cuda' and gpu_percentage is None:
        cmd.extend(['-ngl', '0'])

    # Use the provided concurrent_level directly
    cmd.extend(['--parallel', str(concurrent_level)])


    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        current_prompt_length = len(tokenizer.encode(current_prompt))
    except Exception as e:
        print(f"Warning: Failed to use tokenizer, falling back to character approximation: {e}")
        current_prompt_length = len(current_prompt) // 4

    min_ctx_per_slot = current_prompt_length + token_size

    ctx_size = int(min_ctx_per_slot * concurrent_level)

    ctx_size = max(ctx_size, 2048)

    cmd.extend(['--ctx-size', str(ctx_size)])

    return cmd, port, container_name


async def run_benchmark_with_samples(model_path, variant, config, models_dir, cpu_cores=None, gpu_percentage=None,
                                     num_samples=1):
    config_desc = f"{model_path.name} - {variant}"
    if cpu_cores is not None:
        config_desc += f" - CPU {cpu_cores} cores"
    if gpu_percentage is not None:
        config_desc += f" - GPU {gpu_percentage}%"

    print(f"Running benchmark with {num_samples} samples: {config_desc}")

    test_prompts = config.get("test_prompts")

    concurrent_levels = config.get("concurrent_levels")
    token_sizes = config.get("token_sizes")

    all_results = []

    for token_size in token_sizes:
        print(f"\n📊 Testing with {token_size} output tokens...")

        for concurrent_level in concurrent_levels:
            print(f"\n🔄 Testing concurrent requests: {concurrent_level}")

            for prompt in test_prompts:

                samples = []

                for sample_num in range(1, num_samples + 1):
                    print(
                        f"\n📝 Sample {sample_num}/{num_samples} - prompt with {concurrent_level} concurrent requests...")

                    # Generate unique identifier for this specific benchmark call
                    import time
                    unique_id = f"{int(time.time() * 1000000)}"  # microsecond timestamp

                    # Start a new container for this specific benchmark call
                    server_cmd, port, container_name = get_start_llama_server_cmd(
                        model_path, variant, config, models_dir,
                        cpu_cores, gpu_percentage, unique_id=unique_id,
                        token_size=token_size, concurrent_level=concurrent_level, current_prompt=prompt
                    )

                    try:
                        print(f"🐳 Starting new container for this benchmark call: {container_name} with cmd: {" ".join(server_cmd)}")
                        subprocess.run(server_cmd, capture_output=True, text=True)

                        # Create benchmark instance for this specific container
                        call_benchmark = LlamaServerBenchmark(server_url=f"http://localhost:{port}")

                        if not await call_benchmark.wait_for_server():
                            print("❌ Server failed to start for this call")
                            continue

                        completion_metrics = await call_benchmark.run_concurrent_completion_benchmark(
                            prompt=prompt,
                            max_tokens=token_size,
                            temperature=config.get("temperature", 0.7),
                            concurrent_requests=concurrent_level,
                            variant=variant,
                            cpu_cores=cpu_cores,
                            gpu_percentage=gpu_percentage,
                            model_name=model_path.name,
                        )

                        # Clean up the container after this call
                        subprocess.run(['docker', 'stop', container_name], capture_output=True, text=True)
                        subprocess.run(['docker', 'rm', container_name], capture_output=True, text=True)
                        print(f"🗑️ Cleaned up container: {container_name}")

                    except Exception as e:
                        print(f"Error in benchmark call: {e}")
                        completion_metrics = {}
                    finally:
                        subprocess.run(['docker', 'stop', container_name], capture_output=True, text=True)
                        subprocess.run(['docker', 'rm', container_name], capture_output=True, text=True)

                    if completion_metrics:
                        samples.append(completion_metrics)
                        print(f"  ✅ Sample {sample_num} completed - "
                              f"{completion_metrics.get('successful_requests', 0)}/{concurrent_level} successful")
                    else:
                        print(f"  ❌ Sample {sample_num} failed")

                if samples:
                    stats_result = calculate_statistics_from_samples(samples, {
                        "endpoint": "concurrent_completion",
                        "prompt": prompt,
                        "variant": variant,
                        "model": model_path.name,
                        "cpu_cores": cpu_cores,
                        "gpu_percentage": gpu_percentage,
                        "token_size": token_size,
                        "concurrent_requests": concurrent_level
                    })
                    all_results.append(stats_result)
                    print(f"stats_result: {stats_result}")
                    print(f"  📊 Statistics calculated from {len(samples)} samples")

    return all_results


def calculate_statistics_from_samples(samples, metadata):
    if not samples:
        return {}

    metrics_to_calculate = [
        "throughput", "prompt_processing_throughput", "token_generation_throughput",
        "time_to_first_token", "total_time", "avg_time_per_request",
        "prompt_ms_per_token", "predicted_ms_per_token", "request_failure_rate"
    ]

    result = metadata.copy()

    for metric in metrics_to_calculate:
        values = [sample.get(metric, 0) for sample in samples if metric in sample]
        if values:
            mean_value = statistics.mean(values)
            stddev_value = statistics.stdev(values) if len(values) > 1 else 0.0

            result[f"{metric}_mean"] = mean_value
            result[f"{metric}_stddev"] = stddev_value
            result[f"{metric}_samples"] = len(values)
            result[f"{metric}_values"] = values
        else:
            result[f"{metric}_mean"] = 0.0
            result[f"{metric}_stddev"] = 0.0
            result[f"{metric}_samples"] = 0
            result[f"{metric}_values"] = []

    first_sample = samples[0]
    copy_fields = [
        "successful_requests", "total_requests", "total_prompt_tokens",
        "total_predicted_tokens", "total_prompt_ms", "total_predicted_ms"
    ]

    for field in copy_fields:
        if field in first_sample:
            result[field] = first_sample[field]

    return result


async def run_all_benchmarks(config_path, models_dir, cpu_only=False, gpu_only=False):
    config = load_config(config_path)
    models = get_models(models_dir)

    for model in models:
        print(f"Model: {model.name}")

    if not models:
        print("❌ No models found in ./models directory")
        print("Please add .gguf or .bin model files to the models directory")
        return

    if not verify_docker_images(cpu_only):
        return

    print(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n📊 Starting benchmark run: {timestamp}")

    all_results = []
    results_count = 0

    for model in models:
        print(f"\n📊 Benchmarking model: {model.name}")

        if not cpu_only:
            gpu_percentages = config.get("gpu_percentages", [25, 50, 75, 100])
            print("\nSCENARIO 1: GPU scenarios with increasing GPU percentage")
            print("-" * 50)

            for gpu_pct in gpu_percentages:
                results = await run_benchmark_with_samples(model, 'cuda', config, models_dir,
                                                           cpu_cores=None, gpu_percentage=gpu_pct)
                if results:
                    all_results.extend(results)
                    results_count += len(results) if results else 0
                    print(f"  ✅ Completed: {model.name} - CUDA - GPU {gpu_pct}%")
                else:
                    print(f"  ❌ Failed: {model.name} - CUDA - GPU {gpu_pct}%")

        if not gpu_only:
            cpu_configs = config.get("cpu_configs", [1, 2, 4, 8])
            print("\nSCENARIO 2: CPU scenarios with increasing CPU core count")
            print("-" * 50)

            for cpu_cores in cpu_configs:
                results = await run_benchmark_with_samples(model, 'cpu', config, models_dir,
                                                           cpu_cores=cpu_cores, gpu_percentage=None)
                if results:
                    all_results.extend(results)
                results_count += len(results) if results else 0
                print(f"  ✅ Completed: {model.name} - CPU - {cpu_cores} cores")

    print(f"\n🎉 Benchmark run completed!")
    print(f"📊 Total results: {results_count}")


def check_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks using llama-server")
    parser.add_argument("--models-dir", default="./models", help="Directory containing model files")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU benchmarks")
    parser.add_argument("--gpu-only", action="store_true", help="Run only GPU benchmarks")
    parser.add_argument("--config-path", default="./benchmarks/benchmark_config.json", help="Path to configuration file")
    parser.add_argument("--server-url", default="http://localhost:8080", help="URL for llama-server")

    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        print("Error: --cpu-only and --gpu-only are mutually exclusive")
        return

    await run_all_benchmarks(args.config_path, args.models_dir,
                             cpu_only=args.cpu_only, gpu_only=args.gpu_only)


if __name__ == "__main__":
    asyncio.run(main())
