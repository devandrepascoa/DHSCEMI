#!/usr/bin/env python3
"""
LLM Benchmarking System using llama-server API
Supports both CUDA and CPU variants with batch size variations
Measures: throughput, prompt processing throughput, token generation throughput, time to first token
"""

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


class LlamaServerBenchmark:
    """Class to handle llama-server benchmarking"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url

        connector = aiohttp.TCPConnector(limit=0)
        self.session = aiohttp.ClientSession(connector=connector)
        self.session.headers.update({"Content-Type": "application/json"})
    
    async def wait_for_server(self, timeout: int = 300) -> bool:
        """Wait for llama-server to be ready"""
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
        """Get model information from the server"""
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
        """Run concurrent completion benchmark"""
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": 1,
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
            # Create progress bar
            pbar = tqdm(total=concurrent_requests, desc=f"Processing {concurrent_requests} concurrent requests")
            
            async def tracked_request():
                result = await single_request()
                pbar.update(1)
                return result
            
            # Run concurrent requests with progress tracking
            tasks = [tracked_request() for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Close progress bar
            pbar.close()
            
            # Filter successful results
            successful_results = [r for r in results if r and not isinstance(r, Exception)]

            if not successful_results:
                return {}
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate aggregate metrics
            total_tokens = 0
            total_prompt_tokens = 0
            total_predicted_tokens = 0
            total_prompt_ms = 0
            total_predicted_ms = 0
            
            for result in successful_results:
                timings = result.get("timings", {})
                
                # Use the actual token counts from the response
                prompt_tokens = timings.get("prompt_n", 0)
                predicted_tokens = timings.get("predicted_n", 0)
                
                total_prompt_tokens += prompt_tokens
                total_predicted_tokens += predicted_tokens
                total_tokens += predicted_tokens
                
                prompt_ms = timings.get("prompt_ms", 0)
                predicted_ms = timings.get("predicted_ms", 0)
                
                total_prompt_ms += prompt_ms
                total_predicted_ms += predicted_ms
            
            # Calculate metrics using proper token counts
            avg_time_per_request = total_time / len(successful_results)
            
            # Global throughput: total tokens generated across all requests per second
            throughput = total_predicted_tokens / total_time if total_time > 0 else 0
            
            # Prompt processing throughput: tokens processed per second
            prompt_processing_throughput = total_prompt_tokens / (total_prompt_ms / 1000.0) if total_prompt_ms > 0 else 0
            
            # Token generation throughput: tokens generated per second
            token_generation_throughput = total_predicted_tokens / (total_predicted_ms / 1000.0) if total_predicted_ms > 0 else 0
            

            metrics = {
                # Core performance metrics
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
                
                # Configuration parameters
                "variant": variant,
                "cpu_cores": cpu_cores,
                "gpu_percentage": gpu_percentage,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "concurrent_requests": concurrent_requests,
                "prompt": prompt,
                "model_name": model_name,

                # Timing details
                "total_prompt_ms": total_prompt_ms,
                "total_predicted_ms": total_predicted_ms,
                
                # Request details
                "request_failure_rate": (concurrent_requests - len(successful_results)) / concurrent_requests if concurrent_requests > 0 else 0
            }
            
            print(json.dumps(metrics, indent=2))
            
            return metrics
            
        except Exception as e:
            import traceback

            # Print stack trace
            traceback.print_exc()
            print(f"Error running concurrent completion benchmark: {e}")
            return {}
    
    async def run_chat_benchmark(self, messages: List[Dict], max_tokens: int = 100, 
                          temperature: float = 0.7) -> Dict:
        """Run a chat completion benchmark"""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(f"{self.server_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                
                result = await response.json()
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                choices = result.get("choices", [])
                
                if choices:
                    message = choices[0].get("message", {}).get("content", "")
                    tokens_generated = len(message.split())
                    
                    # Get token usage information
                    usage = result.get("usage", {})
                    
                    metrics = {
                        "total_time": total_time,
                        "tokens_generated": tokens_generated,
                        "throughput": tokens_generated / total_time if total_time > 0 else 0,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "time_to_first_token": total_time,  # For chat, this is total response time
                        "prompt_processing_throughput": usage.get("prompt_tokens", 0) / total_time if total_time > 0 else 0,
                        "token_generation_throughput": usage.get("completion_tokens", 0) / total_time if total_time > 0 else 0
                    }
                    
                    print(json.dumps(metrics, indent=2))
                    
                    return metrics
                
        except Exception as e:
            print(f"Error running chat benchmark: {e}")
            return {}


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {config_path}")
    return config


def get_models(models_dir):
    """Get list of available model files"""
    models_dir = Path(models_dir)
    models = []
    for ext in ['.gguf', '.bin']:
        models.extend(models_dir.glob(f'*{ext}'))
    return sorted(models)


def verify_docker_images():
    """Verify that required Docker images exist"""
    print("Verifying Docker images...")
    
    # Map our local tags to the actual prebuilt images (full versions)
    images = [
        'ghcr.io/ggml-org/llama.cpp:full',
        'ghcr.io/ggml-org/llama.cpp:full-cuda'
    ]
    
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


def start_llama_server(model_path, variant, config, models_dir, cpu_cores=None, gpu_percentage=None):
    """Start llama-server in a Docker container with unique naming for each config"""

    # Create unique identifier for this configuration
    config_id = f"{model_path.name}_{variant}"
    if cpu_cores is not None:
        config_id += f"_cpu{cpu_cores}"
    if gpu_percentage is not None:
        config_id += f"_gpu{gpu_percentage}"

    container_name = f"llama-server-{config_id}"
    port = 8080 + hash(config_id) % 1000  # Simple port assignment

    config_desc = f"{model_path.name} - {variant}"
    if cpu_cores is not None:
        config_desc += f" - CPU {cpu_cores} cores"
    if gpu_percentage is not None:
        config_desc += f" - GPU {gpu_percentage}%"

    print(f"Starting llama-server: {config_desc}")

    # Select appropriate prebuilt image
    image_map = {
        'cpu': 'ghcr.io/ggml-org/llama.cpp:full',
        'cuda': 'ghcr.io/ggml-org/llama.cpp:full-cuda'
    }
    image = image_map[variant]

    # Prepare Docker command
    cmd = [
        'docker', 'run', '-d', '--rm', '--name', container_name,
        '--privileged',
        '-p', f'{port}:8080',
        '-v', f'{models_dir}:/models',
    ]

    # Add GPU support if needed
    if variant == 'cuda':
        cmd.extend(['--gpus', 'all'])

    # Add the image and server command
    cmd.extend([image, '--server'])

    # Add model parameter
    cmd.extend(['-m', str(Path("/models") / model_path.name)])

    # Add server parameters
    server_params = config.get("server_params", [])
    cmd.extend(server_params)

    # Ensure GPU percentage and CPU cores are mutually exclusive
    if gpu_percentage is not None and cpu_cores is not None:
        raise ValueError("Cannot specify both GPU percentage and CPU cores - they are mutually exclusive")

    # Set thread count based on CPU cores if specified (CPU variant)
    if cpu_cores is not None:
        cmd.extend(['-t', str(cpu_cores)])
    else:
        cmd.extend(['-t', str(config.get("default_threads", 4))])

    # Set GPU layers based on GPU percentage if specified (CUDA variant)
    if gpu_percentage is not None and variant == 'cuda':
        gpu_layers = max(1, int(gpu_percentage))
        cmd.extend(['-ngl', str(gpu_layers)])
    elif variant == 'cuda' and gpu_percentage is None:
        cmd.extend(['-ngl', '0'])  # Default to CPU offload

    return cmd, port, container_name


async def run_benchmark(model_path, variant, config, models_dir, cpu_cores=None, gpu_percentage=None):
    """Run a single benchmark configuration"""
    config_desc = f"{model_path.name} - {variant}"
    if cpu_cores is not None:
        config_desc += f" - CPU {cpu_cores} cores"
    if gpu_percentage is not None:
        config_desc += f" - GPU {gpu_percentage}%"

    print(f"Running benchmark: {config_desc}")

    # Build Docker command to start server
    server_cmd, port, container_name = start_llama_server(model_path, variant, config, models_dir,
                                                         cpu_cores, gpu_percentage)

    # Start server using Docker command
    try:
        print("🐳 Starting llama-server container with command:", " ".join(server_cmd))
        subprocess.run(server_cmd, capture_output=True, text=True)
        print(f"✅ llama-server container started: {container_name}")

        # Wait for server to be ready
        benchmark = LlamaServerBenchmark(server_url=f"http://localhost:{port}")

        if not await benchmark.wait_for_server(timeout=300):
            print("❌ Server failed to start")
            return {}

        # Get model info
        model_info = await benchmark.get_model_info()
        print(f"Model info: {model_info}")

        # Prepare test prompts
        test_prompts = config.get("test_prompts")
        # Test configurations
        concurrent_levels = config.get("concurrent_levels", [1, 4, 16, 32, 128, 512, 2048])
        token_sizes = config.get("token_sizes", [128, 512])

        results = []

        for token_size in token_sizes:
            print(f"\n📊 Testing with {token_size} output tokens...")

            for concurrent_level in concurrent_levels:
                print(f"\n🔄 Testing concurrent requests: {concurrent_level}")

                for prompt in test_prompts:
                    # Test concurrent completion endpoint
                    print(f"\n📝 Testing prompt with {concurrent_level} concurrent requests...")
                    completion_metrics = await benchmark.run_concurrent_completion_benchmark(
                        prompt=prompt,
                        max_tokens=token_size,
                        temperature=config.get("temperature", 0.7),
                        concurrent_requests=concurrent_level,
                        variant=variant,
                        cpu_cores=cpu_cores,
                        gpu_percentage=gpu_percentage,
                        model_name=model_path.name,
                    )

                    if completion_metrics:
                        completion_metrics["endpoint"] = "concurrent_completion"
                        completion_metrics["prompt"] = prompt
                        completion_metrics["variant"] = variant
                        completion_metrics["model"] = model_path.name
                        completion_metrics["cpu_cores"] = cpu_cores
                        completion_metrics["gpu_percentage"] = gpu_percentage
                        completion_metrics["token_size"] = token_size
                        results.append(completion_metrics)
                        print(f"  ✅ Concurrent completion test completed - "
                              f"{completion_metrics.get('successful_requests', 0)}/{concurrent_level} successful, "
                              f"Throughput: {completion_metrics.get('throughput', 0):.2f} tokens/sec")
                    else:
                        print(f"  ❌ Concurrent completion test failed")

        return results

    finally:
        # Stop server container using Docker
        try:
            subprocess.run(['docker', 'stop', container_name],
                         check=False, capture_output=True)
            subprocess.run(['docker', 'rm', container_name],
                         check=False, capture_output=True)
            print(f"✅ llama-server container stopped and removed: {container_name}")
        except Exception as e:
            print(f"Error stopping container: {e}")


async def save_results(results, output_dir="./benchmark_results"):
    """Save benchmark results to CSV and JSON"""
    if not results:
        print("No results to save")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save as JSON
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Print summary
    print("\n📊 Benchmark Summary:")
    print("-" * 50)
    
    for variant in ['cpu', 'cuda']:
        variant_results = [r for r in results if r.get('variant') == variant]
        if variant_results:
            print(f"\n{variant.upper()} Results:")
            
            # Calculate averages
            avg_throughput = statistics.mean([r.get('throughput', 0) for r in variant_results])
            avg_prompt_throughput = statistics.mean([r.get('prompt_processing_throughput', 0) for r in variant_results])
            avg_token_throughput = statistics.mean([r.get('token_generation_throughput', 0) for r in variant_results])
            avg_time_to_first = statistics.mean([r.get('time_to_first_token', 0) for r in variant_results])
            
            print(f"  Average Throughput: {avg_throughput:.2f} tokens/sec")
            print(f"  Average Prompt Processing: {avg_prompt_throughput:.2f} tokens/sec")
            print(f"  Average Token Generation: {avg_token_throughput:.2f} tokens/sec")
            print(f"  Average Time to First Token: {avg_time_to_first:.3f} seconds")


async def run_all_benchmarks(config_path, models_dir, cpu_only=False, gpu_only=False):
    """Run all benchmark configurations"""
    config = load_config(config_path)
    models = get_models(models_dir)
    
    if not models:
        print("❌ No models found in ./models directory")
        print("Please add .gguf or .bin model files to the models directory")
        return
    
    if not verify_docker_images():
        return
    
    print(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n📊 Starting benchmark run: {timestamp}")
    
    all_results = []
    results_count = 0
    
    for model in models:
        print(f"\n📊 Benchmarking model: {model.name}")
        
        # GPU scenarios with increasing GPU percentage
        if not cpu_only:
            gpu_percentages = config.get("gpu_percentages", [25, 50, 75, 100])
            print("\nSCENARIO 1: GPU scenarios with increasing GPU percentage")
            print("-" * 50)
            
            for gpu_pct in gpu_percentages:
                results = await run_benchmark(model, 'cuda', config, models_dir, 
                                              cpu_cores=None, gpu_percentage=gpu_pct)
                if results:
                    all_results.extend(results)
                results_count += len(results) if results else 0
                print(f"  ✅ Completed: {model.name} - CUDA - GPU {gpu_pct}%")
        
        # CPU scenarios with increasing CPU core count
        if not gpu_only:
            cpu_configs = config.get("cpu_configs", [1, 2, 4, 8])
            print("\nSCENARIO 2: CPU scenarios with increasing CPU core count")
            print("-" * 50)
            
            for cpu_cores in cpu_configs:
                results = await run_benchmark(model, 'cpu', config, models_dir, 
                                              cpu_cores=cpu_cores, gpu_percentage=0)
                if results:
                    all_results.extend(results)
                results_count += len(results) if results else 0
                print(f"  ✅ Completed: {model.name} - CPU - {cpu_cores} cores")
    
    print(f"\n🎉 Benchmark run completed!")
    print(f"📊 Total results: {results_count}")
    
    # Save results
    await save_results(all_results)


def check_gpu():
    """Check if GPU is available"""
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
    parser.add_argument("--config-path", default="benchmark_configv2.json", help="Path to configuration file")
    parser.add_argument("--server-url", default="http://localhost:8080", help="URL for llama-server")
    
    args = parser.parse_args()
    
    # Validate mutually exclusive options
    if args.cpu_only and args.gpu_only:
        print("Error: --cpu-only and --gpu-only are mutually exclusive")
        return
    
    await run_all_benchmarks(args.config_path, args.models_dir, 
                            cpu_only=args.cpu_only, gpu_only=args.gpu_only)


if __name__ == "__main__":
    asyncio.run(main())
