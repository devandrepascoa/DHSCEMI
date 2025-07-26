#!/usr/bin/env python3
"""
LLM Benchmarking System using llama.cpp
Supports both CUDA and CPU variants with batch size variations
"""

import os
import json
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


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


def run_benchmark(model_path, variant, config, models_dir, cpu_cores=None, gpu_percentage=None):
    """Run a single benchmark configuration with real-time log streaming"""
    config_desc = f"{model_path.name} - {variant}"
    if cpu_cores is not None:
        config_desc += f" - CPU {cpu_cores} cores"
    if gpu_percentage is not None:
        config_desc += f" - GPU {gpu_percentage}%"

    print(f"Running benchmark: {config_desc}")

    # Select appropriate prebuilt image (full versions)
    image_map = {
        'cpu': 'ghcr.io/ggml-org/llama.cpp:full',
        'cuda': 'ghcr.io/ggml-org/llama.cpp:full-cuda'
    }
    image = image_map[variant]
    print(f"🐳 Using prebuilt image: {image}")

    # Prepare command for prebuilt image
    cmd = ["--bench",
           "-m",
           str(Path("/models") / model_path.name),
           ]
    cmd.extend(config["llama_bench_params"].copy())

    # Set thread count based on CPU cores if specified
    if cpu_cores is not None:
        cmd.extend(["-t", str(cpu_cores)])
    else:
        cmd.extend(["-t", str(config["default_threads"])])  # Default

    # Set GPU layers based on GPU percentage if specified
    if gpu_percentage is not None and variant == 'cuda':
        # Map GPU percentage to GPU layers (0-99)
        gpu_layers = max(1, int(gpu_percentage))  # Ensure at least 1 layer
        cmd.extend(["-ngl", str(gpu_layers)])
    elif variant == 'cpu' or gpu_percentage is None:
        cmd.extend(["-ngl", "0"])  # No GPU layers for CPU or no GPU percentage

    abs_models_dir = Path(models_dir).resolve()
    print(f"📂 Models directory (absolute): {abs_models_dir}")

    # Build docker run command
    docker_cmd = [
        'docker', 'run', '--rm',
        '-v', f'{abs_models_dir}:/models:ro',
        '--privileged',
    ]

    # Add GPU support for CUDA variant
    if variant == 'cuda':
        docker_cmd.extend(['--gpus', 'all'])

    docker_cmd.extend([image])
    docker_cmd.extend(cmd)

    try:
        print(f"🐳 Starting container with command: {' '.join(cmd)}")
        # Print docker command
        print("📊 Docker command:")
        print("=" * 50)
        print(" ".join(docker_cmd))
        print("=" * 50)
        print("📊 Container output:")
        print("=" * 50)

        # Run container and stream output
        process = subprocess.run(docker_cmd, capture_output=False, text=True)

        print(f"✅ Benchmark completed: {config_desc}")
        print(f"Exit code: {process.returncode}")

    except Exception as e:
        print(f"Error running benchmark: {e}")


def run_all_benchmarks(config_path, models_dir, cpu_only=False, gpu_only=False):
    """Run all benchmark configurations with real-time log streaming"""
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

    results_count = 0

    for model in models:
        print(f"\n📊 Benchmarking model: {model.name}")

        # Scenario 1: GPU scenarios with increasing GPU percentage (CPU not limited)
        if not cpu_only:
            gpu_percentages = config["gpu_percentages"]
            print("\nSCENARIO 1: GPU scenarios with increasing GPU percentage")
            print("-" * 50)

            for gpu_pct in gpu_percentages:
                run_benchmark(model, 'cuda', config, models_dir, cpu_cores=None, gpu_percentage=gpu_pct)
                results_count += 1
                print(f"  ✅ Completed: {model.name} - CUDA - GPU {gpu_pct}%")

        # Scenario 2: CPU scenarios with increasing CPU core count (GPU disabled)
        if not gpu_only:
            cpu_configs = config["cpu_configs"]
            print("\nSCENARIO 2: CPU scenarios with increasing CPU core count")
            print("-" * 50)

            for cpu_cores in cpu_configs:
                run_benchmark(model, 'cpu', config, models_dir, cpu_cores=cpu_cores, gpu_percentage=0)
                results_count += 1
                print(f"  ✅ Completed: {model.name} - CPU - {cpu_cores} cores")

    print(f"\n🎉 Benchmark run completed!")
    print(f"📊 Total results: {results_count}")


def check_gpu():
    """Check if GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--models-dir", default="./models", help="Directory containing model files")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU benchmarks")
    parser.add_argument("--gpu-only", action="store_true", help="Run only GPU benchmarks")
    parser.add_argument("--config-path", default="benchmark_config.json", help="Path to configuration file")

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.cpu_only and args.gpu_only:
        print("Error: --cpu-only and --gpu-only are mutually exclusive")
        return

    run_all_benchmarks(args.config_path, args.models_dir, cpu_only=args.cpu_only, gpu_only=args.gpu_only)


if __name__ == "__main__":
    main()
