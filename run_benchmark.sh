#!/bin/bash

set -e

echo "Starting benchmark process..."

echo "1. Running setup..."
./scripts/setup.sh

echo "2. Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Enabling MPS..."
    sudo ./scripts/enable_mps.sh
else
    echo "No NVIDIA GPU found. Skipping MPS enablement."
fi

echo "3. Downloading models..."
#uv run scripts/download_models.py

echo "4. Running benchmark..."
uv run benchmarks/benchmark.py --config benchmarks/benchmark_config_cluster_test_1.json
uv run benchmarks/benchmark.py --config benchmarks/benchmark_config_cluster_test_2.json

echo "Benchmark process completed!"
