#!/bin/bash

set -e

# Cleanup function to ensure MPS is disabled
cleanup() {
    echo "Running cleanup..."
    if [ -f "./scripts/disable_mps.sh" ]; then
        ./scripts/disable_mps.sh
    else
        echo "Warning: disable_mps.sh not found, skipping MPS cleanup"
    fi
}

# Set up signal handlers to ensure cleanup runs on exit/termination
trap cleanup EXIT
trap 'echo "Received SIGINT, cleaning up..."; cleanup; exit 130' INT
trap 'echo "Received SIGTERM, cleaning up..."; cleanup; exit 143' TERM

echo "Starting benchmark process..."

echo "1. Running setup..."
./scripts/setup.sh

echo "2. Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Enabling MPS..."
    ./scripts/enable_mps.sh
else
    echo "No NVIDIA GPU found. Skipping MPS enablement."
fi

echo "3. Downloading models..."
uv run scripts/download_models.py

echo "4. Running benchmark..."
uv run benchmarks/benchmark.py --config benchmarks/benchmark_config_cluster_test_1.json
uv run benchmarks/benchmark.py --config benchmarks/benchmark_config_cluster_test_2.json

echo "Benchmark process completed!"
# Note: cleanup() will be called automatically via EXIT trap
