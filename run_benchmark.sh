#!/bin/bash

# Simple benchmark runner script
# Runs setup, enable MPS, download models, then benchmark

set -e  # Exit on any error

echo "Starting benchmark process..."

# Step 1: Run setup
echo "1. Running setup..."
./scripts/setup.sh

# Step 2: Enable MPS
echo "2. Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Enabling MPS..."
    sudo ./scripts/enable_mps.sh
else
    echo "No NVIDIA GPU found. Skipping MPS enablement."
fi

# Step 3: Download models
echo "3. Downloading models..."
uv run scripts/download_models.py

# Step 4: Run benchmark
echo "4. Running benchmark..."
uv run benchmarks/scripts/benchmark.py

echo "Benchmark process completed!"
