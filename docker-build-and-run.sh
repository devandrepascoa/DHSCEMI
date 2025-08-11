#!/bin/bash

# Simple Docker build and run script for ML inference proxy benchmarks

set -e

IMAGE_NAME="ml-inference-proxy"
CONTAINER_NAME="ml-proxy-benchmark"

echo "=== Building Docker Image ==="
docker build -t $IMAGE_NAME .

echo ""
echo "=== Running Benchmark Container ==="
echo "This will run the complete benchmark sequence:"
echo "1. Install UV"
echo "2. Run uv sync"
echo "3. Create models directory"
echo "4. Execute ./run_benchmark.sh"
echo ""

# Run the container with GPU support and privileged mode for MPS/MIG management
docker run --rm \
    --name $CONTAINER_NAME \
    --privileged \
    --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/nvidia-container-toolkit:/usr/bin/nvidia-container-toolkit \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/_models:/app/_models \
    -v $(pwd)/benchmarks:/app/benchmarks \
    $IMAGE_NAME

echo ""
echo "=== Benchmark Complete ==="
