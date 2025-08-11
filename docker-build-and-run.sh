#!/bin/bash

# Docker-in-Docker build and run script for ML inference proxy benchmarks

set -e

IMAGE_NAME="ml-inference-proxy-dind"
CONTAINER_NAME="ml-proxy-benchmark-dind"

echo "=== Building Docker-in-Docker Image ==="
docker build -t $IMAGE_NAME .

echo ""
echo "=== Running Docker-in-Docker Benchmark Container ==="
echo "This will run the complete benchmark sequence with Docker-in-Docker:"
echo "1. Start Docker daemon inside container"
echo "2. Install UV"
echo "3. Run uv sync"
echo "4. Create models directory"
echo "5. Execute ./run_benchmark.sh"
echo ""

# Run the container with Docker-in-Docker support, GPU access, and all necessary privileges
docker run --rm \
    --name $CONTAINER_NAME \
    --privileged \
    --gpus all \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/_models:/app/_models \
    -v $(pwd)/benchmarks:/app/benchmarks \
    -e DOCKER_TLS_CERTDIR="" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    $IMAGE_NAME

echo ""
echo "=== Docker-in-Docker Benchmark Complete ==="
