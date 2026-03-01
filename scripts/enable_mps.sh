#!/bin/bash
# Enable NVIDIA MPS daemon for use with Docker containers.
#
# Key requirements for MPS inside Docker:
#   - Mount the pipe directory: -v /tmp/nvidia-mps-pipe:/tmp/nvidia-mps-pipe
#   - Set env: -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-pipe
#   - Share IPC namespace: --ipc=host
#   - Run as host user: -u $(id -u):$(id -g)
#   - Set thread limit: -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<N>

set -euo pipefail

MPS_PIPE_DIR="/tmp/nvidia-mps-pipe"

echo "--- Enabling NVIDIA MPS ---"

# Kill any existing MPS processes and clean stale state
echo "Cleaning stale MPS state..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1
pkill -f nvidia-cuda-mps 2>/dev/null || true
sleep 1
rm -rf "$MPS_PIPE_DIR"/* /tmp/nvidia-mps/* 2>/dev/null || true

# Create pipe directory with open permissions (needed for Docker)
mkdir -p "$MPS_PIPE_DIR"
chmod 777 "$MPS_PIPE_DIR"

# Start MPS daemon with explicit pipe directory
echo "Starting MPS daemon (pipe dir: $MPS_PIPE_DIR)..."
export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
nvidia-cuda-mps-control -d 2>&1 || true
sleep 2

# Verify
if pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
    echo "MPS daemon is active."
else
    echo "ERROR: MPS daemon failed to start."
    exit 1
fi

echo ""
echo "--- MPS Enabled ---"
echo "Docker run flags for MPS containers:"
echo "  --ipc=host \\"
echo "  -u $(id -u):$(id -g) \\"
echo "  -e CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE_DIR \\"
echo "  -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<N> \\"
echo "  -v $MPS_PIPE_DIR:$MPS_PIPE_DIR"
