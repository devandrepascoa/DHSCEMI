#!/bin/bash
# Disable NVIDIA MPS daemon and clean up state.

set -euo pipefail

MPS_PIPE_DIR="/tmp/nvidia-mps-pipe"

echo "--- Disabling NVIDIA MPS ---"

# Send quit command
echo "Stopping MPS daemon..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 2

# Force kill any remaining processes
if pgrep -f "nvidia-cuda-mps" >/dev/null; then
    echo "Force killing remaining MPS processes..."
    pkill -9 -f "nvidia-cuda-mps" 2>/dev/null || true
    sleep 1
fi

# Clean stale state
echo "Cleaning MPS state..."
rm -rf "$MPS_PIPE_DIR"/* /tmp/nvidia-mps/* 2>/dev/null || true

# Verify
if pgrep -f "nvidia-cuda-mps" >/dev/null; then
    echo "WARNING: MPS processes still running."
    ps aux | grep mps | grep -v grep
else
    echo "MPS fully stopped."
fi

echo "--- MPS Disabled ---"
