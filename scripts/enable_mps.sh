#!/bin/bash


get_compute_mode() {
    nvidia-smi -q -d COMPUTE | grep "Compute Mode" | awk '{print $NF}' | head -n 1
}

is_mps_daemon_running() {
    pgrep -x "nvidia-cuda-mps" >/dev/null
    return $?
}

echo "--- Enabling NVIDIA MPS ---"

num_gpus=$(nvidia-smi -L | wc -l)
if [ "$num_gpus" -ne 1 ]; then
    echo "Error: This script is for single GPU machines. Found $num_gpus GPUs."
    echo "Please adjust the script or ensure only one GPU is present."
    exit 1
fi

echo "Checking GPU compute mode (GPU 0)..."
current_mode=$(get_compute_mode)

if [[ "$current_mode" == "EXCLUSIVE_PROCESS" ]]; then
    echo "GPU 0 is already in EXCLUSIVE_PROCESS mode."
else
    echo "Setting GPU compute mode to EXCLUSIVE_PROCESS for GPU 0..."
    if ! nvidia-smi -i 0 -c EXCLUSIVE_PROCESS; then
        echo "Warning: Failed to set EXCLUSIVE_PROCESS mode for GPU 0."
        echo "This usually means the GPU is in use. A reboot might be required for changes to apply."
    fi
fi

echo "Attempting to start MPS daemon..."
if is_mps_daemon_running; then
    echo "MPS daemon is already running."
else
    if nvidia-cuda-mps-control -d; then
        echo "MPS daemon started successfully."
        sleep 2 # Give it a moment to initialize
        if is_mps_daemon_running; then
             echo "MPS daemon is active."
        else
             echo "Failed to start MPS daemon. Check logs."
        fi
    else
        echo "Failed to start MPS daemon. Check permissions or if it's already running."
    fi
fi

