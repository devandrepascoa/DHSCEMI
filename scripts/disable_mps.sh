#!/bin/bash

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: This script must be run with sudo or as root."
        echo "Usage: sudo $0"
        exit 1
    fi
}

get_compute_mode() {
    nvidia-smi -q -d COMPUTE | grep "Compute Mode" | awk '{print $NF}' | head -n 1
}

is_mps_daemon_running() {
    pgrep -x "nvidia-cuda-mps-control" >/dev/null
    return $?
}

check_root
echo "--- Disabling NVIDIA MPS ---"

num_gpus=$(nvidia-smi -L | wc -l)
if [ "$num_gpus" -ne 1 ]; then
    echo "Error: This script is for single GPU machines. Found $num_gpus GPUs."
    echo "Please adjust the script or ensure only one GPU is present."
    exit 1
fi

echo "Attempting to stop MPS daemon..."
if is_mps_daemon_running; then
    echo "Sending quit command to MPS daemon..."
    echo quit | sudo nvidia-cuda-mps-control
    sleep 2 # Give it a moment to shut down
    if is_mps_daemon_running; then
        echo "MPS daemon is still running. Attempting force kill..."
        sudo pkill -9 nvidia-mps-server || true # Use || true to prevent script from exiting if not found
        sudo pkill -9 nvidia-cuda-mps-control || true
        echo "MPS daemon force killed (if it was running)."
    else
        echo "MPS daemon stopped successfully."
    fi
else
    echo "MPS daemon is not running."
fi

echo "Setting GPU compute mode to DEFAULT for GPU 0..."
if ! sudo nvidia-smi -i 0 -c DEFAULT; then
    echo "Warning: Failed to set DEFAULT mode for GPU 0."
    echo "This usually means the GPU is in use. A reboot might be required for changes to apply."
fi

echo "--- MPS Disablement Attempt Complete ---"
echo "RECOMMENDATION: If compute mode was changed and GPU was in use, consider a reboot."
echo "Verification: Run 'nvidia-smi -q -d COMPUTE' and 'ps -ef | grep mps'."
