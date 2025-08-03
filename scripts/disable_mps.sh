#!/bin/bash

# ==============================================================================
# disable_mps.sh: Disable NVIDIA CUDA Multi-Process Service (MPS)
# (Single GPU Version)
# ==============================================================================
# This script stops the NVIDIA MPS daemon and sets the GPU compute mode back
# to DEFAULT for a system with a single NVIDIA GPU.
#
# IMPORTANT WARNINGS:
# 1. ROOT PRIVILEGES: This script must be run with 'sudo' or as root.
# 2. GPU USAGE: Ensure ALL CUDA applications and Docker containers using MPS
#    are stopped BEFORE running this script. Otherwise, it may fail or cause
#    instability. A reboot might be necessary if the GPU is in heavy use.
# 3. COMPATIBILITY: Assumes a single NVIDIA GPU (compute capability 3.5+).
# 4. MIG: MPS is distinct from MIG.
# ==============================================================================

# --- Functions ---

# Function to check for root privileges
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: This script must be run with sudo or as root."
        echo "Usage: sudo $0"
        exit 1
    fi
}

# Function to check GPU compute mode for the single GPU
get_compute_mode() {
    nvidia-smi -q -d COMPUTE | grep "Compute Mode" | awk '{print $NF}' | head -n 1
}

# Function to check if MPS daemon is running
is_mps_daemon_running() {
    pgrep -x "nvidia-cuda-mps-control" >/dev/null
    return $?
}

# --- Main Script Logic ---
check_root
echo "--- Disabling NVIDIA MPS ---"

# Verify a single GPU is present
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