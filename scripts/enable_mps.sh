#!/bin/bash

# ==============================================================================
# enable_mps.sh: Enable NVIDIA CUDA Multi-Process Service (MPS)
# (Single GPU Version)
# ==============================================================================
# This script sets the GPU compute mode to EXCLUSIVE_PROCESS and starts the
# NVIDIA MPS daemon for a system with a single NVIDIA GPU.
#
# IMPORTANT WARNINGS:
# 1. ROOT PRIVILEGES: This script must be run with 'sudo' or as root.
# 2. GPU USAGE: Ensure no critical GPU applications are running before enabling MPS.
#    Ideally, the GPU should be idle. A reboot might be necessary if the GPU is
#    in heavy use and you change modes.
# 3. COMPATIBILITY: Assumes a single NVIDIA GPU (compute capability 3.5+)
#    with NVIDIA drivers and the NVIDIA Container Toolkit installed.
# 4. MIG: MPS is distinct from MIG. Do not use this script if you intend
#    to use MIG or if your GPU is already in MIG mode.
# 5. DOCKER: Remember to run Docker containers with '--ipc=host' to use MPS.
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
echo "--- Enabling NVIDIA MPS ---"

# Verify a single GPU is present
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
    if ! sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS; then
        echo "Warning: Failed to set EXCLUSIVE_PROCESS mode for GPU 0."
        echo "This usually means the GPU is in use. A reboot might be required for changes to apply."
    fi
fi

echo "Attempting to start MPS daemon..."
if is_mps_daemon_running; then
    echo "MPS daemon is already running."
else
    if sudo nvidia-cuda-mps-control -d; then
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

echo "--- MPS Enablement Attempt Complete ---"
echo "RECOMMENDATION: If compute mode was changed and GPU was in use, consider a reboot."
echo "Verification: Run 'nvidia-smi -q -d COMPUTE' and 'ps -ef | grep mps'."
echo "Docker containers MUST use '--ipc=host'."