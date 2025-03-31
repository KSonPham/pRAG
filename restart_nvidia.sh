#!/bin/bash

# Check if running as root (or with sudo)
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root or with sudo."
    exit 1
fi

# Get PIDs of processes using /dev/nvidia-uvm
pids=$(sudo lsof | grep nvidia.uvm | awk '{print $2}' | sort -u)

if [ -z "$pids" ]; then
    echo "No processes found using /dev/nvidia-uvm."
    exit 0
fi

echo "Found the following PIDs using /dev/nvidia-uvm:"
echo "$pids"

# Kill all identified processes
for pid in $pids; do
    echo "Killing PID: $pid"
    kill -9 "$pid" 2>/dev/null
done

echo "All processes using /dev/nvidia-uvm have been terminated."
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
echo "NVIDIA UVM module has been reloaded."
echo "Restarting NVIDIA driver..."