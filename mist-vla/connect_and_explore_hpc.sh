#!/bin/bash
# Connect to HPC and explore available resources

HPC_USER="asahai2024"
HPC_HOST="athene-login.fau.edu"

echo "========================================="
echo "Connecting to HPC and Exploring Resources"
echo "========================================="
echo ""

# Try to connect and run exploration commands
ssh ${HPC_USER}@${HPC_HOST} << 'ENDSSH'
echo "=== System Information ==="
hostname
uname -a
echo ""

echo "=== Home Directory ==="
pwd
ls -lh ~/ | head -20
echo ""

echo "=== Available Partitions ==="
sinfo 2>/dev/null || echo "SLURM not available or sinfo command failed"
echo ""

echo "=== GPU Information ==="
sinfo -o "%20P %10G %10c %10m %25f %15N" 2>/dev/null || echo "GPU info not available"
echo ""

echo "=== User Account/Allocation ==="
sacctmgr show associations user=$USER 2>/dev/null || echo "Account info not available"
echo ""

echo "=== Disk Space ==="
df -h ~/ 2>/dev/null
quota 2>/dev/null || echo "Quota command not available"
echo ""

echo "=== Available Modules ==="
module avail 2>&1 | head -30 || echo "Module system not available"
echo ""

echo "=== Current Jobs ==="
squeue -u $USER 2>/dev/null || echo "No jobs or squeue not available"
echo ""

echo "=== Check if mist-vla directory exists ==="
if [ -d ~/mist-vla ]; then
    echo "✓ mist-vla directory exists"
    ls -lh ~/mist-vla/ | head -20
else
    echo "✗ mist-vla directory does not exist"
fi
echo ""

echo "=== Python/Conda Availability ==="
which python python3 conda 2>/dev/null
python3 --version 2>/dev/null || echo "Python3 not found"
conda --version 2>/dev/null || echo "Conda not found"
echo ""

echo "=== CUDA Availability ==="
which nvcc nvidia-smi 2>/dev/null
nvcc --version 2>/dev/null || echo "CUDA compiler not found"
echo ""

ENDSSH

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Connection successful!"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Connection failed"
    echo "========================================="
    echo "Please check:"
    echo "1. Hostname: $HPC_HOST"
    echo "2. Username: $HPC_USER"
    echo "3. Network connectivity"
fi
