#!/bin/bash
# Run this ON THE HPC CLUSTER after SSH'ing in
# This will explore available resources and suggest best configuration

echo "========================================="
echo "MIST-VLA HPC Resource Exploration"
echo "========================================="
echo ""

echo "=== System Information ==="
hostname
uname -a
echo ""

echo "=== Current Directory ==="
pwd
ls -lh | head -20
echo ""

echo "=== Available Partitions ==="
sinfo
echo ""

echo "=== GPU Types Available ==="
sinfo -o "%20P %10G %10c %10m %25f %15N"
echo ""

echo "=== User Account/Allocation ==="
sacctmgr show associations user=$USER
echo ""

echo "=== Disk Space and Quota ==="
df -h ~/
quota 2>/dev/null || echo "Quota command not available"
echo ""

echo "=== Available Modules ==="
echo "Checking common modules..."
module avail cuda 2>&1 | head -10
module avail gcc 2>&1 | head -10
module avail python 2>&1 | head -10
module avail conda 2>&1 | head -10
echo ""

echo "=== Current Jobs ==="
squeue -u $USER
echo ""

echo "=== Python/Conda Availability ==="
which python python3 conda
python3 --version 2>/dev/null
conda --version 2>/dev/null
echo ""

echo "=== CUDA Availability ==="
which nvcc nvidia-smi
nvcc --version 2>/dev/null
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (not on GPU node)"
echo ""

echo "=== Check mist-vla directory ==="
if [ -d ~/mist-vla ]; then
    echo "✓ mist-vla directory exists"
    echo "Contents:"
    ls -lh ~/mist-vla/ | head -20
    echo ""
    echo "Disk usage:"
    du -sh ~/mist-vla
else
    echo "✗ mist-vla directory does not exist yet"
fi
echo ""

echo "=== Recommendations ==="
echo ""
echo "Based on the output above, you should:"
echo "1. Note the GPU partition name (from sinfo)"
echo "2. Note the GPU types available (A100, V100, etc.)"
echo "3. Check available disk space"
echo "4. Verify CUDA/Python modules"
echo ""
echo "Then update run_hpc.slurm with:"
echo "  - Correct partition name"
echo "  - Appropriate GPU request (--gres=gpu:1 or --gres=gpu:a100:1)"
echo "  - Memory requirements"
echo ""
