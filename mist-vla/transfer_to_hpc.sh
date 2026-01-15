#!/bin/bash
# Transfer MIST-VLA to HPC
# Run this script when connected to university network or VPN

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           Transferring MIST-VLA to HPC                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# HPC credentials
HPC_USER="asahai2024"
HPC_HOST="athene-login.fau.edu"
HPC_PATH="~/mist-vla"

# Source directory
SRC_DIR="/home/mpcr/Desktop/SalusV5/mist-vla"

echo "[1/4] Cleaning local cache files..."
cd "$SRC_DIR"
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Cleaned"

echo ""
echo "[2/4] Testing connection to HPC..."
if ping -c 1 -W 2 "$HPC_HOST" &>/dev/null; then
    echo "  ✓ HPC is reachable"
else
    echo "  ✗ Cannot reach $HPC_HOST"
    echo ""
    echo "⚠️  Possible issues:"
    echo "   - Not connected to university network"
    echo "   - VPN not running"
    echo "   - Hostname incorrect"
    echo ""
    echo "Please ensure you are connected to the university network or VPN"
    exit 1
fi

echo ""
echo "[3/4] Transferring files to HPC..."
echo "  Source: $SRC_DIR"
echo "  Destination: ${HPC_USER}@${HPC_HOST}:${HPC_PATH}"
echo ""

rsync -avz --progress \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='results/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.pkl' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='experiments/' \
    --exclude='notebooks/' \
    "$SRC_DIR/" "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/"

if [ $? -eq 0 ]; then
    echo ""
    echo "[4/4] Transfer complete! ✅"
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                   Next Steps on HPC                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "1. SSH to HPC:"
    echo "   ssh ${HPC_USER}@${HPC_HOST}"
    echo ""
    echo "2. Navigate to project:"
    echo "   cd ~/mist-vla"
    echo ""
    echo "3. Verify setup:"
    echo "   python scripts/verify_phase0.py"
    echo ""
    echo "4. Quick test (5 rollouts, ~1 hour):"
    echo "   python scripts/collect_phase1_data.py --num-rollouts 5 --max-steps 50"
    echo ""
    echo "5. Full pipeline (submit job):"
    echo "   sbatch run_hpc.slurm"
    echo ""
    echo "See NEXT_STEPS.md for detailed instructions."
    echo ""
else
    echo ""
    echo "✗ Transfer failed"
    exit 1
fi
