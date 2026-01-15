#!/bin/bash
# Manual HPC Transfer Script
# Run this from your LOCAL machine after connecting to FAU VPN

echo "========================================="
echo "MIST-VLA HPC Transfer Script"
echo "========================================="

# HPC login details
HPC_USER="asahai2024"
HPC_HOST="athene-login.hpc.fau.edu"
HPC_PATH="~/mist-vla"

echo "Transferring to: ${HPC_USER}@${HPC_HOST}:${HPC_PATH}"
echo ""

# Check if we can reach HPC
echo "Testing connection..."
ssh -o ConnectTimeout=5 ${HPC_USER}@${HPC_HOST} "echo '✓ Connection successful'" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Cannot connect to HPC"
    echo ""
    echo "Possible fixes:"
    echo "1. Connect to FAU VPN first"
    echo "2. Check if hostname is correct: ${HPC_HOST}"
    echo "3. Verify your username: ${HPC_USER}"
    echo ""
    echo "Try manually:"
    echo "  ssh ${HPC_USER}@${HPC_HOST}"
    exit 1
fi

echo ""
echo "Starting transfer..."
echo ""

# Transfer files
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='data/rollouts/*' \
    --exclude='checkpoints/*' \
    --exclude='results/*' \
    /home/mpcr/Desktop/SalusV5/mist-vla/ \
    ${HPC_USER}@${HPC_HOST}:${HPC_PATH}/

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Transfer Complete!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. SSH into HPC:"
    echo "   ssh ${HPC_USER}@${HPC_HOST}"
    echo ""
    echo "2. Navigate to project:"
    echo "   cd ~/mist-vla"
    echo ""
    echo "3. Run setup:"
    echo "   chmod +x HPC_SETUP.sh"
    echo "   bash HPC_SETUP.sh"
    echo ""
    echo "4. Submit job:"
    echo "   sbatch run_hpc.slurm"
    echo ""
else
    echo "❌ Transfer failed"
    exit 1
fi
