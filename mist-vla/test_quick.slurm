#!/bin/bash
#SBATCH --job-name=mist-quick-test
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "========================================="
echo "MIST-VLA Quick Test (5-10 minutes)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Activate conda environment
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mist-vla

# Verify environment
if [[ "$CONDA_DEFAULT_ENV" != "mist-vla" ]]; then
    echo "ERROR: Failed to activate mist-vla environment"
    exit 1
fi
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"

cd $SLURM_SUBMIT_DIR
mkdir -p logs data/test_rollouts

# Print environment info
echo ""
echo "=== Environment ==="
python --version
which python
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Test imports
echo "=== Testing Imports ==="
python -c "
import torch
import transformers
from transformers import AutoProcessor
import libero
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ LIBERO available')
print('✓ CUDA available:', torch.cuda.is_available())
" || {
    echo "ERROR: Import test failed"
    exit 1
}

# Run quick test
echo ""
echo "========================================="
echo "Running Quick Real Data Test..."
echo "========================================="
python test_real_quick.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ QUICK TEST PASSED!"
    echo "========================================="
    echo ""
    echo "The system is working correctly!"
    echo "You can now run the full pipeline:"
    echo "  sbatch run_hpc.slurm"
else
    echo ""
    echo "❌ Test failed - check errors above"
    exit 1
fi
