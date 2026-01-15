#!/bin/bash
#SBATCH --job-name=mist-vla
#SBATCH --output=logs/mist_vla_%j.out
#SBATCH --error=logs/mist_vla_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Adjust the above based on your HPC's configuration

echo "========================================="
echo "MIST-VLA Experiment on HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Load modules (uncomment and adjust for your HPC)
# module load cuda/12.1
# module load gcc/11.2.0

# Activate conda environment properly
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate mist-vla

# Verify environment is active
if [[ "$CONDA_DEFAULT_ENV" != "mist-vla" ]]; then
    echo "ERROR: Failed to activate mist-vla environment"
    exit 1
fi
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/rollouts
mkdir -p data/steering_vectors
mkdir -p results

# Print environment info
echo ""
echo "=== Environment Info ==="
python --version
which python
pip list | grep -E "(torch|transformers|libero)"
echo ""

# Print GPU info
nvidia-smi
echo ""

# Test imports before starting
echo "=== Testing Imports ==="
python -c "
import torch
import transformers
from transformers import AutoProcessor
print('✓ All critical imports work')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
" || {
    echo "ERROR: Import test failed"
    exit 1
}
echo ""

echo "========================================="
echo "Step 1: Collecting Failure Data"
echo "========================================="
python scripts/collect_failure_data.py \
    --env libero_spatial \
    --n_success 100 \
    --n_failure 100 \
    --save_dir data/rollouts

if [ $? -ne 0 ]; then
    echo "ERROR: Data collection failed"
    exit 1
fi
echo "✓ Step 1 complete"

echo ""
echo "========================================="
echo "Step 2: Training Failure Detector"
echo "========================================="
python scripts/train_failure_detector.py \
    --data_dir data/rollouts \
    --detector_type mlp \
    --epochs 50

if [ $? -ne 0 ]; then
    echo "ERROR: Detector training failed"
    exit 1
fi
echo "✓ Step 2 complete"

echo ""
echo "========================================="
echo "Step 3: Extracting Steering Vectors"
echo "========================================="
python scripts/extract_steering_vectors.py \
    --model openvla/openvla-7b \
    --save_path data/steering_vectors

if [ $? -ne 0 ]; then
    echo "ERROR: Steering vector extraction failed"
    exit 1
fi
echo "✓ Step 3 complete"

echo ""
echo "========================================="
echo "Step 4: Running Evaluation"
echo "========================================="
python scripts/run_libero_eval.py \
    --task_suite libero_spatial \
    --model_path openvla/openvla-7b \
    --detector_path checkpoints/best_detector.pt \
    --steering_path data/steering_vectors/all_vectors.pt \
    --n_episodes 50

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi
echo "✓ Step 4 complete"

echo ""
echo "========================================="
echo "✅ Experiment Complete!"
echo "========================================="

# Copy results to a timestamped directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results_archive/${TIMESTAMP}
cp -r results/* results_archive/${TIMESTAMP}/ 2>/dev/null || echo "No results to archive"
cp -r checkpoints results_archive/${TIMESTAMP}/ 2>/dev/null || echo "No checkpoints to archive"

echo ""
echo "Results saved to: results_archive/${TIMESTAMP}/"
echo ""
echo "Summary:"
ls -lh results/
ls -lh checkpoints/
