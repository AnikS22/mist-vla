#!/bin/bash
#SBATCH --job-name=train_risk
#SBATCH --output=logs/train_risk_%j.out
#SBATCH --error=logs/train_risk_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu

# Load modules
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mist-vla

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Set working directory
cd /mnt/onefs/home/asahai2024/mist-vla

# Default paths (can be overridden with --data and --output-dir)
DATA_PATH="${DATA_PATH:-data/training_datasets/openvla_oft_dataset_v3.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/risk_predictor_openvla_oft}"

echo "=== Training Risk Predictor ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Run training
python scripts/train_risk_predictor.py \
    --data "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model-name openvla_oft \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --hidden-dims 512 256 \
    --dropout 0.1 \
    --loss-type mse \
    --normalize \
    --device cuda

echo ""
echo "=== Training Complete ==="
