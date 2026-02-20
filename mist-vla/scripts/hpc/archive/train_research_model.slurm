#!/bin/bash
#SBATCH --job-name=train_research
#SBATCH --output=logs/train_research_%j.out
#SBATCH --error=logs/train_research_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --constraint=a100

set -euo pipefail

echo "=== Research Risk Predictor Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /mnt/onefs/home/asahai2024/mist-vla

# Step 1: Prepare comprehensive dataset
echo ""
echo "=== Step 1: Preparing Dataset ==="
python scripts/prepare_comprehensive_dataset.py \
    --success data/rollouts_oft_eval_big/seed_0/success_rollouts.pkl \
    --failure data/rollouts_oft_eval_big/seed_0/failure_rollouts.pkl \
    --output data/training_datasets/research_dataset.pkl \
    --k-before-failure 30 \
    --risk-decay 0.9 \
    --balance-ratio 1.0 \
    --include-all-failure-steps

# Step 2: Train research model
echo ""
echo "=== Step 2: Training Research Model ==="
python scripts/train_risk_predictor_research.py \
    --data data/training_datasets/research_dataset.pkl \
    --output-dir checkpoints/risk_predictor_research \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --embed-dim 256 \
    --hidden-dim 512 \
    --n-ensemble 3 \
    --augment-factor 3 \
    --contrastive-weight 0.1 \
    --ttf-weight 0.2 \
    --device cuda

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
