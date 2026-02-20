#!/bin/bash
#SBATCH --job-name=true_loo
#SBATCH --output=logs/true_loo_%j.out
#SBATCH --error=logs/true_loo_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu

set -eo pipefail

echo "=== True LOO + Collision Correlation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Start: $(date)"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /mnt/onefs/home/asahai2024/mist-vla
export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:$PYTHONPATH"

echo ""
echo "========================================="
echo "STEP 1: Collision Ground Truth Correlation"
echo "========================================="
python scripts/collision_ground_truth_correlation.py

echo ""
echo "========================================="
echo "STEP 2: True Leave-One-Task-Out (×10)"
echo "========================================="
python scripts/true_loo_evaluation.py \
    --success-data data/combined/success_rollouts.pkl \
    --failure-data data/combined/failure_rollouts.pkl \
    --epochs 60 \
    --lr 5e-4 \
    --batch-size 256 \
    --hidden-dim 512 \
    --save-dir checkpoints/true_loo \
    --seed 42

echo ""
echo "=== All Done ==="
echo "End: $(date)"
