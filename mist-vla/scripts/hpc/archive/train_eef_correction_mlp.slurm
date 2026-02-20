#!/bin/bash
#SBATCH --job-name=eef_corr_mlp
#SBATCH --output=logs/eef_correction_mlp_%j.out
#SBATCH --error=logs/eef_correction_mlp_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu

set -eo pipefail

echo "===== EEF Cartesian Correction MLP ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:$PYTHONPATH"
cd /mnt/onefs/home/asahai2024/mist-vla

mkdir -p logs checkpoints/eef_correction_mlp

# ── Pooled Training + True LOO ──
python scripts/train_eef_correction_mlp.py \
    --success-data data/combined/success_rollouts.pkl \
    --failure-data data/combined/failure_rollouts.pkl \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --hidden-dim 512 \
    --save-dir checkpoints/eef_correction_mlp \
    --run-loo \
    --seed 42

echo ""
echo "===== EEF Correction MLP Complete ====="
echo "Results: checkpoints/eef_correction_mlp/results.json"
date
