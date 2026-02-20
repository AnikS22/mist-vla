#!/bin/bash
#SBATCH --job-name=train_dir_mlp
#SBATCH --output=logs/train_dir_mlp_%j.out
#SBATCH --error=logs/train_dir_mlp_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu

set -eo pipefail

echo "=== Train Directional Failure MLP ==="
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

echo "Training directional MLP on combined data (single pooled model)..."
python scripts/train_directional_mlp.py \
    --success-data data/combined/success_rollouts.pkl \
    --failure-data data/combined/failure_rollouts.pkl \
    --epochs 100 \
    --lr 5e-4 \
    --batch-size 256 \
    --hidden-dim 512 \
    --save-dir checkpoints/directional_mlp \
    --seed 42

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
