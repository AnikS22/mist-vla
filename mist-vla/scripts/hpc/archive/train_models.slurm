#!/bin/bash
#SBATCH --job-name=mistvla_train
#SBATCH --output=logs/mistvla_train_%j.out
#SBATCH --error=logs/mistvla_train_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7-gpu

set -euo pipefail

echo "=== MIST-VLA Train (Risk + Time-to-Failure) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs models/risk_predictor_oft_eval_run1 models/time_failure

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"

python -u scripts/train_risk_predictor.py \
  --data data/datasets/risk_dataset_oft_eval_run1.pkl \
  --output-dir models/risk_predictor_oft_eval_run1 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cuda

python -u scripts/train_time_to_failure.py \
  --data data/datasets/risk_dataset_oft_eval_run1.pkl \
  --output models/time_failure/time_failure_oft_eval_run1.pt \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --device cuda

echo "Done: $(date)"
