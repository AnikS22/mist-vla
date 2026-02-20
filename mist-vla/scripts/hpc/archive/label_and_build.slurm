#!/bin/bash
#SBATCH --job-name=mistvla_label
#SBATCH --output=logs/mistvla_label_%j.out
#SBATCH --error=logs/mistvla_label_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=shortq7-gpu

set -euo pipefail

echo "=== MIST-VLA Label + Build Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module load miniconda3/24.3.0-gcc-13.2.0-rslr3to

eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs data/rollouts_oft_eval_run1_labeled data/datasets

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"

python -u scripts/label_failure_data.py \
  --input data/rollouts_oft_eval_run1/success_rollouts.pkl \
  --output data/rollouts_oft_eval_run1_labeled/success_rollouts_labeled.pkl

python -u scripts/label_failure_data.py \
  --input data/rollouts_oft_eval_run1/failure_rollouts.pkl \
  --output data/rollouts_oft_eval_run1_labeled/failure_rollouts_labeled.pkl

python -u scripts/build_risk_dataset.py \
  --success data/rollouts_oft_eval_run1_labeled/success_rollouts_labeled.pkl \
  --failure data/rollouts_oft_eval_run1_labeled/failure_rollouts_labeled.pkl \
  --output data/datasets/risk_dataset_oft_eval_run1.pkl

echo "Done: $(date)"
