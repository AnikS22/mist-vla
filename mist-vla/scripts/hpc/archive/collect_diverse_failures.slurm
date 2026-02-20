#!/bin/bash
#SBATCH --job-name=collect_diverse
#SBATCH --output=logs/collect_diverse_%j.out
#SBATCH --error=logs/collect_diverse_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --constraint=a100

set -eo pipefail

echo "=== Collect Diverse Failures ==="
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
# Use SLURM_TMPDIR if available, otherwise use home directory
HF_CACHE_DIR="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024}/.cache/huggingface"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

cd /mnt/onefs/home/asahai2024/mist-vla

# Set PYTHONPATH for openvla-oft imports
export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:$PYTHONPATH"
export MUJOCO_GL=egl

echo "Starting data collection with finetuned model across ALL 10 tasks..."
python scripts/collect_diverse_failures.py \
    --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
    --n-rollouts 100 \
    --perturbation-prob 0.3 \
    --perturbation-strength 0.4 \
    --save-dir data/diverse_failures \
    --checkpoint-every 50 \
    --seed 42

echo ""
echo "=== Collection Complete ==="
echo "End: $(date)"
