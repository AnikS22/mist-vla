#!/bin/bash
#SBATCH --job-name=mistvla_oft_eval
#SBATCH --output=logs/mistvla_oft_eval_%j.out
#SBATCH --error=logs/mistvla_oft_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --constraint=a100
#SBATCH --nodelist=nodegpu039

set -euo pipefail

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs

export WANDB_DISABLED=true
export WANDB_MODE=offline
export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"

CACHE_ROOT="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024/hf_cache}"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("moojink/openvla-7b-oft-finetuned-libero-spatial", resume_download=True)
PY

python -u /mnt/onefs/home/asahai2024/mist-vla/openvla-oft/experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --num_open_loop_steps 8 \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_l1_regression True \
  --use_diffusion False \
  --center_crop True \
  --use_wandb False
