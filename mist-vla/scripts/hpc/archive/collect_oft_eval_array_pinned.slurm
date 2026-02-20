#!/bin/bash
#SBATCH --job-name=mistvla_collect_pin
#SBATCH --output=logs/mistvla_collect_pin_%A_%a.out
#SBATCH --error=logs/mistvla_collect_pin_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --constraint=a100
#SBATCH --nodelist=nodegpu039,nodegpu040
#SBATCH --array=0-1

set -euo pipefail

echo "=== MIST-VLA Collect (Pinned A100) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-}"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs data/rollouts_oft_eval_big

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | head -n1)"
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-}"
nvidia-smi -L || true

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export WANDB_DISABLED=true
export WANDB_MODE=offline

CACHE_ROOT="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024/hf_cache}"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("moojink/openvla-7b-oft-finetuned-libero-spatial", resume_download=True)
PY

SEED="${SLURM_ARRAY_TASK_ID:-0}"
SAVE_DIR="data/rollouts_oft_eval_big/seed_${SEED}"
mkdir -p "${SAVE_DIR}"

python -u scripts/collect_failure_data_oft_eval.py \
  --env libero_spatial \
  --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
  --n_success 200 \
  --n_failure 400 \
  --max-attempts-per-task 50 \
  --camera-res 256 \
  --save_dir "${SAVE_DIR}" \
  --checkpoint-every 25 \
  --num-images 2 \
  --seed "${SEED}"

echo "Done: $(date)"
