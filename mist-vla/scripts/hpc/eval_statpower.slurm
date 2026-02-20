#!/bin/bash
#SBATCH --job-name=stat_eval
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/stat_eval_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/stat_eval_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7
#SBATCH --nodelist=nodegpu042

set -eo pipefail

echo "================================================================"
echo "  STATISTICAL POWER EVAL — 20 episodes/task"
echo "  Configs: vanilla vs clamp=0.01m"
echo "  Goal: reliable ±10pp confidence intervals"
echo "================================================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    ${SLURM_NODELIST:-unknown}"
echo "Date:    $(date)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

# ── Environment setup ──
module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | head -n1)"
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=offline

CACHE_ROOT="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024/hf_cache}"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs results/tuning

echo ""
nvidia-smi -L || true
echo ""

echo "Ensuring model is cached..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial', resume_download=True)
"
echo "  ✓ Cached"
echo ""

# ══════════════════════════════════════════════════════════════
#  20 episodes × 4 tasks × 2 configs (vanilla + clamp=0.01m)
#  = 160 total episodes → reliable confidence intervals
# ══════════════════════════════════════════════════════════════

python -u scripts/eval_tuning.py \
    --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
    --mlp-checkpoint checkpoints/eef_correction_mlp/best_model.pt \
    --env libero_spatial \
    --tasks 4 6 7 8 \
    --episodes-per-task 20 \
    --alpha 0.1 \
    --ema-beta 0.7 \
    --action-scale 0.05 \
    --gate-mode magnitude \
    --correction-threshold 0.005 \
    --max-corrections 0.01 \
    --seed 42 \
    --save-dir results/tuning

echo ""
echo "================================================================"
echo "  STATISTICAL POWER EVAL COMPLETE"
echo "================================================================"
echo "Results: results/tuning/clamping_sweep.json"
echo "Date:    $(date)"
echo "================================================================"
