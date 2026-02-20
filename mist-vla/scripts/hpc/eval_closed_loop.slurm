#!/bin/bash
#SBATCH --job-name=eval_cl
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/eval_cl_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/eval_cl_%j.err
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
echo "  CLOSED-LOOP EVALUATION: 3-WAY BASELINE COMPARISON  (v4)"
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

# GPU / rendering
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

# HuggingFace cache
CACHE_ROOT="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024/hf_cache}"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs results/closed_loop_v4

echo ""
echo "GPU sanity check:"
nvidia-smi -L || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo ""

# ── Ensure model is cached ──
echo "Ensuring model is cached..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial', resume_download=True)
"
echo "  ✓ Model cached"
echo ""

# ══════════════════════════════════════════════════════════════
#  RUN CLOSED-LOOP EVALUATION  (v4 — Magnitude Gating)
# ══════════════════════════════════════════════════════════════
#
#  THE FIX (v4):
#  ─────────────
#  Previous runs showed p_fail gating was poorly calibrated,
#  causing false-positive interventions (Task 7 catastrophe).
#
#  v4 uses MAGNITUDE GATING: only intervene when the MLP's
#  predicted correction vector ‖c‖ > δ meters.  The regression
#  head is better calibrated — if the VLA is on track, ‖c‖ ≈ 0.
#
#  Parameters (will be refined by gating sweep):
#    gate_mode = magnitude
#    correction_threshold = 0.01 m  (= 1 cm)
#    alpha = 0.1
#    action_scale = 0.05  (1 action unit ≈ 5 cm)

echo "Running closed-loop evaluation (v4 — magnitude gating)..."
echo "  Modes: vanilla, noise, steering"
echo "  Gate:  magnitude  δ=0.01m"
echo "  α=0.1  scale=0.05"
echo ""

python -u scripts/eval_closed_loop_study.py \
    --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
    --mlp-checkpoint checkpoints/eef_correction_mlp/best_model.pt \
    --env libero_spatial \
    --n-episodes 50 \
    --alpha 0.1 \
    --ema-beta 0.7 \
    --gate-mode magnitude \
    --correction-threshold 0.01 \
    --action-scale 0.05 \
    --noise-sigma 0.05 \
    --modes vanilla noise steering \
    --expert-data data/multi_suite \
    --save-dir results/closed_loop_v4 \
    --save-video \
    --seed 42

echo ""
echo "================================================================"
echo "  EVALUATION COMPLETE  (v4)"
echo "================================================================"
echo "Results:  results/closed_loop_v4/results_table.json"
echo "Details:  results/closed_loop_v4/episode_details.json"
echo "Video:    results/closed_loop_v4/combined_video.mp4"
echo "Date:     $(date)"
echo "================================================================"
