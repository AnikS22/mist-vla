#!/bin/bash
#SBATCH --job-name=paper_eval
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/paper_eval_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/paper_eval_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

###############################################################################
#  PAPER TABLE — Category 1: Full 6-Mode OpenVLA Ablation
#
#  Uses the GOOD spatial-only v4 MLP (checkpoints/eef_correction_mlp/best_model.pt)
#
#  Mode A: Vanilla VLA                — unsteered baseline
#  Mode B: Random Noise (σ=0.05)      — null hypothesis
#  Mode C: EMA Smoothing Only (β=0.9) — proves smoothing alone isn't enough
#  Mode D: Random Latent Jiggle       — matched-magnitude random (proves direction)
#  Mode E: Action MPPI                — sampling-based optimization
#  Mode F: Latent Steering (Ours)     — MLP-guided correction
#
#  20 episodes × 10 tasks × 6 modes = 1200 total episodes
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  PAPER TABLE — Category 1: OpenVLA Ablations"
echo "  6 modes × 10 tasks × 20 episodes = 1200 episodes"
echo "================================================================"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    ${SLURM_NODELIST:-unknown}"
echo "  Date:    $(date)"
echo "================================================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2
eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla

# GPU / rendering — debug what SLURM exposes
echo "=== GPU DEBUG ==="
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}"
echo "CUDA_VISIBLE_DEVICES (before)=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null
echo "=== end GPU DEBUG ==="

# Don't override CUDA_VISIBLE_DEVICES if SLURM already set it correctly
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | head -n1)"
    else
        export CUDA_VISIBLE_DEVICES=0
    fi
fi

# EGL rendering for MuJoCo
# Use device 0 since CUDA_VISIBLE_DEVICES already constrains to the allocated GPU
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# Ensure NVIDIA EGL libraries are findable
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"

echo "CUDA_VISIBLE_DEVICES (after)=${CUDA_VISIBLE_DEVICES}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"

# Quick CUDA check
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>&1 || true

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=offline

CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs results/paper_table/category1_shared_profile_v3

# Verify EGL works
echo "Verifying EGL rendering..."
python3 -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')
" || true

nvidia-smi -L
echo ""

# Use the GOOD spatial-only v4 MLP
MLP_CKPT="checkpoints/eef_correction_mlp/best_model.pt"
# Shared conservative steering profile (used for cross-model comparability)
ALPHA=0.15
EMA_BETA=0.9
MAX_CORR=0.004
CORR_THRESH=0.003
FAIL_THRESH=0.6
echo "  MLP Checkpoint: ${MLP_CKPT}"
echo "  Shared Steering: alpha=${ALPHA} ema_beta=${EMA_BETA} max_corr=${MAX_CORR}m threshold=${CORR_THRESH}m fail_gate=p>=${FAIL_THRESH}"
echo ""

python -u scripts/eval_tuning.py \
    --model-name "moojink/openvla-7b-oft-finetuned-libero-spatial" \
    --mlp-checkpoint "${MLP_CKPT}" \
    --env libero_spatial \
    --modes vanilla noise ema_only latent_jiggle mppi steering \
    --episodes-per-task 20 \
    --noise-sigma 0.05 \
    --ema-only-beta 0.9 \
    --alpha "${ALPHA}" \
    --ema-beta "${EMA_BETA}" \
    --max-correction "${MAX_CORR}" \
    --correction-threshold "${CORR_THRESH}" \
    --use-fail-gate \
    --fail-threshold "${FAIL_THRESH}" \
    --mppi-samples 16 \
    --mppi-temperature 5.0 \
    --save-dir results/paper_table/category1_shared_profile_v3

echo ""
echo "================================================================"
echo "  PAPER TABLE — Category 1 COMPLETE"
echo "================================================================"
echo "  Date: $(date)"
echo "  Results: results/paper_table/category1_shared_profile_v3/eval_results.json"
echo "================================================================"
