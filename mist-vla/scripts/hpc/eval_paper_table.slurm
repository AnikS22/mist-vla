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
#  PAPER TABLE — Category 1: Full 7-Mode OpenVLA Ablation
#
#  Uses the GOOD spatial-only v4 MLP (checkpoints/eef_correction_mlp/best_model.pt)
#
#  Mode A: Vanilla VLA                — unsteered baseline
#  Mode B: Latent Stop                — SAFE-style freeze baseline
#  Mode C: Random Noise (σ=0.05)      — null hypothesis
#  Mode D: EMA Smoothing Only (β=0.9) — proves smoothing alone isn't enough
#  Mode E: Random Latent Jiggle       — matched-magnitude random (proves direction)
#  Mode F: Action MPPI                — sampling-based optimization
#  Mode G: Latent Steering (Ours)     — MLP-guided correction
#
#  20 episodes × 10 tasks × 7 modes = 1400 total episodes
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  PAPER TABLE — Category 1: OpenVLA Ablations"
echo "  7 modes × 10 tasks × 20 episodes = 1400 episodes"
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
nvidia-smi -L 2>/dev/null || true
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

# Quick CUDA check + hard fail if no GPU visible
python3 - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
sys.exit(0 if ok else 2)
PY

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export WANDB_MODE=offline

CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

RUN_TAG="${RUN_TAG:-shared_profile_v3}"
N_EPISODES="${N_EPISODES:-20}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
SEED="${SEED:-42}"
mkdir -p logs "results/paper_table/category1_${RUN_TAG}"

# Verify EGL works
echo "Verifying EGL rendering..."
python3 -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')
" || true

nvidia-smi -L || true
echo ""

# Use the GOOD spatial-only v4 MLP
MLP_CKPT="checkpoints/eef_correction_mlp/best_model.pt"
# Shared conservative steering profile (used for cross-model comparability)
ALPHA="${ALPHA:-0.15}"
EMA_BETA="${EMA_BETA:-0.9}"
MAX_CORR="${MAX_CORR:-0.004}"
CORR_THRESH="${CORR_THRESH:-0.003}"
FAIL_THRESH="${FAIL_THRESH:-0.6}"
STOP_THRESH="${STOP_THRESH:-0.85}"
OOD_OBSTACLE="${OOD_OBSTACLE:-0}"
OOD_STEP_MIN="${OOD_STEP_MIN:-40}"
OOD_STEP_MAX="${OOD_STEP_MAX:-160}"
OOD_DURATION="${OOD_DURATION:-20}"
OOD_PUSH_MAG="${OOD_PUSH_MAG:-0.08}"
OOD_BDDL_MAP="${OOD_BDDL_MAP:-}"
STRICT_OOD_BDDL="${STRICT_OOD_BDDL:-0}"
echo "  MLP Checkpoint: ${MLP_CKPT}"
echo "  Run tag: ${RUN_TAG}  episodes=${N_EPISODES}"
echo "  Tasks: ${TASKS}  seed=${SEED}"
echo "  Shared Steering: alpha=${ALPHA} ema_beta=${EMA_BETA} max_corr=${MAX_CORR}m threshold=${CORR_THRESH}m fail_gate=p>=${FAIL_THRESH}"
echo "  Latent Stop: stop_threshold=${STOP_THRESH}"
echo "  OOD obstacle: enable=${OOD_OBSTACLE} step=[${OOD_STEP_MIN},${OOD_STEP_MAX}] duration=${OOD_DURATION} push=${OOD_PUSH_MAG}"
echo "  OOD BDDL map: ${OOD_BDDL_MAP:-<none>}  strict=${STRICT_OOD_BDDL}"
echo ""

python -u scripts/eval_tuning.py \
    --model-name "moojink/openvla-7b-oft-finetuned-libero-spatial" \
    --mlp-checkpoint "${MLP_CKPT}" \
    --env libero_spatial \
    --tasks ${TASKS} \
    --modes vanilla latent_stop noise ema_only latent_jiggle mppi steering \
    --episodes-per-task "${N_EPISODES}" \
    --noise-sigma 0.05 \
    --ema-only-beta 0.9 \
    --alpha "${ALPHA}" \
    --ema-beta "${EMA_BETA}" \
    --max-correction "${MAX_CORR}" \
    --correction-threshold "${CORR_THRESH}" \
    --use-fail-gate \
    --fail-threshold "${FAIL_THRESH}" \
    --stop-threshold "${STOP_THRESH}" \
    --mppi-samples 16 \
    --mppi-temperature 5.0 \
    --seed "${SEED}" \
    $( [ "${OOD_OBSTACLE}" = "1" ] && echo "--ood-obstacle --ood-step-min ${OOD_STEP_MIN} --ood-step-max ${OOD_STEP_MAX} --ood-duration ${OOD_DURATION} --ood-push-magnitude ${OOD_PUSH_MAG}" ) \
    $( [ -n "${OOD_BDDL_MAP}" ] && echo "--ood-bddl-map ${OOD_BDDL_MAP}" ) \
    $( [ "${STRICT_OOD_BDDL}" = "1" ] && echo "--strict-ood-bddl" ) \
    --save-dir "results/paper_table/category1_${RUN_TAG}"

echo ""
echo "================================================================"
echo "  PAPER TABLE — Category 1 COMPLETE"
echo "================================================================"
echo "  Date: $(date)"
echo "  Results: results/paper_table/category1_${RUN_TAG}/eval_results.json"
echo "================================================================"
