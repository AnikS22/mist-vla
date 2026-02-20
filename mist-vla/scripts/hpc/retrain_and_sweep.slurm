#!/bin/bash
#SBATCH --job-name=retrain_v3
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/retrain_v3_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/retrain_v3_%j.err
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7
#SBATCH --nodelist=nodegpu042

set -eo pipefail

echo "================================================================"
echo "  RETRAIN MLP v3 (reduced capacity + strong regularisation)"
echo "  hidden_dim=256  dropout=0.3/0.2  noise=0.05  wd=5e-4"
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

mkdir -p logs results/tuning checkpoints/eef_correction_mlp_v2

echo ""
echo "GPU sanity check:"
nvidia-smi -L || true
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 1: RETRAIN MLP WITH ANTI-OVERFITTING MEASURES
# ══════════════════════════════════════════════════════════════
#
#  ANTI-OVERFITTING v3:
#    Previous run (v2) showed cosine gap = 0.20 (train=0.93, val=0.73)
#    → correction head was overfitting hard.
#
#    Fixes applied:
#    1. Reduced hidden_dim: 512→256 (~600K params instead of 2.4M)
#    2. Higher dropout: 0.3/0.2 (was 0.2/0.1) + extra layer before heads
#    3. Stronger input noise: σ=0.05 (was 0.01)
#    4. Higher weight decay: 5e-4 (was 1e-4)
#    5. Cosine-gap early stop: halts if train-val cos gap > 0.20
#    6. Correction magnitude L2 penalty (λ=0.1) on success samples

echo "STEP 1: Retraining MLP (v3 — reduced capacity)..."
echo "  hidden_dim=256  dropout=0.3/0.2  noise=0.05  wd=5e-4"
echo ""

# Back up old model
if [ -f checkpoints/eef_correction_mlp/best_model.pt ]; then
    cp checkpoints/eef_correction_mlp/best_model.pt \
       checkpoints/eef_correction_mlp/best_model_v1_backup.pt
    echo "  ✓ Backed up old model to best_model_v1_backup.pt"
fi

python -u scripts/train_eef_correction_mlp.py \
    --success-data data/combined/success_rollouts.pkl \
    --failure-data data/combined/failure_rollouts.pkl \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --hidden-dim 256 \
    --dropout1 0.3 \
    --dropout2 0.2 \
    --weight-decay 5e-4 \
    --save-dir checkpoints/eef_correction_mlp \
    --input-noise 0.05 \
    --corr-mag-penalty 0.1 \
    --seed 42

echo ""
echo "STEP 1 COMPLETE — MLP retrained"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 2: RE-RUN CLAMPING SWEEP WITH NEW MODEL
# ══════════════════════════════════════════════════════════════

echo "STEP 2: Ensuring VLA model is cached..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial', resume_download=True)
"
echo "  ✓ Model cached"
echo ""

echo "STEP 2: Clamping sweep with anti-overfit MLP..."
echo ""

python -u scripts/eval_tuning.py \
    --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
    --mlp-checkpoint checkpoints/eef_correction_mlp/best_model.pt \
    --env libero_spatial \
    --tasks 4 6 7 8 \
    --episodes-per-task 5 \
    --alpha 0.1 \
    --ema-beta 0.7 \
    --action-scale 0.05 \
    --gate-mode magnitude \
    --correction-threshold 0.005 \
    --max-corrections 0.005 0.01 0.02 0.05 999.0 \
    --seed 42 \
    --save-dir results/tuning

echo ""
echo "================================================================"
echo "  RETRAIN + SWEEP COMPLETE"
echo "================================================================"
echo "MLP:     checkpoints/eef_correction_mlp/best_model.pt"
echo "Curves:  checkpoints/eef_correction_mlp/training_curves.json"
echo "Results: results/tuning/clamping_sweep.json"
echo "Date:    $(date)"
echo "================================================================"
