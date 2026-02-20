#!/bin/bash
#SBATCH --job-name=v4_eval
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/v4_eval_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/v4_eval_%j.err
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
echo "  ARCHITECTURE v4 — SHORTCUT-LEARNING FIX"
echo "  Step 1: Retrain MLP (InputLN + dynamic pos_weight + Huber)"
echo "  Step 2: Final eval (N=20, α=1.0, clamp=0.01m)"
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

CACHE_ROOT="${SLURM_TMPDIR:-/mnt/onefs/home/asahai2024/hf_cache}"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs results/eval_v4 checkpoints/eef_correction_mlp

echo ""
nvidia-smi -L || true
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 1: RETRAIN MLP  (v4 architecture)
# ══════════════════════════════════════════════════════════════
#
#  What changed vs v3:
#    1. nn.LayerNorm(4096) on raw VLA embeddings — tames activations
#    2. hidden_dim=256 hardcoded (was configurable)
#    3. Dropout=0.3 everywhere (was 0.3/0.2)
#    4. BCEWithLogitsLoss with DYNAMIC per-batch pos_weight
#       → kills shortcut "this task always fails" prior
#    5. HuberLoss(δ=0.1) for corrections — robust to OOD outliers
#    6. AdamW weight_decay=1e-3 (was 5e-4)

echo "STEP 1: Retraining MLP (v4 — shortcut-learning fix)..."
echo ""

# Back up old model
if [ -f checkpoints/eef_correction_mlp/best_model.pt ]; then
    cp checkpoints/eef_correction_mlp/best_model.pt \
       checkpoints/eef_correction_mlp/best_model_v3_backup.pt
    echo "  ✓ Backed up v3 model"
fi

python -u scripts/train_eef_correction_mlp.py \
    --success-data data/combined/success_rollouts.pkl \
    --failure-data data/combined/failure_rollouts.pkl \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --save-dir checkpoints/eef_correction_mlp \
    --input-noise 0.01 \
    --corr-mag-penalty 0.1 \
    --seed 42

echo ""
echo "STEP 1 COMPLETE — MLP v4 retrained"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 2: FINAL EVALUATION
# ══════════════════════════════════════════════════════════════
#
#  N=20 episodes × 10 tasks × 2 modes = 400 episodes
#  α=1.0 (trust direction, clamp bounds magnitude)
#  clamp=0.01m (1cm max perturbation → 0.2 action units)

echo "Ensuring VLA model is cached..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('moojink/openvla-7b-oft-finetuned-libero-spatial', resume_download=True)
"
echo "  ✓ Cached"
echo ""

echo "STEP 2: Final evaluation (N=20, α=1.0, clamp=0.01m)..."
echo ""

python -u scripts/eval_tuning.py \
    --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
    --mlp-checkpoint checkpoints/eef_correction_mlp/best_model.pt \
    --env libero_spatial \
    --tasks 0 1 2 3 4 5 6 7 8 9 \
    --episodes-per-task 20 \
    --alpha 1.0 \
    --ema-beta 0.7 \
    --action-scale 0.05 \
    --correction-threshold 0.005 \
    --max-correction 0.01 \
    --seed 42 \
    --save-dir results/eval_v4

echo ""
echo "================================================================"
echo "  v4 RETRAIN + EVAL COMPLETE"
echo "================================================================"
echo "MLP:     checkpoints/eef_correction_mlp/best_model.pt"
echo "Curves:  checkpoints/eef_correction_mlp/training_curves.json"
echo "Eval:    results/eval_v4/eval_results.json"
echo "Date:    $(date)"
echo "================================================================"
