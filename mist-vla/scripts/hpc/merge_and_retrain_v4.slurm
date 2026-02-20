#!/bin/bash
#SBATCH --job-name=merge_train
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/merge_train_v4_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/merge_train_v4_%j.err
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

###############################################################################
#  MERGE ALL DATA + RETRAIN v4 MLP ON ALL 4 SUITES
#
#  Step 1: Merge all existing data (combined + multi_suite) → merged_all/
#  Step 2: Train v4 MLP (LayerNorm, GELU, HuberLoss, dynamic pos_weight)
#  Step 3: Run closed-loop evaluation on libero_spatial
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  MERGE + RETRAIN v4 MLP — ALL SUITES"
echo "================================================================"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    ${SLURM_NODELIST:-unknown}"
echo "  Date:    $(date)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

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

# Shared HF cache
CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs data/merged_all checkpoints/eef_correction_mlp_allsuites

# ─── STEP 1: MERGE ALL DATA ───
echo ""
echo "================================================================"
echo "  STEP 1/3: MERGE DATA"
echo "================================================================"

python -u scripts/merge_multi_model_data.py \
    --output data/merged_all \
    --deduplicate

echo ""
echo "  Merged data sizes:"
ls -lh data/merged_all/*.pkl 2>/dev/null

# ─── STEP 2: TRAIN v4 MLP ───
echo ""
echo "================================================================"
echo "  STEP 2/3: TRAIN v4 MLP (ALL SUITES)"
echo "================================================================"

python -u scripts/train_eef_correction_mlp.py \
    --success-data data/merged_all/success_rollouts.pkl \
    --failure-data data/merged_all/failure_rollouts.pkl \
    --save-dir checkpoints/eef_correction_mlp_allsuites \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --input-noise 0.01 \
    --corr-mag-penalty 0.1 \
    --seed 42

echo ""

# ─── STEP 3: CLOSED-LOOP EVAL ───
echo ""
echo "================================================================"
echo "  STEP 3/3: CLOSED-LOOP EVALUATION"
echo "================================================================"

# Check if eval script exists and checkpoint was produced
CKPT="checkpoints/eef_correction_mlp_allsuites/best_model.pt"
if [ -f "${CKPT}" ]; then
    echo "  Running eval with MLP: ${CKPT}"
    python -u scripts/eval_tuning.py \
        --mlp-checkpoint "${CKPT}" \
        --model-name "moojink/openvla-7b-oft-finetuned-libero-spatial" \
        --env libero_spatial \
        --episodes-per-task 20 \
        --save-dir results/allsuites_v4_eval
else
    echo "  ⚠ No checkpoint found at ${CKPT}, skipping eval"
fi

echo ""
echo "================================================================"
echo "  DONE — MERGE + TRAIN + EVAL"
echo "================================================================"
echo "  Date: $(date)"
echo "  Checkpoint: ${CKPT}"
echo "  Results: results/allsuites_v4_eval/"
echo "================================================================"
