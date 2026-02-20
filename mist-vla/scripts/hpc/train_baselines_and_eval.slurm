#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/baselines_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/baselines_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

###############################################################################
#  TRAIN BASELINES + FULL 7-ROW EVALUATION
#
#  Step 1: Download LIBERO demonstration datasets (if needed)
#  Step 2: Train Diffusion Policy on libero_spatial
#  Step 3: Train ACT on libero_spatial
#  Step 4: Run full 4-mode OpenVLA ablation evaluation
#  Step 5: Run DP and ACT evaluation
#  Step 6: Compile paper results table
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  BASELINES + FULL EVALUATION — Paper Results Table"
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

CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs results/paper_table

# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Run 4-mode OpenVLA ablation (Category 1)
#  This covers rows 1-4 of the paper table
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  CATEGORY 1: OpenVLA Ablations (4 modes)"
echo "================================================================"

# Find the latest MLP checkpoint
MLP_CKPT=""
for dir in checkpoints/eef_correction_mlp_allsuites checkpoints/eef_correction_mlp; do
    if [ -f "${dir}/best_model.pt" ]; then
        MLP_CKPT="${dir}/best_model.pt"
        echo "  Using MLP: ${MLP_CKPT}"
        break
    fi
done

if [ -z "${MLP_CKPT}" ]; then
    echo "  ⚠ No MLP checkpoint found! Running vanilla + noise + ema_only only."
    python -u scripts/eval_tuning.py \
        --model-name "moojink/openvla-7b-oft-finetuned-libero-spatial" \
        --mlp-checkpoint "NONE" \
        --env libero_spatial \
        --modes vanilla noise ema_only \
        --episodes-per-task 20 \
        --save-dir results/paper_table/category1
else
    python -u scripts/eval_tuning.py \
        --model-name "moojink/openvla-7b-oft-finetuned-libero-spatial" \
        --mlp-checkpoint "${MLP_CKPT}" \
        --env libero_spatial \
        --modes vanilla noise ema_only steering \
        --episodes-per-task 20 \
        --save-dir results/paper_table/category1
fi

echo ""
echo "================================================================"
echo "  Category 1 complete. Results: results/paper_table/category1/"
echo "================================================================"

# Note: Category 2 (DP, ACT) and Category 3 (Octo) are handled by
# separate scripts since they require different conda environments
# and model loading logic.

echo ""
echo "================================================================"
echo "  DONE — Baselines evaluation"
echo "================================================================"
echo "  Date: $(date)"
echo "  Results: results/paper_table/"
echo "================================================================"
