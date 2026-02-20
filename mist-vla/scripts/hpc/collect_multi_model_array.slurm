#!/bin/bash
#SBATCH --job-name=collect_mm
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/collect_mm_%A_%a.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/collect_mm_%A_%a.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7
#SBATCH --array=0-4

###############################################################################
#  MULTI-MODEL DATA COLLECTION — SLURM ARRAY JOB (A100 ONLY)
#
#  Launches 5 parallel A100 jobs, one per (model, suite) pair:
#    0: OpenVLA-OFT  → libero_object
#    1: OpenVLA-OFT  → libero_goal
#    2: OpenVLA-OFT  → libero_10
#    3: OpenVLA-OFT  → libero_spatial (top-up for more data)
#    4: OpenVLA-OFT  → ALL-SUITE combined model on libero_spatial
#
#  Each job collects 50S + 50F per task ≈ 1000 rollouts per job
#  Total target: ~5000 new rollouts across 5 jobs
###############################################################################

set -eo pipefail

# ─── Model/Suite Configuration per Array Index ───
declare -a MODELS=(
    "moojink/openvla-7b-oft-finetuned-libero-object"
    "moojink/openvla-7b-oft-finetuned-libero-goal"
    "moojink/openvla-7b-oft-finetuned-libero-10"
    "moojink/openvla-7b-oft-finetuned-libero-spatial"
    "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
)
declare -a SUITES=(
    "libero_object"
    "libero_goal"
    "libero_10"
    "libero_spatial"
    "libero_spatial"
)
declare -a LABELS=(
    "openvla_oft__libero_object"
    "openvla_oft__libero_goal"
    "openvla_oft__libero_10"
    "openvla_oft__libero_spatial_topup"
    "openvla_oft_allsuite__libero_spatial"
)
declare -a N_SUCCESS=(50 50 50 40 50)
declare -a N_FAILURE=(50 50 50 40 50)

IDX=${SLURM_ARRAY_TASK_ID}
MODEL="${MODELS[$IDX]}"
SUITE="${SUITES[$IDX]}"
LABEL="${LABELS[$IDX]}"
NS="${N_SUCCESS[$IDX]}"
NF="${N_FAILURE[$IDX]}"

SAVE_DIR="data/multi_model/${LABEL}"

echo "================================================================"
echo "  MULTI-MODEL DATA COLLECTION (A100 ONLY)"
echo "================================================================"
echo "  Array Index:  ${IDX} / 4"
echo "  Label:        ${LABEL}"
echo "  Model:        ${MODEL}"
echo "  Suite:        ${SUITE}"
echo "  Target:       ${NS}S + ${NF}F per task"
echo "  Save Dir:     ${SAVE_DIR}"
echo "================================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         ${SLURM_NODELIST:-unknown}"
echo "  Date:         $(date)"
echo "  GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================================================"

# ─── Environment Setup ───
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

# *** SHARED HuggingFace cache — NOT per-job tmpdir ***
CACHE_ROOT="/mnt/onefs/home/asahai2024/hf_cache"
mkdir -p "${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"

mkdir -p logs "${SAVE_DIR}"

nvidia-smi -L || true
echo ""

# ─── Pre-cache model (shared across jobs) ───
echo "[1/2] Pre-caching model weights: ${MODEL##*/}..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', resume_download=True)
print('  ✓ Cached')
" || echo "  ⚠ Cache step had issues, continuing..."
echo ""

# ─── Run Data Collection ───
echo "[2/2] Starting data collection..."
echo "  Model: ${MODEL##*/}"
echo "  Suite: ${SUITE}"
echo "  Target: ${NS}S + ${NF}F per task"
echo ""

python -u scripts/collect_failure_data_oft_eval.py \
    --model-name "${MODEL}" \
    --env "${SUITE}" \
    --n_success "${NS}" \
    --n_failure "${NF}" \
    --max-attempts-per-task 30 \
    --save_dir "${SAVE_DIR}" \
    --checkpoint-every 50 \
    --seed "${IDX}42"

echo ""
echo "================================================================"
echo "  COLLECTION COMPLETE: ${LABEL}"
echo "================================================================"
echo "  Save Dir:  ${SAVE_DIR}"
echo "  Date:      $(date)"
ls -lh "${SAVE_DIR}/"*.pkl 2>/dev/null
echo "================================================================"
