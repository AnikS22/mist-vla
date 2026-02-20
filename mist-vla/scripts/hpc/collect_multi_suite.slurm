#!/bin/bash
#SBATCH --job-name=collect_multi
#SBATCH --output=logs/collect_multi_%j_%a.out
#SBATCH --error=logs/collect_multi_%j_%a.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --array=0-3

# ── Per-suite finetuned models from moojink (OpenVLA-OFT) ──
# Each model is finetuned on its specific LIBERO suite.
# Array mapping:
#   0 = libero_spatial  (10 tasks) — more data on existing suite
#   1 = libero_object   (10 tasks) — different objects, pick-and-place
#   2 = libero_goal     (10 tasks) — diverse goals: drawers, stove, push
#   3 = libero_10       (10 tasks) — complex multi-step tasks

set -eo pipefail

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
MODELS=(
    "moojink/openvla-7b-oft-finetuned-libero-spatial"
    "moojink/openvla-7b-oft-finetuned-libero-object"
    "moojink/openvla-7b-oft-finetuned-libero-goal"
    "moojink/openvla-7b-oft-finetuned-libero-10"
)
N_SUCCESS=(15 15 15 10)
N_FAILURE=(15 15 15 10)
MAX_ATT=(60 60 60 80)

SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
NS=${N_SUCCESS[$SLURM_ARRAY_TASK_ID]}
NF=${N_FAILURE[$SLURM_ARRAY_TASK_ID]}
MA=${MAX_ATT[$SLURM_ARRAY_TASK_ID]}

echo "===== Multi-Suite Data Collection ====="
echo "Job ID: $SLURM_JOB_ID | Array: $SLURM_ARRAY_TASK_ID"
echo "Suite: $SUITE"
echo "Model: $MODEL"
echo "Target: ${NS}S + ${NF}F per task"
echo "Max attempts/task: $MA"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Date: $(date)"
echo "========================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:$PYTHONPATH"
export HF_HOME="/mnt/onefs/home/asahai2024/.cache/huggingface"
export TRANSFORMERS_CACHE="/mnt/onefs/home/asahai2024/.cache/huggingface/hub"
cd /mnt/onefs/home/asahai2024/mist-vla

SAVE_DIR="data/multi_suite/${SUITE}"
mkdir -p logs "$SAVE_DIR"

echo "[$(date)] Starting data collection for $SUITE with $MODEL"

python scripts/collect_failure_data_oft_eval.py \
    --env "$SUITE" \
    --model-name "$MODEL" \
    --n_success "$NS" \
    --n_failure "$NF" \
    --max-attempts-per-task "$MA" \
    --save_dir "$SAVE_DIR" \
    --checkpoint-every 10 \
    --seed 42

echo ""
echo "===== Collection Complete: $SUITE ====="
echo "Data: $SAVE_DIR/"
ls -lh "$SAVE_DIR/"*.pkl 2>/dev/null
date
