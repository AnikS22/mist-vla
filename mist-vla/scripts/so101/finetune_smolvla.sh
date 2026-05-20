#!/usr/bin/env bash
# Fine-tune SmolVLA (~450M) on a local SO-101 LeRobot dataset.
#
# Prereqs: run record_dataset.sh first (≥20–30 demos; 50+ recommended).
# On a single RTX 2080 Ti (11 GB), batch_size=8 and ~10k–20k steps is typical.
#
# Usage:
#   ./scripts/so101/finetune_smolvla.sh
#   STEPS=15000 BATCH_SIZE=8 ./scripts/so101/finetune_smolvla.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_ROOT="${DATASET_ROOT:-$REPO/research_data/lerobot_datasets}"
DATASET_REPO="${DATASET_REPO:-local/so101_smolvla_demos}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/research_data/checkpoints/so101/smolvla_finetune}"
JOB_NAME="${JOB_NAME:-so101_smolvla}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
EVAL_FREQ="${EVAL_FREQ:-0}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_base}"

if [[ ! -d "${DATASET_ROOT}/${DATASET_REPO}" ]]; then
  echo "Dataset not found: ${DATASET_ROOT}/${DATASET_REPO}"
  echo "Run ./scripts/so101/record_dataset.sh first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Fine-tuning ${PRETRAINED}"
echo "  dataset: ${DATASET_ROOT}/${DATASET_REPO}"
echo "  output:  ${OUTPUT_DIR}"
echo "  steps=${STEPS} batch_size=${BATCH_SIZE} device=cuda"
echo ""

lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path="${PRETRAINED}" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --eval_freq="${EVAL_FREQ}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --num_workers=4

CKPT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
echo ""
echo "Training done. Checkpoint: ${CKPT}"
echo ""
echo "Closed-loop eval with your PULSE stack:"
echo "  CUDA_VISIBLE_DEVICES=0 python scripts/so101/collect_rollouts.py \\"
echo "    --mode policy --policy smolvla --policy-repo ${CKPT} \\"
echo "    --n-episodes 10"
echo ""
echo "Or native LeRobot inference:"
echo "  POLICY_PATH=${CKPT} ./scripts/so101/eval_lerobot_policy.sh"
