#!/usr/bin/env bash
# Run a fine-tuned SmolVLA checkpoint on the SO-101 (LeRobot record + policy).
#
# Usage:
#   POLICY_PATH=research_data/checkpoints/so101/smolvla_finetune/checkpoints/last/pretrained_model \
#     ./scripts/so101/eval_lerobot_policy.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-my_follower}"
WRIST_CAM="${WRIST_CAM:-0}"
SCENE_CAM="${SCENE_CAM:-2}"
NUM_EPISODES="${NUM_EPISODES:-10}"
TASK="${TASK:-pick up the black bowl and place it on the plate}"
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to your checkpoint dir (…/pretrained_model)}"

if [[ ! -d "$POLICY_PATH" && ! -f "$POLICY_PATH/config.json" ]]; then
  echo "POLICY_PATH not found: $POLICY_PATH"
  exit 1
fi

DATASET_ROOT="${DATASET_ROOT:-$REPO/research_data/lerobot_datasets}"
DATASET_REPO="${DATASET_REPO:-local/so101_smolvla_eval}"
CAMERAS="{ camera1: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, camera2: {type: opencv, index_or_path: ${SCENE_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG} }"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.use_degrees=true \
  --robot.cameras="${CAMERAS}" \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK}" \
  --dataset.episode_time_s=60 \
  --dataset.push_to_hub=false \
  --display_data=true
