#!/usr/bin/env bash
# Record a LeRobot dataset on the SO-101 for SmolVLA fine-tuning.
#
# Prereqs: arms calibrated (my_follower / my_leader under ~/.cache/huggingface/lerobot).
# Usage:
#   ./scripts/so101/record_dataset.sh
#   NUM_EPISODES=30 TASK="pick up the bowl and place it on the plate" ./scripts/so101/record_dataset.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-my_follower}"
LEADER_ID="${LEADER_ID:-my_leader}"
WRIST_CAM="${WRIST_CAM:-0}"
SCENE_CAM="${SCENE_CAM:-2}"
NUM_EPISODES="${NUM_EPISODES:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-10}"
FPS="${FPS:-30}"
TASK="${TASK:-pick up the black bowl and place it on the plate}"
DATASET_ROOT="${DATASET_ROOT:-$REPO/research_data/lerobot_datasets}"
DATASET_REPO="${DATASET_REPO:-local/so101_smolvla_demos}"

mkdir -p "$DATASET_ROOT"

# camera keys camera1/camera2 match SmolVLA base (wrist + scene).
CAMERAS="{ camera1: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: ${FPS}, fourcc: MJPG}, camera2: {type: opencv, index_or_path: ${SCENE_CAM}, width: 640, height: 480, fps: ${FPS}, fourcc: MJPG} }"

echo "Recording ${NUM_EPISODES} episodes → ${DATASET_ROOT}/${DATASET_REPO}"
echo "  follower ${FOLLOWER_PORT} (${FOLLOWER_ID}), leader ${LEADER_PORT} (${LEADER_ID})"
echo "  cameras wrist=${WRIST_CAM} scene=${SCENE_CAM}, task: ${TASK}"
echo ""
echo "Controls during record: see LeRobot UI / keyboard hints in terminal."
echo "Reset the scene between episodes when prompted."
echo ""

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.use_degrees=true \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}" \
  --teleop.use_degrees=true \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true

echo ""
echo "Dataset saved under: ${DATASET_ROOT}/${DATASET_REPO}"
echo "Next: ./scripts/so101/finetune_smolvla.sh"
