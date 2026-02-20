#!/bin/bash
#SBATCH --job-name=mistvla_collect
#SBATCH --output=logs/mistvla_collect_%j.out
#SBATCH --error=logs/mistvla_collect_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=shortq7-gpu
#SBATCH --exclude=nodegpu041

set -euo pipefail

echo "=== MIST-VLA Collect (OFT Eval) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2

eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs data/rollouts_oft_eval_run1

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | head -n1)"
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-}"
ls -l /dev/nvidia* || true
ls -l /dev/dri || true
nvidia-smi -L || true
nvidia-smi || true

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:/mnt/onefs/home/asahai2024/mist-vla/openvla-oft:${PYTHONPATH:-}"

python -u scripts/collect_failure_data_oft_eval.py \
  --env libero_spatial \
  --model-name moojink/openvla-7b-oft-finetuned-libero-spatial \
  --n_success 200 \
  --n_failure 400 \
  --max-attempts-per-task 50 \
  --camera-res 256 \
  --save_dir data/rollouts_oft_eval_run1 \
  --seed 0

echo "Done: $(date)"
