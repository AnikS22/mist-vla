#!/bin/bash
#SBATCH --job-name=act_steer_eval
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/act_steer_eval_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/act_steer_eval_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

###############################################################################
#  ACT STEERING EVALUATION — Category 2: Cross-Architecture Proof
#
#  Compares Vanilla ACT vs ACT + Latent Steering
#  Uses the subsampled/honest ACT MLP (256-dim input)
#
#  2 modes × 10 tasks × 20 episodes = 400 total episodes
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  ACT STEERING EVALUATION — Cross-Architecture Proof"
echo "  2 modes × 10 tasks × 20 episodes = 400 episodes"
echo "================================================================"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Node:    ${SLURM_NODELIST:-unknown}"
echo "  Date:    $(date)"
echo "================================================================"

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2
eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla

# GPU setup
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi -L 2>/dev/null || true

# Quick CUDA check
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')" 2>&1 || true

export PYTHONPATH="/mnt/onefs/home/asahai2024/mist-vla:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true

mkdir -p logs results/eval_act_steering

# Checkpoints
ACT_CKPT="checkpoints/act/best_model.pt"
MLP_CKPT="checkpoints/eef_correction_mlp_act/best_model.pt"

echo "  ACT Checkpoint: ${ACT_CKPT}"
echo "  MLP Checkpoint: ${MLP_CKPT}"
echo ""

# Verify checkpoints exist
if [ ! -f "${ACT_CKPT}" ]; then
    echo "ERROR: ACT checkpoint not found: ${ACT_CKPT}"
    exit 1
fi
if [ ! -f "${MLP_CKPT}" ]; then
    echo "ERROR: MLP checkpoint not found: ${MLP_CKPT}"
    exit 1
fi

python -u scripts/eval_act_steering.py \
    --act-checkpoint "${ACT_CKPT}" \
    --mlp-checkpoint "${MLP_CKPT}" \
    --env libero_spatial \
    --modes vanilla steering \
    --episodes-per-task 20 \
    --alpha 1.0 \
    --max-correction 0.01 \
    --correction-threshold 0.005 \
    --save-dir results/eval_act_steering

echo ""
echo "================================================================"
echo "  ACT STEERING EVALUATION COMPLETE"
echo "================================================================"
echo "  Date: $(date)"
echo "  Results: results/eval_act_steering/eval_results.json"
echo "================================================================"
