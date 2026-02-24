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
#  Compares Vanilla ACT vs Jiggle vs MPPI vs Steering
#  Uses the subsampled/honest ACT MLP (256-dim input)
#
#  4 modes × 10 tasks × 20 episodes = 800 total episodes
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  ACT STEERING EVALUATION — Cross-Architecture Proof"
echo "  4 modes × 10 tasks × 20 episodes = 800 episodes"
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

RUN_TAG="${RUN_TAG:-shared_profile_v3}"
N_EPISODES="${N_EPISODES:-20}"
mkdir -p logs "results/eval_act_steering_${RUN_TAG}"

# Checkpoints
ACT_CKPT="checkpoints/act/best_model.pt"
# Use the honest/subsampled ACT MLP for fair comparison.
MLP_CKPT="checkpoints/eef_correction_mlp_act_honest/best_model.pt"

# Shared conservative steering profile (used for cross-model comparability)
ALPHA="${ALPHA:-0.15}"
EMA_BETA="${EMA_BETA:-0.9}"
MAX_CORR="${MAX_CORR:-0.004}"
CORR_THRESH="${CORR_THRESH:-0.003}"
FAIL_THRESH="${FAIL_THRESH:-0.6}"

echo "  ACT Checkpoint: ${ACT_CKPT}"
echo "  MLP Checkpoint: ${MLP_CKPT}"
echo "  Run tag: ${RUN_TAG}  episodes=${N_EPISODES}"
echo "  Shared Steering: alpha=${ALPHA} ema_beta=${EMA_BETA} max_corr=${MAX_CORR}m threshold=${CORR_THRESH}m fail_gate=p>=${FAIL_THRESH}"
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
    --modes vanilla latent_jiggle mppi steering \
    --episodes-per-task "${N_EPISODES}" \
    --alpha "${ALPHA}" \
    --ema-beta "${EMA_BETA}" \
    --max-correction "${MAX_CORR}" \
    --correction-threshold "${CORR_THRESH}" \
    --use-fail-gate \
    --fail-threshold "${FAIL_THRESH}" \
    --mppi-samples 16 \
    --mppi-temperature 5.0 \
    --save-dir "results/eval_act_steering_${RUN_TAG}"

echo ""
echo "================================================================"
echo "  ACT STEERING EVALUATION COMPLETE"
echo "================================================================"
echo "  Date: $(date)"
echo "  Results: results/eval_act_steering_${RUN_TAG}/eval_results.json"
echo "================================================================"
