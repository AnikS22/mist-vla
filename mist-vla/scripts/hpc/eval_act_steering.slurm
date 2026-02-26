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
#  Compares Vanilla ACT vs LatentStop vs Jiggle vs MPPI vs Steering
#  Uses the subsampled/honest ACT MLP (256-dim input)
#
#  5 modes × 10 tasks × 20 episodes = 1000 total episodes
###############################################################################

set -eo pipefail

echo "================================================================"
echo "  ACT STEERING EVALUATION — Cross-Architecture Proof"
echo "  5 modes × 10 tasks × 20 episodes = 1000 episodes"
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

# Quick CUDA check + hard fail if no GPU visible
python3 - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
sys.exit(0 if ok else 2)
PY

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
STOP_THRESH="${STOP_THRESH:-0.85}"
OOD_OBSTACLE="${OOD_OBSTACLE:-0}"
OOD_STEP_MIN="${OOD_STEP_MIN:-40}"
OOD_STEP_MAX="${OOD_STEP_MAX:-160}"
OOD_DURATION="${OOD_DURATION:-20}"
OOD_PUSH_MAG="${OOD_PUSH_MAG:-0.08}"
OOD_BDDL_MAP="${OOD_BDDL_MAP:-}"
STRICT_OOD_BDDL="${STRICT_OOD_BDDL:-0}"

echo "  ACT Checkpoint: ${ACT_CKPT}"
echo "  MLP Checkpoint: ${MLP_CKPT}"
echo "  Run tag: ${RUN_TAG}  episodes=${N_EPISODES}"
echo "  Shared Steering: alpha=${ALPHA} ema_beta=${EMA_BETA} max_corr=${MAX_CORR}m threshold=${CORR_THRESH}m fail_gate=p>=${FAIL_THRESH}"
echo "  Latent Stop: stop_threshold=${STOP_THRESH}"
echo "  OOD obstacle: enable=${OOD_OBSTACLE} step=[${OOD_STEP_MIN},${OOD_STEP_MAX}] duration=${OOD_DURATION} push=${OOD_PUSH_MAG}"
echo "  OOD BDDL map: ${OOD_BDDL_MAP:-<none>}  strict=${STRICT_OOD_BDDL}"
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
    --modes vanilla latent_stop latent_jiggle mppi steering \
    --episodes-per-task "${N_EPISODES}" \
    --alpha "${ALPHA}" \
    --ema-beta "${EMA_BETA}" \
    --max-correction "${MAX_CORR}" \
    --correction-threshold "${CORR_THRESH}" \
    --use-fail-gate \
    --fail-threshold "${FAIL_THRESH}" \
    --stop-threshold "${STOP_THRESH}" \
    --mppi-samples 16 \
    --mppi-temperature 5.0 \
    $( [ "${OOD_OBSTACLE}" = "1" ] && echo "--ood-obstacle --ood-step-min ${OOD_STEP_MIN} --ood-step-max ${OOD_STEP_MAX} --ood-duration ${OOD_DURATION} --ood-push-magnitude ${OOD_PUSH_MAG}" ) \
    $( [ -n "${OOD_BDDL_MAP}" ] && echo "--ood-bddl-map ${OOD_BDDL_MAP}" ) \
    $( [ "${STRICT_OOD_BDDL}" = "1" ] && echo "--strict-ood-bddl" ) \
    --save-dir "results/eval_act_steering_${RUN_TAG}"

echo ""
echo "================================================================"
echo "  ACT STEERING EVALUATION COMPLETE"
echo "================================================================"
echo "  Date: $(date)"
echo "  Results: results/eval_act_steering_${RUN_TAG}/eval_results.json"
echo "================================================================"
