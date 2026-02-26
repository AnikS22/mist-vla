#!/bin/bash
#SBATCH --job-name=act_zsafety
#SBATCH --output=/mnt/onefs/home/asahai2024/mist-vla/logs/act_zsafety_%j.out
#SBATCH --error=/mnt/onefs/home/asahai2024/mist-vla/logs/act_zsafety_%j.err
#SBATCH --time=5:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=shortq7

set -eo pipefail

module purge
module load miniconda3/24.3.0-gcc-13.2.0-rslr3to
module load cuda/12.4.0-gcc-13.2.0-shyinv2
eval "$(conda shell.bash hook)"
conda activate mist-vla

cd /mnt/onefs/home/asahai2024/mist-vla
mkdir -p logs results

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

python3 - <<'PY'
import sys, torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
sys.exit(0 if ok else 2)
PY

# ---------------------------------------------------------------------------
# STRICT ZERO-SHOT TASK SPLIT
# Train MLP on TRAIN_TASKS only, evaluate on unseen TEST_TASKS only.
# ---------------------------------------------------------------------------
RUN_TAG="${RUN_TAG:-zs_act_tA01234567_tB89}"
TRAIN_TASKS="${TRAIN_TASKS:-0 1 2 3 4 5 6 7}"
TEST_TASKS="${TEST_TASKS:-8 9}"
N_EPISODES="${N_EPISODES:-20}"
SEED="${SEED:-42}"

ACT_CKPT="checkpoints/act/best_model.pt"
ZS_MLP_DIR="checkpoints/eef_correction_mlp_act_zero_shot_${RUN_TAG}"
ZS_MLP_CKPT="${ZS_MLP_DIR}/best_model.pt"
RESULT_DIR="results/eval_act_zero_shot_${RUN_TAG}"

ALPHA="${ALPHA:-0.12}"
EMA_BETA="${EMA_BETA:-0.9}"
MAX_CORR="${MAX_CORR:-0.003}"
CORR_THRESH="${CORR_THRESH:-0.002}"
FAIL_THRESH="${FAIL_THRESH:-0.65}"
STOP_THRESH="${STOP_THRESH:-0.85}"
OOD_OBSTACLE="${OOD_OBSTACLE:-1}"
OOD_STEP_MIN="${OOD_STEP_MIN:-40}"
OOD_STEP_MAX="${OOD_STEP_MAX:-160}"
OOD_DURATION="${OOD_DURATION:-20}"
OOD_PUSH_MAG="${OOD_PUSH_MAG:-0.08}"
OOD_BDDL_MAP="${OOD_BDDL_MAP:-}"
STRICT_OOD_BDDL="${STRICT_OOD_BDDL:-0}"

echo "================================================================"
echo "  ACT STRICT ZERO-SHOT GENERALIZATION"
echo "================================================================"
echo "  Job:          ${SLURM_JOB_ID}"
echo "  Train tasks:  ${TRAIN_TASKS}"
echo "  Test tasks:   ${TEST_TASKS}"
echo "  Run tag:      ${RUN_TAG}"
echo "  Date:         $(date)"
echo "================================================================"

python -u scripts/train_eef_correction_mlp.py \
    --success-data data/multi_model/act_spatial/success_rollouts.pkl \
    --failure-data data/multi_model/act_spatial/failure_rollouts.pkl \
    --save-dir "${ZS_MLP_DIR}" \
    --epochs 80 \
    --lr 5e-4 \
    --batch-size 256 \
    --input-noise 0.01 \
    --corr-mag-penalty 0.1 \
    --subsample-chunks \
    --zero-shot-train-tasks ${TRAIN_TASKS} \
    --zero-shot-test-tasks ${TEST_TASKS} \
    --seed "${SEED}"

python -u scripts/eval_act_steering.py \
    --act-checkpoint "${ACT_CKPT}" \
    --mlp-checkpoint "${ZS_MLP_CKPT}" \
    --env libero_spatial \
    --tasks ${TEST_TASKS} \
    --modes vanilla latent_stop mppi steering \
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
    --seed "${SEED}" \
    --save-dir "${RESULT_DIR}"

echo ""
echo "================================================================"
echo "  ACT ZERO-SHOT GENERALIZATION COMPLETE"
echo "================================================================"
echo "  Train report: ${ZS_MLP_DIR}/results.json"
echo "  Eval report:  ${RESULT_DIR}/eval_results.json"
echo "================================================================"
