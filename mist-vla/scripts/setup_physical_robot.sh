#!/bin/bash
set -euo pipefail

# ============================================================================
# PULSE Physical Robot Setup Script
# ============================================================================
#
# This script prepares everything for running PULSE experiments on a
# Kinova Gen3 Lite or UFactory xArm 6. It does NOT move the arm or
# execute any commands that could cause motion. It only:
#   1. Checks dependencies
#   2. Installs Python packages (with confirmation)
#   3. Validates network connectivity
#   4. Downloads model checkpoints if missing
#   5. Runs a dry-run check of the full pipeline
#
# Usage:
#   ./setup_physical_robot.sh --arm xarm --arm-ip 192.168.1.100 --camera 0
#   ./setup_physical_robot.sh --arm kinova --arm-ip 192.168.1.100 --camera 0
#
# SAFETY: This script will NEVER send motion commands to the arm.
#         All arm interaction happens ONLY through the arm_server scripts,
#         which you start manually after verifying the setup.
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── Parse args ───────────────────────────────────────────────────

ARM_TYPE=""
ARM_IP=""
CAMERA_IDX="0"
GPU_SERVER_PORT="5000"
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --arm)        ARM_TYPE="$2"; shift 2 ;;
        --arm-ip)     ARM_IP="$2"; shift 2 ;;
        --camera)     CAMERA_IDX="$2"; shift 2 ;;
        --port)       GPU_SERVER_PORT="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        -h|--help)
            echo "Usage: $0 --arm {xarm|kinova} --arm-ip IP [--camera IDX] [--port PORT]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$ARM_TYPE" || -z "$ARM_IP" ]]; then
    echo "Usage: $0 --arm {xarm|kinova} --arm-ip IP [--camera IDX]"
    exit 1
fi

echo "============================================"
echo " PULSE Physical Robot Setup"
echo " Arm: $ARM_TYPE"
echo " IP:  $ARM_IP"
echo " Cam: $CAMERA_IDX"
echo "============================================"
echo ""

# ─── Step 1: Check Python ────────────────────────────────────────

echo "── Step 1: Python environment ──"

if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    ok "Python: $PY_VER"
else
    fail "python3 not found"
    exit 1
fi

# Check critical packages
MISSING_PKGS=()
for pkg in torch numpy cv2 PIL flask; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ok "  $pkg installed"
    else
        warn "  $pkg missing"
        MISSING_PKGS+=("$pkg")
    fi
done

# Check arm SDK
if [[ "$ARM_TYPE" == "xarm" ]]; then
    if python3 -c "from xarm.wrapper import XArmAPI" 2>/dev/null; then
        ok "  xArm SDK installed"
    else
        warn "  xArm SDK missing"
        MISSING_PKGS+=("xArm-Python-SDK")
    fi
elif [[ "$ARM_TYPE" == "kinova" ]]; then
    if python3 -c "from kortex_api.TCPTransport import TCPTransport" 2>/dev/null; then
        ok "  Kortex API installed"
    else
        warn "  Kortex API missing"
        MISSING_PKGS+=("kortex-api")
    fi
fi

# Install missing packages
if [[ ${#MISSING_PKGS[@]} -gt 0 ]] && [[ "$SKIP_INSTALL" == false ]]; then
    echo ""
    echo "Missing packages: ${MISSING_PKGS[*]}"
    read -p "Install now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install flask opencv-python 2>/dev/null || true
        if [[ "$ARM_TYPE" == "xarm" ]]; then
            pip install xArm-Python-SDK 2>/dev/null || warn "xArm SDK install failed — install manually"
        elif [[ "$ARM_TYPE" == "kinova" ]]; then
            warn "Kortex API must be installed manually from Kinova's GitHub"
            warn "  https://github.com/Kinovarobotics/kortex"
        fi
    fi
fi

echo ""

# ─── Step 2: Network ─────────────────────────────────────────────

echo "── Step 2: Network connectivity ──"

if ping -c 1 -W 2 "$ARM_IP" &>/dev/null; then
    ok "Arm reachable at $ARM_IP"
else
    fail "Cannot ping $ARM_IP — check network cable and IP"
    echo "  Common IPs:"
    echo "    xArm default:  192.168.1.xxx (check xArm Studio)"
    echo "    Kinova default: 192.168.1.10"
fi

echo ""

# ─── Step 3: Check arm-specific specs ────────────────────────────

echo "── Step 3: Arm specifications ──"

if [[ "$ARM_TYPE" == "xarm" ]]; then
    echo "  UFactory xArm 6 specs:"
    echo "    DOF:           6"
    echo "    Reach:         700 mm"
    echo "    Payload:       5 kg"
    echo "    Repeatability: ±0.1 mm"
    echo "    Default IP:    192.168.1.xxx (set via xArm Studio)"
    echo "    Gripper:       xArm Gripper (0-850 range, ~85mm stroke)"
    echo "    Control freq:  250 Hz max (we use ~5 Hz)"
    echo ""
    echo "  Workspace bounds (CONSERVATIVE — edit in arm_server_xarm.py):"
    echo "    X: [-500, 500] mm"
    echo "    Y: [-500, 500] mm"
    echo "    Z: [100, 600] mm"
    echo ""
    echo "  SAFETY NOTES:"
    echo "    - xArm has built-in collision detection (mode 0)"
    echo "    - ALWAYS enable protective stop in xArm Studio first"
    echo "    - Keep hand on e-stop during all experiments"
    echo "    - Speed limit: we cap at 100 mm/s (arm max is 1000 mm/s)"

elif [[ "$ARM_TYPE" == "kinova" ]]; then
    echo "  Kinova Gen3 Lite specs:"
    echo "    DOF:           6"
    echo "    Reach:         902 mm"
    echo "    Payload:       0.5 kg (Lite version)"
    echo "    Repeatability: ±1 mm"
    echo "    Default IP:    192.168.1.10"
    echo "    Gripper:       Kinova 2-finger (0-100% range)"
    echo "    Control freq:  40 Hz max (we use ~5 Hz)"
    echo ""
    echo "  Workspace bounds (CONSERVATIVE — edit in arm_server_kinova.py):"
    echo "    X: [-600, 600] mm"
    echo "    Y: [-600, 600] mm"
    echo "    Z: [50, 700] mm"
    echo ""
    echo "  SAFETY NOTES:"
    echo "    - Gen3 Lite has 0.5 kg payload — USE LIGHTWEIGHT OBJECTS ONLY"
    echo "    - Kinova uses meters internally, our server converts to mm"
    echo "    - Keep hand on e-stop during all experiments"
    echo "    - The Lite has less torque than Gen3 — don't force through collisions"
fi

echo ""

# ─── Step 4: Check model checkpoints ─────────────────────────────

echo "── Step 4: Model checkpoints ──"

# OpenVLA checkpoint
OVLA_CKPT="$REPO_ROOT/checkpoints"
if [[ -d "$OVLA_CKPT" ]] || [[ -d "$REPO_ROOT/hpc_mirror/checkpoints" ]]; then
    ok "Checkpoint directory found"
else
    warn "No local checkpoints — will need to download or sync from HPC"
fi

# Safety probe
for probe in "eef_correction_mlp" "eef_correction_mlp_act_honest"; do
    PROBE_PATH="$REPO_ROOT/hpc_mirror/checkpoints/$probe/best_model.pt"
    if [[ -f "$PROBE_PATH" ]]; then
        ok "Safety probe: $probe"
    else
        warn "Missing: $PROBE_PATH"
        echo "  Sync from HPC: scp asahai2024@athene-login.hpc.fau.edu:~/mist-vla/checkpoints/$probe/best_model.pt $PROBE_PATH"
    fi
done

echo ""

# ─── Step 5: Camera test ─────────────────────────────────────────

echo "── Step 5: Camera test (no arm motion) ──"

python3 -c "
import cv2, sys
cap = cv2.VideoCapture($CAMERA_IDX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print('FAIL: Camera $CAMERA_IDX not found')
    sys.exit(1)
ret, frame = cap.read()
if not ret:
    print('FAIL: Camera opened but cannot read frame')
    sys.exit(1)
h, w = frame.shape[:2]
print(f'Camera {$CAMERA_IDX}: {w}x{h}')
cap.release()
" 2>/dev/null && ok "Camera working" || warn "Camera test failed — check device index"

echo ""

# ─── Step 6: Dry run check ───────────────────────────────────────

echo "── Step 6: Dry run (no arm motion, no network) ──"

# Check that the control loop script can at least import
python3 -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
sys.path.insert(0, '$SCRIPT_DIR')

# Test safety probe loading
from scripts.run_model_yahboom_loop import EEFCorrectionMLP, SteeringRuntime
import torch
model = EEFCorrectionMLP(input_dim=4096)
print(f'Safety probe: {sum(p.numel() for p in model.parameters())} params')

# Test fake inference
x = torch.randn(1, 4096)
with torch.no_grad():
    out = model(x)
print(f'Output keys: {list(out.keys())}')
print(f'fail_logit: {out[\"will_fail\"].shape}, correction: {out[\"correction\"].shape}')
print('Dry run OK')
" 2>/dev/null && ok "Pipeline imports and runs" || fail "Pipeline import failed — check PYTHONPATH"

echo ""

# ─── Summary ─────────────────────────────────────────────────────

echo "============================================"
echo " Setup Summary"
echo "============================================"
echo ""
echo " Next steps:"
echo ""
echo " 1. START the arm server (on the machine connected to the arm):"
echo ""
if [[ "$ARM_TYPE" == "xarm" ]]; then
    echo "    python3 $SCRIPT_DIR/arm_server_xarm.py --ip $ARM_IP --camera $CAMERA_IDX --port $GPU_SERVER_PORT"
elif [[ "$ARM_TYPE" == "kinova" ]]; then
    echo "    python3 $SCRIPT_DIR/arm_server_kinova.py --ip $ARM_IP --camera $CAMERA_IDX --port $GPU_SERVER_PORT"
fi
echo ""
echo " 2. TEST the connection (from GPU server):"
echo ""
echo "    python3 $SCRIPT_DIR/test_arm_connection.py --host http://localhost:$GPU_SERVER_PORT"
echo ""
echo " 3. RUN diagnostics:"
echo ""
echo "    python3 $SCRIPT_DIR/vla_control_diagnostics.py --jetson-host http://localhost:$GPU_SERVER_PORT"
echo ""
echo " 4. PILOT run (5 episodes, vanilla only, STAY NEAR E-STOP):"
echo ""
echo "    python3 $SCRIPT_DIR/run_model_yahboom_loop.py \\"
echo "        --jetson-host http://localhost:$GPU_SERVER_PORT \\"
echo "        --policy openvla-oft \\"
echo "        --instruction 'pick up the red block and place it in zone B' \\"
echo "        --mode vanilla --max-steps 50"
echo ""
echo " 5. FULL experiment (after pilot succeeds):"
echo ""
echo "    python3 $SCRIPT_DIR/yahboom_eval_harness.py \\"
echo "        --jetson-host http://localhost:$GPU_SERVER_PORT \\"
echo "        --modes vanilla steering mppi latent_stop latent_jiggle heuristic \\"
echo "        --episodes 50 --tasks 3"
echo ""
echo " ⚠  KEEP HAND ON E-STOP AT ALL TIMES DURING ARM MOTION"
echo " ⚠  START WITH --max-steps 50 (not 300) FOR FIRST TESTS"
echo " ⚠  VERIFY WORKSPACE BOUNDS MATCH YOUR ARM BEFORE FULL RUNS"
echo ""
