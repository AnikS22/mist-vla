# PULSE Physical Robot Deployment Guide

## Overview

PULSE attaches to any VLA that exposes hidden states. The deployment pipeline:

```
Camera → VLA (frozen) → hidden state → PULSE probe → gated correction → robot arm
```

## Prerequisites

- Robot arm: UFactory xArm 6 or Kinova Gen3 Lite (or any arm with Cartesian control API)
- GPU server: RTX 3090/4090 or better (for VLA inference)
- Wrist-mounted RGB camera (640×480)
- Network connection between GPU server and arm controller
- Trained PULSE probe checkpoint (train on ~1000 rollouts)

## Step-by-Step

### 1. Setup (safe — no arm motion)

```bash
./scripts/setup_physical_robot.sh --arm xarm --arm-ip 192.168.1.100 --camera 0
```

This checks: Python deps, network, camera, probe checkpoint, dry-run inference.

### 2. Start arm server

On the machine connected to the arm:

```bash
# xArm 6
python3 scripts/arm_server_xarm.py --ip 192.168.1.100 --camera 0 --port 5000

# Kinova Gen3 Lite
python3 scripts/arm_server_kinova.py --ip 192.168.1.10 --camera 0 --port 5000
```

This exposes HTTP endpoints:
- `GET /status` → `{ok, coords: [x,y,z,rx,ry,rz]}`
- `POST /action` → `{action: "move_to", coords: [...], speed: int}`
- `GET /snapshot` → JPEG bytes

### 3. Test connection

```bash
python3 scripts/test_arm_connection.py --host http://ARM_IP:5000
```

Tests: status, camera, POST, small 5mm move (and back).

### 4. VLA diagnostics

```bash
python3 scripts/vla_control_diagnostics.py --jetson-host http://ARM_IP:5000
```

Checks that the VLA produces non-degenerate actions on real camera input.

### 5. Pilot run

```bash
python3 scripts/run_model_yahboom_loop.py \
    --jetson-host http://ARM_IP:5000 \
    --policy openvla-oft \
    --instruction "pick up the red block" \
    --mode vanilla \
    --max-steps 50
```

**STAY NEAR E-STOP. Start with 50 steps, not 300.**

### 6. Collect training data

Run ~500 success + ~500 failure episodes in vanilla mode. Log hidden states:

```bash
python3 scripts/run_model_yahboom_loop.py \
    --jetson-host http://ARM_IP:5000 \
    --policy openvla-oft \
    --mode vanilla \
    --max-steps 300 \
    --log-hidden-states \
    --output data/real_rollouts/
```

### 7. Train probe on real data

```bash
python3 scripts/train_eef_correction_mlp.py \
    --data data/real_rollouts/ \
    --output checkpoints/real_probe/
```

### 8. Calibrate threshold

```bash
python3 scripts/fix_probe_threshold.py \
    --checkpoint checkpoints/real_probe/best_model.pt \
    --data data/real_rollouts/
```

This outputs the calibrated logit threshold for your specific model + real data.

### 9. Full experiment

```bash
python3 scripts/yahboom_eval_harness.py \
    --jetson-host http://ARM_IP:5000 \
    --modes vanilla steering mppi latent_stop latent_jiggle heuristic \
    --episodes 50 \
    --steering-checkpoint checkpoints/real_probe/best_model.pt \
    --steering-fail-threshold YOUR_CALIBRATED_THRESHOLD
```

## Adapting to a New VLA

To use PULSE with a VLA not currently supported:

1. Write a wrapper that implements `get_action_with_features(image, instruction) → (action, hidden_state)`
2. The hidden state must be a 1-D tensor/array (any dimension)
3. Collect rollouts, train probe, calibrate threshold
4. Done — the probe architecture adapts to any input dimension

## Safety

- **Workspace bounding:** ±50mm XY, ±30mm Z per step (configurable)
- **Velocity saturation:** Freeze if action L∞ > 0.95
- **Stuck detection:** Abort after 4 identical consecutive actions
- **E-stop:** Human must be within arm's reach at all times
- **The probe is a guardrail, not a guarantee.** It does not replace hardware safety interlocks.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Risk = 1.0 on every step | Wrong threshold (sigmoid saturation) | Calibrate with `fix_probe_threshold.py` |
| Gate fires 0% | Threshold too high for this model | Lower threshold or recalibrate |
| Gate fires 100% | Correction magnitude always above τ_c | Raise `--correction-threshold` |
| VLA produces same action every step | Action conditioning issue | Run `vla_control_diagnostics.py` |
| Arm doesn't move | Network/API issue | Run `test_arm_connection.py` |
