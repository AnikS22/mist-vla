# SO-101 Physical Robot Pipeline

End-to-end code for the Hiwonder SO-101 leader/follower kit. Two tracks:

1. **SmolVLA fine-tune (recommended, ~450M)** — native LeRobot dataset + `lerobot-train`.
2. **PULSE probe (Pi0 / SmolVLA hiddens)** — pkl rollouts for failure-head training.

Outputs from (2) match the sim pkl schema so existing training/eval scripts work unchanged.

## Quick start: fine-tune SmolVLA on your desk

Hardware is already calibrated on this rig (`my_follower` / `my_leader` under
`~/.cache/huggingface/lerobot`). Ports: leader `ACM0`, follower `ACM1`; cameras
wrist OpenCV `0`, scene `2`; joints in **degrees** (`use_degrees=true`).

```bash
cd mist-vla

# 1) Record ~30 teleop demos (LeRobot dataset, ~1–2 hours)
./scripts/so101/record_dataset.sh
#    NUM_EPISODES=50 TASK="your task sentence" ./scripts/so101/record_dataset.sh

# 2) Fine-tune SmolVLA on GPU 0 (~2–4 h on 2080 Ti at 10k steps, batch 8)
./scripts/so101/finetune_smolvla.sh

# 3) Autonomous eval (LeRobot)
POLICY_PATH=research_data/checkpoints/so101/smolvla_finetune/checkpoints/last/pretrained_model \
  ./scripts/so101/eval_lerobot_policy.sh

# Or eval + hidden states for PULSE probe:
CUDA_VISIBLE_DEVICES=0 python scripts/so101/collect_rollouts.py \
  --mode policy --policy smolvla \
  --policy-repo research_data/checkpoints/so101/smolvla_finetune/checkpoints/last/pretrained_model \
  --n-episodes 10
```

**Tips:** Aim for ≥30 good episodes (50+ if the task has layout variation). Keep the
task sentence identical at record and eval. Set `CUDA_VISIBLE_DEVICES=0` if you have
multiple GPUs (SmolVLA must stay on one device). Community starting points:
`lerobot/smolvla_base`, `orsoromeo/smolvla_so101_pick_and_place` (use `smolvla_pp` in
`collect_rollouts.py` for that checkpoint’s camera key names).

## Hardware

- 2 × SO-101 arms (leader + follower) wired to a single BusLinker board
- Wrist camera (480p), scene camera (1080p)
- Bench: ~30 cm of clear table for the follower's workspace
- 4 × NVIDIA 2080 Ti (11 GB each)

## Software

```bash
# LeRobot 0.4.2 (already installed in this env), Pi0 weights pulled on first run
pip show lerobot   # confirm
```

Pi0 weights download automatically from HuggingFace the first time
`PI0Policy.from_pretrained("lerobot/pi0")` is called (~6 GB).

## Day-by-day workflow

### Day 1: hardware bring-up
```bash
# Verify USB enumeration
ls /dev/ttyACM*
v4l2-ctl --list-devices

# Calibrate each arm (one-time per port)
python scripts/so101/calibrate.py --follower-port /dev/ttyACM0
python scripts/so101/calibrate.py --leader-port  /dev/ttyACM1

# Smoke test: load Pi0 + verify hidden-state hook fires, no robot needed
python scripts/so101/collect_rollouts.py --dry-run
```

### Day 2: teleoperation data collection (~30 episodes)
```bash
python scripts/so101/collect_rollouts.py \
    --mode teleop --n-episodes 30 \
    --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1 \
    --wrist-cam 0 --scene-cam 2 \
    --task-id so101_pick_bowl_place_plate \
    --instruction "pick up the black bowl and place it on the plate"
```
Press ENTER between episodes; reset the scene as needed. The script saves a per-episode
`raw_epNNN.pkl` immediately (so a crash loses at most one episode).

### Day 3: label and split
```bash
python scripts/so101/label_rollouts.py \
    --run-dir research_data/rollouts/so101/<your_run_tag>
```
For each episode, label `s` (success) / `f` (failure) / `d` (discard). Output:
`success_rollouts.pkl` + `failure_rollouts.pkl` — same schema as sim data.

### Day 4: train the probe on real data
```bash
python scripts/so101/train_probe.py \
    --data-dir research_data/rollouts/so101/<your_run_tag> \
    --epochs 80
```
Auto-detects Pi0's feature dimension from the data, writes the trained probe to
`research_data/checkpoints/so101/<run_tag>/best_model.pt` + `results.json`
(AUC + precision/recall at σ ∈ {0.50, 0.80, 0.99}).

### Day 5: closed-loop evaluation
```bash
# Vanilla baseline (Pi0 alone, probe logs risk only)
python scripts/so101/eval_realtime.py \
    --controller vanilla \
    --probe research_data/checkpoints/so101/<run_tag>/best_model.pt \
    --n-episodes 20

# PULSE-as-MPPI-cost (headline experiment)
python scripts/so101/eval_realtime.py \
    --controller pulse_cost \
    --probe research_data/checkpoints/so101/<run_tag>/best_model.pt \
    --n-episodes 20
```
Each call writes `summary.json` with success rate + intervention rate.

### Day 6+: paper writeup
- Replace `figures/robot_setup_*.jpg` placeholders with actual photos
- Drop the real-robot AUC and success rates into a new
  `paper/tables/tab_so101_results.tex`
- Recompile: `cd paper && tectonic main.tex`

## Files

| File | Purpose |
|------|---------|
| `record_dataset.sh` | Teleop → LeRobot dataset for SmolVLA training |
| `finetune_smolvla.sh` | `lerobot-train` on `lerobot/smolvla_base` |
| `eval_lerobot_policy.sh` | Closed-loop eval with a fine-tuned checkpoint |
| `common.py` | Config dataclass, hidden-state hook, safety interlocks, pkl writer |
| `calibrate.py` | One-time arm calibration via LeRobot |
| `collect_rollouts.py` | Record teleop or policy rollouts with hidden states |
| `label_rollouts.py` | Interactive blind labeling → success/failure split |
| `train_probe.py` | Train `EEFCorrectionMLP` on real-robot hiddens |
| `eval_realtime.py` | Closed-loop eval: vanilla / pulse_cost / steering controllers |

## Schema compatibility

Every rollout written matches the sim pkl schema:
```python
{
  'actions': List[np.ndarray(action_dim,)],
  'features': List[np.ndarray(hidden_dim,)],
  'rewards': List[float],
  'robot_states': List[{'eef_pos', 'qpos', 'qvel'}],
  'steps': List[{'action', 'hidden_state', 'robot_state', 'collision', 'done', ...}],
  'success': bool, 'collision_occurred': bool, 'collision_step': int | None,
  'instruction': str, 'task_id': str, 'model_tag': str, 'suite_tag': 'so101_real',
}
```
This means **the existing `scripts/train_eef_correction_mlp.py` from sim training will
load these rollouts directly** — `train_probe.py` is a thin wrapper that pins
trajectory-disjoint splits and saves results to `research_data/checkpoints/so101/`.

## Safety

`WorkspaceClamp` and the e-stop SIGINT handler in `common.py` are wired into every
loop. Ctrl+C cleanly disconnects the follower (which disables torque per the
LeRobot config). Set `--workspace-bounds` if your bench geometry differs from the
defaults (x: 0.05–0.45 m, y: ±0.30 m, z: 0–0.40 m).

## Known gaps (TODOs)

- **EEF position**: `extract_qpos_eef` falls back to a coarse joint surrogate if the
  SO-101 follower object doesn't expose `get_end_effector_position()`. For the paper
  this is fine (probe trains on hiddens; eef_pos is only logged for diagnostics) but
  if you want millimeter-accurate eef-trajectory plots, plug in a URDF FK call here.
- **MPPI cost**: `apply_pulse_cost` uses a lightweight perturbation-and-pick heuristic
  rather than a full simulator-rollout MPPI (which would require a sim of the real
  robot in the loop). The headline latency claim (<2 ms/step) is reproducible
  because the heuristic is genuinely cheap; the success-rate claim should be
  validated against vanilla on the same task set before reporting.
- **Camera streaming latency**: at 30 Hz with two OpenCV cameras some frames will
  drop. The pkl rate gets paced to `--fps`, so this affects responsiveness, not
  rollout integrity.
