# PULSE: Probing Underlying Latent Safety Embeddings

**Robot policies encode safety. PULSE reads it and steers.**

A lightweight MLP probe (108K–1.1M params) that attaches to any frozen VLA or imitation policy's hidden states at inference time, detects impending failure, and applies bounded corrective steering — without retraining the base model.

**Paper:** *Robot Policies Encode Safety: Probing and Exploiting the Latent Safety Manifold for Inference-Time Steering* (targeting CoRL 2026)

---

## Key Results

| Finding | Evidence |
|---|---|
| Models encode failure risk | AUC 0.83 (OpenVLA 4096-d), 0.98 (ACT 256-d) |
| Models encode time-to-failure | r = 0.86, R² = 0.74 |
| Probe self-calibrates to difficulty | r = −0.98 (p < 10⁻⁶) intervention rate vs task difficulty |
| Steering matches MPPI at 6× less latency | 0.9ms vs 5.6ms per step |
| Generalizes zero-shot to unseen tasks | Maintains vanilla parity on held-out tasks |
| Works across architectures | Same pattern on OpenVLA (4096-d) and ACT (256-d) |

## How It Works

```
Camera Image → Frozen VLA → Hidden State h_t (4096-d) → PULSE Probe (<1ms)
                    ↓                                        ↓
              Raw Action a_t                     fail_logit, TTF, correction Δp
                    ↓                                        ↓
              Double Gate: fire only if |Δp| > τ_c AND logit > τ_f
                    ↓
              Modified Action: a_t + α · correction → Robot
```

The probe adds <1ms of latency. The base policy is never modified.

## Quick Start

### 1. Train a safety probe on your VLA

```bash
# Collect rollouts (need ~500 success + ~500 failure)
python3 scripts/collect_baseline_data.py --model your_vla --episodes 1000

# Train probe
python3 scripts/train_eef_correction_mlp.py --data data/rollouts --output checkpoints/my_probe
```

### 2. Deploy on a robot

```bash
# Start arm server (on machine connected to robot)
python3 scripts/arm_server_xarm.py --ip 192.168.1.XXX --camera 0 --port 5000

# Run with PULSE steering
python3 scripts/run_model_yahboom_loop.py \
    --jetson-host http://ARM_IP:5000 \
    --policy openvla-oft \
    --steering-checkpoint checkpoints/my_probe/best_model.pt \
    --mode steering \
    --instruction "pick up the red block"
```

### 3. Run the demo (simulation)

```bash
# LIBERO + OpenVLA-OFT + real probe (requires HPC or multi-GPU)
MUJOCO_GL=glfw python3 scripts/demo_pulse_libero.py --mode steering

# ManiSkill + xArm6 (local, lighter)
python3 scripts/demo_pulse_live.py --mode steering --no-vla
```

## Repository Structure

```
mist-vla/
├── src/
│   ├── models/              # VLA wrappers (OpenVLA, OFT, ACT, xVLA)
│   │   ├── vla_wrapper.py          # OpenVLA with hidden state hooks
│   │   ├── openvla_oft_wrapper.py  # OpenVLA-OFT wrapper
│   │   └── xvla_wrapper.py         # SmolVLA/xVLA wrapper
│   ├── steering/            # Activation steering module
│   └── training/            # Risk predictor training
├── scripts/
│   ├── train_eef_correction_mlp.py  # Train the safety probe
│   ├── eval_act_steering.py         # Evaluate ACT + steering
│   ├── eval_closed_loop_study.py    # Evaluate OpenVLA-OFT + steering
│   ├── run_model_yahboom_loop.py    # Real robot control loop
│   ├── arm_server_xarm.py          # HTTP server for UFactory xArm 6
│   ├── arm_server_kinova.py        # HTTP server for Kinova Gen3 Lite
│   ├── setup_physical_robot.sh     # Setup script for physical deployment
│   ├── test_arm_connection.py      # Connectivity test
│   ├── demo_pulse_libero.py        # LIBERO simulation demo
│   ├── demo_pulse_live.py          # ManiSkill live demo
│   └── hpc/                        # SLURM scripts for HPC
├── paper/
│   ├── main.tex                    # CoRL 2026 paper (compiles with tectonic)
│   ├── corl_2026.sty              # CoRL template
│   ├── sections/                   # Paper sections
│   ├── tables/                     # Auto-generated LaTeX tables
│   ├── figures/                    # Generated figures
│   ├── scripts/                    # Paper table/figure generation
│   │   ├── run_stat_tests.py       # Statistical analysis
│   │   ├── generate_tables.py      # LaTeX table generation
│   │   └── generate_visuals.py     # Figure generation
│   ├── data/                       # Frozen eval results (JSON)
│   └── robot_proposal.tex          # Physical robot experiment proposal
└── hpc_mirror/
    ├── checkpoints/                # Trained probe checkpoints
    └── results/                    # HPC evaluation results
```

## Physical Robot Deployment

Supported arms: **UFactory xArm 6** and **Kinova Gen3 Lite**

```bash
# 1. Setup (checks deps, network, camera — NO arm motion)
./scripts/setup_physical_robot.sh --arm xarm --arm-ip 192.168.1.100

# 2. Start arm server
python3 scripts/arm_server_xarm.py --ip 192.168.1.100 --camera 0

# 3. Test connection
python3 scripts/test_arm_connection.py --host http://localhost:5000

# 4. Run diagnostics
python3 scripts/vla_control_diagnostics.py --jetson-host http://localhost:5000

# 5. Pilot (5 episodes, stay near e-stop)
python3 scripts/run_model_yahboom_loop.py \
    --jetson-host http://localhost:5000 \
    --mode vanilla --max-steps 50

# 6. Full experiment
python3 scripts/yahboom_eval_harness.py \
    --jetson-host http://localhost:5000 \
    --modes vanilla steering mppi latent_stop latent_jiggle heuristic \
    --episodes 50
```

**Safety interlocks:** Workspace bounding (±50mm/step), velocity saturation freeze, repeated-action abort, human e-stop.

## Reproducing the Paper

```bash
cd paper/scripts
python3 run_stat_tests.py      # Statistical analysis → tables + JSON
python3 generate_tables.py     # LaTeX tables
python3 generate_visuals.py    # Figures

cd ..
tectonic main.tex              # Compile paper (CoRL 2026 format)
```

## Probe Architecture

```
Input (m-d) → LayerNorm → Linear(256) → GELU → Dropout(0.3)
                        → Linear(128) → GELU → Dropout(0.3)
                        → Linear(64)  → GELU → Dropout(0.3)
                        → 3 heads: fail_logit | TTF | correction (3-d)
```

- **108K params** for ACT (256-d input)
- **1.1M params** for OpenVLA (4096-d input)
- **<1ms inference** on GPU

## Important: Logit Threshold Calibration

The probe outputs raw logits, not probabilities. Each model has different logit ranges:

| Model | Success logits | Failure logits | Threshold |
|---|---|---|---|
| OpenVLA-OFT (4096-d) | mean 16.09 | mean 17.03 | 16.56 |
| ACT (256-d) | mean −2.55 | mean 5.78 | 1.61 |

**Always calibrate the threshold for your specific model** using `scripts/fix_probe_threshold.py`.

## Citation

```bibtex
@inproceedings{pulse2026,
  title={Robot Policies Encode Safety: Probing and Exploiting the Latent Safety Manifold for Inference-Time Steering},
  author={Sahai, Anik and Nojoumian, Mehrdad and Hahn, William},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```

## License

Research use. See LICENSE for details.
