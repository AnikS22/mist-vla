# MIST-VLA Project Index

> **Mechanistic Interpretability for Safer Targeted Steering in Vision-Language-Action Models**
>
> Universal Latent Safety Steering: Can a lightweight MLP, using *only* a VLA's
> hidden states, predict impending failures and steer the robot's actions for safety?

Last updated: 2026-02-23

---

## Repository Layout

```
SalusV5/
├── README.md                       ← Top-level project README
├── LICENSE
├── .gitignore
├── docs/                           ← All documentation (this folder)
│   ├── README.md                   ← THIS FILE — master project index
│   ├── COMPREHENSIVE_PROJECT_REPORT.md  ← Full research report
│   ├── ARCHITECTURE.md             ← System design overview
│   ├── API.md                      ← API reference
│   ├── GETTING_STARTED.md          ← Installation & setup
│   ├── FAQ.md                      ← Common questions
│   ├── CHANGELOG.md                ← Version history
│   ├── CONTRIBUTING.md             ← Contribution guide
│   ├── DEPENDENCIES.md             ← External deps list
│   ├── SECURITY.md                 ← Security policy
│   ├── claude.md                   ← Implementation blueprint
│   └── archive/                    ← Superseded docs from earlier iterations
│
├── mist-vla/                       ← CORE PROJECT
│   ├── src/                        ← Python library (pip-installable)
│   ├── scripts/                    ← Runnable scripts & SLURM jobs
│   ├── configs/                    ← YAML/JSON experiment configs
│   ├── research_data/              ← All data, checkpoints, results
│   ├── archive/                    ← Old analysis, checkpoints, tests
│   ├── setup.py                    ← Package installer
│   └── requirements.txt            ← Pip dependencies
│
├── LIBERO/                         ← [external] LIBERO benchmark (gitignored)
├── LIBERO-PRO/                     ← [external] LIBERO-PRO (gitignored)
├── openvla/                        ← [external] OpenVLA repo (gitignored)
├── openvla-oft/                    ← [external] OpenVLA-OFT repo (gitignored)
├── openpi/                         ← [external] Pi0 / OpenPi (gitignored)
├── open-pi-zero/                   ← [external] Open Pi Zero (gitignored)
├── FailSafe_code/                  ← [external] FailSafe baseline (gitignored)
├── SAELens/                        ← [external] SAE Lens toolkit (gitignored)
├── SimplerEnv/                     ← [external] SimplerEnv (gitignored)
└── TransformerLens/                ← [external] TransformerLens (gitignored)
```

---

## Active Scripts — Quick Reference

### Training

| Script | Purpose |
|--------|---------|
| `scripts/train_eef_correction_mlp.py` | Train the EEFCorrectionMLP (v4 arch, 3-head: fail/ttf/correction). Supports `--subsample-chunks` for ACT. |
| `scripts/train_act_libero.py` | Train ACT (Action Chunking Transformer) on LIBERO demos |
| `scripts/train_diffusion_policy_libero.py` | Train Diffusion Policy on LIBERO demos |

### Data Collection

| Script | Purpose |
|--------|---------|
| `scripts/collect_baseline_data.py` | Collect rollout data from DP/ACT in SafeVLA format (features + EEF + actions) |
| `scripts/collect_octo_data.py` | Collect rollout data from Octo-Base (JAX) in SafeVLA format |
| `scripts/collect_failure_data_oft_eval.py` | Collect rollout data from OpenVLA-OFT models |
| `scripts/merge_multi_model_data.py` | Merge rollouts from multiple sources into unified dataset |

### Evaluation

| Script | Purpose |
|--------|---------|
| `scripts/eval_tuning.py` | **Primary:** OpenVLA 4-mode ablation (vanilla / noise / ema / steering) |
| `scripts/eval_act_steering.py` | **Primary:** ACT steering evaluation (vanilla vs. steering) |
| `scripts/eval_closed_loop_study.py` | Closed-loop steering evaluation (legacy, still functional) |
| `scripts/eval_baseline_model.py` | Evaluate baseline model success rates (DP/ACT) |

### Utilities

| Script | Purpose |
|--------|---------|
| `scripts/organize_data.py` | Inspect and organize collected rollout data |
| `scripts/sim_steering_math.py` | Mathematical simulation of steering corrections |
| `scripts/inspect_data_dims.py` | Inspect feature dimensions in collected data |
| `scripts/build_data_index.py` | Build data index from rollout files |

### HPC / SLURM Jobs

| Script | Purpose |
|--------|---------|
| **Data Collection** | |
| `hpc/collect_multi_model_array.slurm` | Array job: collect OpenVLA-OFT data across tasks |
| `hpc/collect_multi_suite.slurm` | Collect OpenVLA data across LIBERO suites |
| `hpc/collect_octo.slurm` | Collect Octo-Base data |
| `hpc/collect_act_data.slurm` | Collect ACT rollout data |
| `hpc/collect_dp_data.slurm` | Collect Diffusion Policy rollout data |
| `hpc/download_libero_demos.slurm` | Download LIBERO expert HDF5 demos |
| **Training** | |
| `hpc/setup_and_train_act.slurm` | Setup env + train ACT from scratch |
| `hpc/setup_and_train_dp.slurm` | Setup env + train DP from scratch |
| `hpc/train_act_mlp.slurm` | Train MLP on ACT hidden states (w/ `--subsample-chunks`) |
| `hpc/merge_and_retrain_v4.slurm` | Merge all data + retrain v4 MLP |
| `hpc/retrain_v4_and_eval.slurm` | Retrain v4 MLP + run evaluation |
| **Evaluation** | |
| `hpc/eval_paper_table.slurm` | **Primary:** 4-mode OpenVLA ablation (paper Table 1) |
| `hpc/eval_act_steering.slurm` | **Primary:** ACT vanilla vs. steering eval |
| `hpc/eval_tuning.slurm` | Hyperparameter tuning evaluation |
| `hpc/eval_closed_loop.slurm` | Closed-loop study |
| `hpc/eval_statpower.slurm` | Statistical power analysis |
| **Environment Setup** | |
| `hpc/setup_baselines.sh` | Setup DP + ACT conda envs |
| `hpc/setup_octo_env.sh` | Setup Octo conda env (JAX + CUDA) |
| **Other** | |
| `hpc/sim_steering.slurm` | Steering simulation |
| `hpc/retrain_and_sweep.slurm` | Retrain + sweep (older) |
| `hpc/train_baselines_and_eval.slurm` | Combined baseline train + eval (older) |

---

## Source Library (`src/`)

```
src/
├── data_collection/
│   ├── hooks.py               ← HiddenStateCollector — register hooks on VLA models
│   └── collision_detection.py ← Contact/collision detection in MuJoCo
├── envs/
│   └── universal_wrapper.py   ← Universal env wrapper for LIBERO
├── evaluation/
│   ├── evaluator.py           ← Evaluation runner
│   └── baselines.py           ← Baseline comparison methods
├── failure_tracking/
│   └── failure_tracker.py     ← Failure event tracking
├── mech_interp/
│   └── activation_probe.py    ← Activation probing for interpretability
├── models/
│   ├── hooked_openvla.py      ← OpenVLA with hidden-state hooks
│   ├── openvla_oft_wrapper.py ← OpenVLA-OFT wrapper
│   ├── policy_wrapper.py      ← Generic policy wrapper
│   └── vla_wrapper.py         ← VLA model wrapper base
├── steering/
│   ├── steering_module.py     ← Core steering logic
│   └── neuron_alignment.py    ← Neuron alignment analysis
└── training/
    ├── dataset.py             ← Dataset classes
    └── risk_predictor.py      ← Risk prediction models
```

---

## Research Data (`research_data/`)

### Checkpoints (trained model weights)

| Folder | Model | Notes |
|--------|-------|-------|
| `eef_correction_mlp/` | EEFCorrectionMLP v4 (OpenVLA, 4096-dim) | Primary MLP, spatial-only |
| `eef_correction_mlp_act/` | EEFCorrectionMLP v4 (ACT, 256-dim) | Cross-model evidence |
| `eef_correction_mlp_act_honest/` | Same, trained with `--subsample-chunks` | Honest eval (no feature duplication) |
| `eef_correction_mlp_allsuites/` | v4 on merged multi-suite data | All LIBERO suites |
| `act/` | ACT model weights | Trained on LIBERO spatial |
| `diffusion_policy/` | DP model weights | Trained on LIBERO spatial |
| `correction_mlp/` | Early correction MLP | Superseded by eef_correction_mlp |
| `directional_mlp/` | Directional failure MLP | Superseded |

### Results (evaluation outputs)

| Folder | What |
|--------|------|
| `results/paper_table/category1/` | OpenVLA 4-mode ablation results |
| `results/eval_v4/` | v4 MLP evaluation |
| `results/eval_act_steering/` | ACT steering evaluation |
| `results/closed_loop*/` | Closed-loop steering experiments (v1, alpha1.0, v3) |
| `results/steering_sim/` | Mathematical steering simulation |
| `results/tuning/` | Hyperparameter sweeps (clamping, gating, threshold) |

### Rollouts (collected trajectory data)

| Folder | Model | Data |
|--------|-------|------|
| `rollouts/multi_model/act_spatial/` | ACT | 499S + 500F rollouts |
| `rollouts/multi_model/dp_spatial/` | Diffusion Policy | 4S + 150F rollouts |
| `rollouts/multi_model/octo_spatial/` | Octo-Base | 0S + ~500F rollouts |
| `rollouts/multi_model/openvla_oft_allsuite__libero_spatial/` | OpenVLA-OFT | Spatial suite |
| `rollouts/openvla_spatial_seed0/` | OpenVLA-OFT | Seed 0 collection |
| `rollouts/rollouts_oft_eval_big/` | OpenVLA-OFT | 8-seed multi-task |
| `rollouts/merged_all/` | Combined | Merged dataset |

---

## Models Tested

| Model | Architecture | Hidden Dim | Status |
|-------|-------------|-----------|--------|
| OpenVLA-OFT | Autoregressive VLM | 4096 | Full data + MLP + steering eval |
| ACT | CVAE Transformer | 256 | Full data + MLP + steering eval |
| Diffusion Policy | Diffusion UNet | 256 | Trained, low SR (partial data) |
| Octo-Base | Diffusion Transformer (JAX) | 384 | 0% SR — proves universal drift |

---

## Key Architecture: EEFCorrectionMLP v4

```
Input (hidden_dim) → LayerNorm → Linear(256) → LN → GELU → Drop(0.3)
                                → Linear(128) → LN → GELU → Drop(0.3)
                                → Linear(64)  → LN → GELU → Drop(0.3)
                                ├── fail_head  → Linear(1)   [BCEWithLogitsLoss, dynamic pos_weight]
                                ├── ttf_head   → Linear(1)   [HuberLoss(δ=0.1)]
                                └── correction → Linear(3)   [HuberLoss(δ=0.1), dx/dy/dz meters]
```

- **Input:** Only hidden states (no EEF positions)
- **Optimizer:** AdamW, weight_decay=1e-3
- **Parameters:** ~108K (for 256-dim input), ~1.1M (for 4096-dim input)

---

## Archive

| Folder | Contents |
|--------|----------|
| `mist-vla/archive/old_analysis/` | Early PCA/t-SNE plots, failure attribution |
| `mist-vla/archive/old_checkpoints/` | v1/v2 MLP weights (binary_risk, directional, failure_predictor) |
| `mist-vla/archive/old_tests/` | Old test scripts |
| `mist-vla/scripts/archive/` | 34 superseded Python scripts |
| `mist-vla/scripts/hpc/archive/` | 15 superseded SLURM scripts |
| `docs/archive/` | Old documentation (training guides, status notes) |

---

## Configs

| File | Purpose |
|------|---------|
| `configs/base_config.yaml` | Base experiment configuration |
| `configs/multi_model_training.json` | Multi-model MLP training config |

---

## Quick Commands

```bash
# Train MLP on OpenVLA data
python scripts/train_eef_correction_mlp.py \
  --success-pkl research_data/rollouts/multi_model/openvla_oft_allsuite__libero_spatial/success_rollouts.pkl \
  --failure-pkl research_data/rollouts/multi_model/openvla_oft_allsuite__libero_spatial/failure_rollouts.pkl

# Train MLP on ACT data (honest, subsampled)
python scripts/train_eef_correction_mlp.py \
  --success-pkl research_data/rollouts/multi_model/act_spatial/success_rollouts.pkl \
  --failure-pkl research_data/rollouts/multi_model/act_spatial/failure_rollouts.pkl \
  --subsample-chunks

# Run OpenVLA steering eval
python scripts/eval_tuning.py \
  --model-family openvla-oft --model-path <path> \
  --mlp-checkpoint research_data/checkpoints/eef_correction_mlp/best_model.pt \
  --modes vanilla steering

# Run ACT steering eval
python scripts/eval_act_steering.py \
  --act-checkpoint research_data/checkpoints/act/best_model.pt \
  --mlp-checkpoint research_data/checkpoints/eef_correction_mlp_act_honest/best_model.pt \
  --modes vanilla steering
```
