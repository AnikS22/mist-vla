# MIST-VLA: Mechanistic Interpretability for Steering and Transparent VLA Failure Recovery

A snap-on module that uses mechanistic interpretability to detect, explain, and correct failures in Vision-Language-Action (VLA) models through activation steering.

## Overview

MIST-VLA combines three key components:
1. **Failure Detection**: SAFE-style detector that monitors VLA internal features
2. **Failure Attribution**: Integrated Gradients to identify WHY the VLA is failing
3. **Activation Steering**: FFN-based steering to correct failure-inducing behaviors

All without retraining the base VLA model.

## Installation

### 1. Clone Repository

```bash
git clone <this-repo>
cd mist-vla
```

### 2. Create Environment

```bash
conda create -n mist-vla python=3.10 -y
conda activate mist-vla
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10

# Flash attention
pip install flash-attn==2.5.5 --no-build-isolation

# Interpretability tools
pip install transformer-lens captum

# Install MIST-VLA
pip install -e .
```

### 4. Install LIBERO (for evaluation)

```bash
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ../mist-vla
```

### 5. Install OpenVLA

```bash
cd ..
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
cd ../mist-vla
```

## Quick Start

### 1. Collect Failure Data

```bash
python scripts/collect_failure_data.py \
  --env libero_spatial \
  --n_success 100 \
  --n_failure 100 \
  --save_dir data/rollouts
```

### 2. Train Failure Detector

```bash
python scripts/train_failure_detector.py \
  --data_dir data/rollouts \
  --detector_type mlp \
  --epochs 50
```

### 3. Extract Steering Vectors

```bash
python scripts/extract_steering_vectors.py \
  --model openvla/openvla-7b \
  --save_path data/steering_vectors
```

### 4. Run Evaluation

```bash
python scripts/run_libero_eval.py \
  --task_suite libero_spatial \
  --model_path openvla/openvla-7b \
  --detector_path checkpoints/best_detector.pt \
  --steering_path data/steering_vectors/all_vectors.pt \
  --n_episodes 50
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MIST-VLA SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input (Image + Instruction)                                 │
│          ↓                                                   │
│  ┌──────────────────┐                                        │
│  │    BASE VLA      │  (OpenVLA/pi0, frozen)                 │
│  │  + Hook Points   │                                        │
│  └────────┬─────────┘                                        │
│           ↓                                                  │
│  ┌─────────────────────────────────────┐                    │
│  │     MIST MODULE (Snap-on)           │                    │
│  │                                     │                    │
│  │  1. Failure Detector                │                    │
│  │     → Monitors latent features       │                    │
│  │                                     │                    │
│  │  2. Failure Localizer               │                    │
│  │     → Attributes failure cause       │                    │
│  │                                     │                    │
│  │  3. Activation Steerer              │                    │
│  │     → Injects corrections            │                    │
│  │                                     │                    │
│  │  4. Recovery Orchestrator           │                    │
│  │     → Coordinates all components     │                    │
│  └─────────────────────────────────────┘                    │
│           ↓                                                  │
│  Corrected Action + Explanation                             │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
mist-vla/
├── configs/              # Configuration files
├── src/                  # Source code
│   ├── models/          # VLA wrappers with hooks
│   ├── failure_detection/  # SAFE-style detector
│   ├── attribution/     # Failure localization
│   ├── steering/        # Activation steering
│   ├── recovery/        # Main orchestrator
│   └── utils/           # Utilities
├── scripts/             # Training & evaluation
├── data/                # Data and artifacts
├── experiments/         # Experiment results
└── notebooks/           # Analysis notebooks
```

## Key Components

### 1. HookedOpenVLA
Wraps OpenVLA with hook points at every FFN layer for:
- Activation caching (for monitoring)
- Activation steering (for correction)

### 2. SAFE Detector
Monitors VLA internal features to predict failure probability with:
- MLP or LSTM failure classifier
- Conformal prediction for calibrated thresholds

### 3. Failure Localizer
Uses Integrated Gradients to identify:
- Which visual patches caused failure
- Which language tokens caused failure
- Human-readable explanations

### 4. Activation Steerer
Modifies FFN activations to correct behavior:
- Extracts semantic directions from FFN weights
- Maps failure causes to steering directions
- Applies real-time activation injection

### 5. Recovery Orchestrator
Coordinates the full pipeline:
1. Detect failure
2. Attribute cause
3. Apply steering
4. Generate corrected action
5. Fuse with original for smooth motion

## Experiments

### Baseline Comparisons
- SAFE: Detection only
- FailSafe: VLM-based recovery
- FPC-VLA: Supervisor-based correction
- Vanilla VLA: No intervention

### Ablations
- Detection only (no steering)
- Steering only (no attribution)
- Random steering vs. attributed steering
- Different layer ranges for steering

### Evaluation Metrics
- Success Rate
- Recovery Rate (% of detected failures recovered)
- Detection Latency
- Inference Overhead
- Explanation Quality

## Citation

If you use this code, please cite:

```bibtex
@article{mist-vla,
  title={MIST-VLA: Mechanistic Interpretability for Steering and Transparent VLA Failure Recovery},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## References

- SAFE: arXiv 2506.09937
- VLA Steering: arXiv 2509.00328
- FailSafe: arXiv 2510.01642
- OpenVLA: arXiv 2406.09246
- LIBERO: arXiv 2306.03310

## License

MIT License
