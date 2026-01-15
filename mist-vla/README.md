# MIST-VLA: Mechanistic Interpretability for Safe VLA Execution

A collision avoidance system for Vision-Language-Action (VLA) models using mechanistic interpretability to predict and prevent collisions through opposition-based steering.

## Key Features

- **Per-Dimension Risk Prediction**: Predict collision risk for each action dimension (x, y, z, roll, pitch, yaw, gripper)
- **Physics-Based Detection**: MuJoCo collision detection with contact geometry
- **Opposition-Based Steering**: If moving right is risky → steer left (intelligent intervention)
- **No Model Retraining**: Works as a snap-on module for any VLA

## Quick Start

### 1. Installation

```bash
# Create environment
conda create -n mist-vla python=3.10
conda activate mist-vla

# Install dependencies
pip install torch torchvision transformers
pip install mujoco robosuite
pip install scikit-learn matplotlib

# Install LIBERO benchmark
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# Install OpenVLA
pip install openvla
```

### 2. Verify Setup

```bash
python test_implementation.py
```

### 3. Run Full Pipeline

See `claude.md` for complete instructions. Quick overview:

```bash
# Collect data
python scripts/collect_phase1_data.py --num-rollouts 2000

# Compute risk labels
python scripts/compute_risk_labels.py

# Train risk predictor
python scripts/train_risk_predictor.py --epochs 50

# Extract steering vectors
python scripts/extract_steering_vectors.py

# Run evaluation
python scripts/run_evaluation.py
```

## How It Works

### 1. Per-Dimension Risk Prediction

```python
# Train MLP probe on VLA hidden states
risk_predictor: [hidden_dim=4096] → [512] → [256] → [7 risks]
```

### 2. Directional Risk Labels

```python
# Risk based on collision geometry
risk_i = max(0, action_i * collision_direction_i)
```

Example: If robot moving right (action[0]=+0.5) toward obstacle (direction[0]=+1.0):
- Risk = max(0, 0.5 × 1.0) = 0.5 (HIGH RISK)

If robot moving left (action[0]=-0.5) away from obstacle:
- Risk = max(0, -0.5 × 1.0) = 0.0 (NO RISK)

### 3. Opposition-Based Steering

```python
# If high risk detected
if risk[0] > threshold:
    if action[0] > 0:  # Moving right
        steer('left')   # Apply left steering
    else:               # Moving left
        steer('right')  # Apply right steering
```

### 4. Activation Steering

```python
# Inject steering into VLA hidden states
hidden = hidden + beta * steering_vector
```

## Project Structure

```
mist-vla/
├── src/
│   ├── data_collection/      # Hidden states & collision detection
│   ├── training/              # Risk predictor training
│   ├── steering/              # Steering vector extraction & injection
│   └── evaluation/            # Baselines & metrics
├── scripts/                   # Pipeline scripts
├── claude.md                  # Complete specification
└── test_implementation.py     # Verification tests
```

## Evaluation

We compare 5 methods:

1. **none**: Vanilla VLA (no intervention)
2. **safe_stop**: Stop when risk detected
3. **random_steer**: Random steering direction
4. **generic_slow**: Always steer 'slow'
5. **mist**: Opposition-based steering (ours)

### Metrics

- **Collision Rate** ↓: % of episodes with collisions
- **Success Rate** ↑: % of episodes completing task
- **Recovery Rate** ↑: % of risky situations recovered from

### Expected Results

MIST should achieve:
- Lowest collision rate (best safety)
- High success rate (minimal task disruption)
- Highest recovery rate (best risk mitigation)

## Hardware Requirements

- **Minimum**: 1× RTX 4090 (24GB VRAM)
- **Recommended**: 1× A100 (40GB VRAM)
- **Disk**: ~500GB for data and models

## Implementation Status

- ✅ Phase 0: Environment setup
- ✅ Phase 1: Data collection (hooks, collision detection, risk labels)
- ✅ Phase 2: Risk predictor training
- ✅ Phase 3: Steering vector extraction
- ✅ Phase 4: Steering implementation
- ✅ Phase 5: Evaluation harness

**Ready for HPC execution!**

## Documentation

- `claude.md` - Complete project specification and implementation guide
- `REAL_PROJECT_SPEC.md` - Detailed phase-by-phase specification
- `test_implementation.py` - Local verification tests

## Citation

```bibtex
@article{mistvla2024,
  title={MIST-VLA: Mechanistic Interpretability for Safe VLA Execution},
  author={Your Name},
  year={2024}
}
```

## References

- **OpenVLA**: [kim2024openvla](https://arxiv.org/abs/2406.09246)
- **LIBERO**: [liu2024libero](https://lifelong-robot-learning.github.io/LIBERO/)
- **Activation Steering**: [Turner et al. 2023](https://arxiv.org/abs/2308.10248)

## License

MIT License - See LICENSE file for details
