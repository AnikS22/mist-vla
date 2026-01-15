# MIST-VLA: Mechanistic Interpretability for Safe VLA Execution

## Project Overview

**What This Is:** A collision avoidance system for Vision-Language-Action (VLA) models using mechanistic interpretability to predict and prevent collisions in robotic manipulation tasks.

**Key Innovation:** Per-dimension risk prediction with opposition-based steering (if moving right is risky → steer left).

## Core Components

### 1. Per-Dimension Risk Prediction
- Predict collision risk for each of 7 action dimensions: `[x, y, z, roll, pitch, yaw, gripper]`
- Uses MLP probe trained on VLA hidden states
- Target: AUC > 0.75 per dimension

### 2. MuJoCo-Based Collision Detection
- Physics-based collision detection using `sim.data.ncon` and `sim.data.contact`
- Distinguishes valid contacts (gripper-object) from collisions (robot-obstacle)

### 3. Directional Risk Labels
- Computed as: `risk_i = max(0, action_i * collision_direction_i)`
- Where `collision_direction` is vector from end-effector to collision point
- Within K=10 steps before collision

### 4. Opposition-Based Steering
- If action[0] > 0 (moving right) is risky → apply 'left' steering
- If action[0] < 0 (moving left) is risky → apply 'right' steering
- Similar for y (forward/backward) and z (up/down)

## Implementation Phases

### Phase 0: Environment Setup ✓
**Files:** `scripts/verify_phase0.py`

Setup conda environment with:
- LIBERO benchmark
- OpenVLA (7B model)
- MuJoCo physics engine
- PyTorch, Transformers

### Phase 1: Data Collection ✓
**Files:**
- `src/data_collection/hooks.py` - Hidden state collection from VLA
- `src/data_collection/collision_detection.py` - MuJoCo collision detection
- `scripts/collect_phase1_data.py` - Rollout collection script
- `scripts/compute_risk_labels.py` - Risk label computation

**Output:** `data/phase1/labeled_data.pkl` (~2000 trajectories)

**Key Code:**
```python
# Collect hidden states
collector = HiddenStateCollector(model)
with collector:
    action = model.predict_action(image, instruction)
    hidden = collector.get_last_layer()  # [1, 4096]

# Detect collisions
detector = CollisionDetector(env)
has_collision, collision_pos = detector.check_collision()

# Compute risk labels
risk_i = max(0, action_i * collision_direction_i)
```

### Phase 2: Train Risk Predictor ✓
**Files:**
- `src/training/dataset.py` - PyTorch dataset for risk prediction
- `src/training/risk_predictor.py` - MLP probe (4096 → 512 → 256 → 7)
- `scripts/train_risk_predictor.py` - Training script

**Output:** `models/risk_predictor/best_model.pt`

**Success Criteria:** Per-dimension AUC > 0.75

**Architecture:**
```
Input: [batch, 4096] hidden states
↓
Linear(4096 → 512) + ReLU + Dropout
↓
Linear(512 → 256) + ReLU + Dropout
↓
Linear(256 → 7) + ReLU
↓
Output: [batch, 7] per-dimension risk
```

### Phase 3: Extract Steering Vectors ✓
**Files:**
- `src/steering/neuron_alignment.py` - Token-neuron alignment extraction
- `scripts/extract_steering_vectors.py` - Steering vector extraction

**Output:** `data/phase3/steering_vectors.pkl`

**Method:**
1. Project FFN neurons onto token embedding space
2. Find neurons that activate for directional concepts (left, right, up, down, slow, fast)
3. Aggregate top-k neuron activation vectors as steering vectors

**Concepts:** left, right, up, down, forward, backward, slow, fast, stop

### Phase 4: Steering Implementation ✓
**Files:**
- `src/steering/steering_module.py` - Activation steering with hooks

**Key Features:**
- Register forward hooks on transformer layers
- Inject steering vectors: `hidden = hidden + beta * steering_vector`
- Opposition-based concept selection

**Usage:**
```python
steerer = SteeringModule(model, steering_vectors, target_layer=20)

# Risk-based steering
steerer.set_steering_from_risk(risk_vector, action, beta=1.0)
with steerer:
    new_action = model.predict_action(image, instruction)
```

### Phase 5: Evaluation ✓
**Files:**
- `src/evaluation/baselines.py` - 5 baseline methods
- `src/evaluation/evaluator.py` - Evaluation harness
- `scripts/run_evaluation.py` - Main evaluation script

**Baselines:**
1. **none**: No intervention (vanilla VLA)
2. **safe_stop**: Set actions to zero when risk detected
3. **random_steer**: Random steering concept
4. **generic_slow**: Always apply 'slow' steering
5. **mist**: Opposition-based steering (our method)

**Metrics:**
- **Collision Rate:** % of episodes with collisions (↓ better)
- **Success Rate:** % of episodes completing task (↑ better)
- **Recovery Rate:** % of risky situations recovered from (↑ better)

**Expected Results:**
- MIST should have lowest collision rate
- MIST should maintain high success rate
- MIST should have highest recovery rate

## Directory Structure

```
mist-vla/
├── src/
│   ├── data_collection/
│   │   ├── hooks.py                    # Hidden state collection
│   │   ├── collision_detection.py      # MuJoCo collision detection
│   │   └── __init__.py
│   ├── training/
│   │   ├── dataset.py                  # Risk prediction dataset
│   │   ├── risk_predictor.py           # MLP probe model
│   │   └── __init__.py
│   ├── steering/
│   │   ├── neuron_alignment.py         # Token-neuron alignments
│   │   ├── steering_module.py          # Activation steering
│   │   └── __init__.py
│   └── evaluation/
│       ├── baselines.py                # 5 baseline methods
│       ├── evaluator.py                # Evaluation harness
│       └── __init__.py
├── scripts/
│   ├── verify_phase0.py                # Environment verification
│   ├── collect_phase1_data.py          # Data collection
│   ├── compute_risk_labels.py          # Risk label computation
│   ├── train_risk_predictor.py         # Risk predictor training
│   ├── extract_steering_vectors.py     # Steering vector extraction
│   └── run_evaluation.py               # Full evaluation
├── data/
│   ├── phase1/                         # Collected trajectories
│   └── phase3/                         # Steering vectors
├── models/
│   └── risk_predictor/                 # Trained models
└── results/
    └── evaluation/                     # Evaluation results
```

## Running the Full Pipeline

### 1. Verify Environment
```bash
python scripts/verify_phase0.py
```

### 2. Collect Data
```bash
python scripts/collect_phase1_data.py \
    --output data/phase1/rollouts.pkl \
    --num-rollouts 2000 \
    --benchmark libero_spatial
```

### 3. Compute Risk Labels
```bash
python scripts/compute_risk_labels.py \
    --input data/phase1/rollouts.pkl \
    --output data/phase1/labeled_data.pkl \
    --K 10
```

### 4. Train Risk Predictor
```bash
python scripts/train_risk_predictor.py \
    --data data/phase1/labeled_data.pkl \
    --output-dir models/risk_predictor \
    --epochs 50 \
    --batch-size 256
```

### 5. Extract Steering Vectors
```bash
python scripts/extract_steering_vectors.py \
    --output data/phase3/steering_vectors.pkl \
    --layers 16 20 24 \
    --top-k 20
```

### 6. Run Evaluation
```bash
python scripts/run_evaluation.py \
    --risk-predictor models/risk_predictor/best_model.pt \
    --steering-vectors data/phase3/steering_vectors.pkl \
    --output-dir results/evaluation \
    --num-episodes 10 \
    --baselines none safe_stop random_steer generic_slow mist
```

## HPC Setup

For running on HPC (e.g., FAU Athene):

```bash
# Transfer code to HPC
rsync -avz --exclude='data/' --exclude='models/' \
    . asahai2024@athene-login.fau.edu:~/mist-vla/

# Submit job
sbatch run_hpc.slurm
```

See `run_hpc.slurm` for SLURM job configuration.

## Success Criteria

### Phase 1: Data Collection
- ✓ Collect 2000+ trajectories
- ✓ Include collision labels and positions
- ✓ Compute per-dimension risk labels

### Phase 2: Risk Prediction
- ✓ Per-dimension AUC > 0.75
- ✓ Model trains without overfitting
- ✓ Risk predictions correlate with actual collisions

### Phase 3: Steering Vectors
- ✓ Find neurons for all directional concepts
- ✓ Steering vectors have non-trivial norms (> 0.01)
- ✓ Concepts align with semantic meanings

### Phase 4: Steering Implementation
- ✓ Hooks inject steering correctly
- ✓ Opposition logic selects appropriate concepts
- ✓ Steered actions differ from original

### Phase 5: Evaluation
- ✓ MIST reduces collision rate vs. vanilla
- ✓ MIST maintains success rate (no excessive stopping)
- ✓ MIST outperforms all 4 baseline methods

## Key Hyperparameters

- **K (risk window)**: 10 steps before collision
- **Risk threshold**: 0.5 (trigger intervention)
- **Steering strength (beta)**: 1.0
- **Target layer**: 20 (middle-to-late layer)
- **Risk predictor**: 4096 → 512 → 256 → 7
- **Batch size**: 256
- **Learning rate**: 1e-3
- **Training epochs**: 50

## Expected Timeline

- **Week 1**: Environment setup, verification
- **Weeks 2-3**: Data collection (2000 rollouts)
- **Week 4**: Train risk predictor, validate AUC
- **Week 5**: Extract steering vectors
- **Week 6**: Full evaluation on 5 baselines
- **Week 7**: Analysis, write-up, visualization

## Citation & References

**OpenVLA:**
```
@article{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

**LIBERO:**
```
@inproceedings{liu2024libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and others},
  booktitle={NeurIPS},
  year={2024}
}
```

## Contact

For questions or issues, see repository: https://github.com/[your-repo]/mist-vla
