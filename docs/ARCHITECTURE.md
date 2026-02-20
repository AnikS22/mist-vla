# Architecture Overview

This document provides a high-level overview of the MIST-VLA architecture and design principles.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MIST-VLA                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐      ┌────────────────┐                │
│  │ Data Collection│─────▶│ Risk Predictor │                │
│  │    Pipeline    │      │    Training    │                │
│  └────────────────┘      └────────────────┘                │
│         │                        │                          │
│         ▼                        ▼                          │
│  ┌────────────────┐      ┌────────────────┐                │
│  │  Rollout Data  │      │ Risk Predictor │                │
│  │  + Activations │      │     Model      │                │
│  └────────────────┘      └────────────────┘                │
│         │                        │                          │
│         ▼                        ▼                          │
│  ┌────────────────┐      ┌────────────────┐                │
│  │   Steering     │      │   Evaluation   │                │
│  │    Vectors     │─────▶│    Pipeline    │                │
│  └────────────────┘      └────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Collection Pipeline

**Purpose**: Collect rollout data with detailed internal signals

**Components**:
- `DataCollector`: Orchestrates rollout collection
- `VLAModel`: Model wrapper for action prediction
- `CollisionDetector`: Detects and labels collisions
- `StateLogger`: Logs robot states, actions, and activations

**Flow**:
1. Initialize environment and model
2. For each task:
   - Reset environment
   - Execute policy until terminal state
   - Log: observations, actions, hidden states, collisions
3. Save rollout data with labels

**Key Files**:
- `src/data/collector.py`
- `src/data/logger.py`
- `scripts/collect_failure_data.py`

---

### 2. Risk Prediction

**Purpose**: Train models to predict per-dimension collision risk

**Architecture**:
```
Hidden States (768-dim)
        │
        ▼
   Linear Layer (768 → 256)
        │
        ▼
      ReLU + Dropout
        │
        ▼
   Linear Layer (256 → 7)
        │
        ▼
    Sigmoid
        │
        ▼
Risk Scores [7-dim] (one per joint)
```

**Training**:
- Loss: Binary Cross-Entropy per dimension
- Optimizer: Adam with weight decay
- Regularization: Dropout, L2 penalty

**Key Files**:
- `src/models/risk_predictor.py`
- `scripts/train_risk_predictor.py`

---

### 3. Activation Steering

**Purpose**: Extract and apply steering vectors for safety interventions

**Methods**:

1. **Mean Difference**:
   ```
   steering_vector = mean(safe_activations) - mean(unsafe_activations)
   ```

2. **Contrastive Activation Addition (CAA)**:
   ```
   steering_vector = PCA(safe - unsafe, n_components=1)
   ```

3. **Top-K Activation**:
   ```
   steering_vector = top_k_diff(safe, unsafe, k=100)
   ```

**Application**:
```python
modified_activation = original_activation + coefficient * steering_vector
```

**Key Files**:
- `src/steering/extractor.py`
- `src/steering/applier.py`
- `scripts/extract_steering_vectors.py`

---

### 4. Evaluation Pipeline

**Purpose**: Measure safety and performance metrics

**Metrics**:

1. **Success Rate**: % of successful task completions
2. **Collision Rate**: % of episodes with collisions
3. **Recovery Rate**: % of near-collisions that were avoided
4. **Task Efficiency**: Steps to completion
5. **Precision/Recall**: Risk predictor performance

**Evaluation Protocol**:
1. Run N episodes per task
2. Sweep steering coefficients [0.0, 0.5, 1.0, 1.5, 2.0]
3. Log all metrics
4. Compute statistical significance

**Key Files**:
- `src/evaluation/evaluator.py`
- `src/evaluation/metrics.py`
- `scripts/run_evaluation.py`

---

## Design Principles

### Modularity

Each component is self-contained with clear interfaces:
- Easy to swap model architectures
- Support for different environments
- Extensible steering methods

### Reproducibility

- Fixed random seeds throughout
- Deterministic data collection
- Versioned configurations
- Logged hyperparameters

### Efficiency

- Batch processing where possible
- GPU acceleration for models
- Parallel data collection on HPC
- Cached activations to avoid recomputation

### Safety

- Collision detection at every timestep
- Per-dimension risk tracking
- Conservative steering by default
- Extensive logging for debugging

---

## Data Flow

### Training Phase

```
Environment
    │
    ▼
Rollout Collection
    │
    ├─── Successful Rollouts ──┐
    │                           │
    └─── Failed Rollouts ───────┤
                                │
                                ▼
                    Labeled Dataset
                                │
                                ▼
                    Risk Predictor Training
                                │
                                ▼
                    Trained Risk Model
```

### Inference Phase

```
Observation + Instruction
          │
          ▼
    VLA Model
          │
          ├─── Action Prediction
          │
          └─── Hidden States
                    │
                    ▼
              Risk Predictor
                    │
                    ▼
              Risk Scores
                    │
                    ▼
        High Risk Detected?
                    │
         ┌──────────┴──────────┐
         │                     │
        YES                   NO
         │                     │
         ▼                     ▼
   Apply Steering      Use Original Action
         │                     │
         └─────────┬───────────┘
                   │
                   ▼
              Execute Action
```

---

## File Structure

```
mist-vla/
├── src/
│   ├── data/
│   │   ├── collector.py       # Data collection logic
│   │   ├── logger.py          # State and signal logging
│   │   └── preprocessor.py    # Data preprocessing
│   │
│   ├── models/
│   │   ├── vla_model.py       # VLA model wrapper
│   │   ├── risk_predictor.py  # Risk prediction model
│   │   └── base.py            # Base model interface
│   │
│   ├── steering/
│   │   ├── extractor.py       # Steering vector extraction
│   │   ├── applier.py         # Apply steering to activations
│   │   └── methods.py         # Steering methods (CAA, etc.)
│   │
│   ├── evaluation/
│   │   ├── evaluator.py       # Main evaluation loop
│   │   └── metrics.py         # Metric computation
│   │
│   └── utils/
│       ├── collision.py       # Collision detection
│       ├── config.py          # Configuration management
│       └── logging.py         # Logging utilities
│
├── scripts/
│   ├── collect_*.py           # Data collection scripts
│   ├── train_*.py             # Training scripts
│   ├── extract_*.py           # Steering extraction
│   └── run_*.py               # Evaluation scripts
│
└── configs/
    ├── data_collection.yaml
    ├── training.yaml
    └── evaluation.yaml
```

---

## Extension Points

### Adding New Environments

1. Implement environment interface in `src/environments/`
2. Add collision detection logic
3. Update configuration files
4. Test with data collection pipeline

### Adding New Models

1. Implement model wrapper in `src/models/`
2. Follow `VLAModel` interface
3. Ensure hidden state extraction is supported
4. Add model-specific configuration

### Adding New Steering Methods

1. Implement method in `src/steering/methods.py`
2. Follow extraction interface
3. Add method to configuration options
4. Validate with evaluation pipeline

### Adding New Metrics

1. Implement metric in `src/evaluation/metrics.py`
2. Add to evaluator output
3. Update visualization code
4. Document metric calculation

---

## Performance Considerations

### Memory Management

- Activations cached per rollout to avoid recomputation
- Batch processing for training
- Gradient checkpointing for large models

### Computational Efficiency

- GPU acceleration for neural networks
- Vectorized operations in NumPy/PyTorch
- Parallel environment execution

### Storage

- Compressed rollout storage (HDF5)
- Selective activation logging (only key layers)
- Periodic cleanup of intermediate files

---

## Security & Safety

### Input Validation

- Sanitize configuration parameters
- Validate model checkpoints
- Check environment bounds

### Error Handling

- Graceful degradation on failures
- Comprehensive logging
- Recovery mechanisms for data collection

### Testing

- Unit tests for core components
- Integration tests for pipelines
- Regression tests for metrics

---

## Future Directions

Potential architectural improvements:

1. **Real-time Inference**: Optimize for low-latency deployment
2. **Multi-Task Learning**: Joint training across LIBERO tasks
3. **Adaptive Steering**: Dynamic coefficient selection
4. **Distributed Training**: Multi-GPU/multi-node support
5. **Model Compression**: Quantization and distillation for efficiency

---

For implementation details, see the [API Reference](API.md).
