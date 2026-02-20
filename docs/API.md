# API Reference

This document provides an overview of the main APIs and interfaces in MIST-VLA.

## Core Modules

### Data Collection

#### `DataCollector`

Collects rollout data with internal signal logging.

```python
from mist_vla.data import DataCollector

collector = DataCollector(
    env_name="libero_spatial",
    model_name="moojink/openvla-7b-oft-finetuned-libero-spatial",
    save_dir="data/rollouts",
    camera_res=256
)

# Collect data
rollouts = collector.collect(
    n_success=10,
    n_failure=10,
    max_attempts_per_task=5,
    seed=0
)
```

**Parameters:**
- `env_name`: Environment to collect from
- `model_name`: VLA model to use
- `save_dir`: Directory to save rollouts
- `camera_res`: Camera resolution (default: 256)

**Methods:**
- `collect(n_success, n_failure, max_attempts_per_task, seed)`: Collect rollout data

---

### Model Wrappers

#### `VLAModel`

Base wrapper for Vision-Language-Action models.

```python
from mist_vla.models import VLAModel

model = VLAModel.load(
    model_name="moojink/openvla-7b-oft-finetuned-libero-spatial",
    device="cuda"
)

# Get action prediction
action = model.predict(
    observation=obs,
    instruction=task_description
)

# Get hidden states
hidden_states = model.get_hidden_states(
    observation=obs,
    instruction=task_description,
    layer_indices=[12, 18, 24]
)
```

**Methods:**
- `predict(observation, instruction)`: Get action prediction
- `get_hidden_states(observation, instruction, layer_indices)`: Extract activations
- `apply_steering(steering_vectors, coefficients)`: Apply activation steering

---

### Risk Prediction

#### `RiskPredictor`

Per-dimension collision risk predictor.

```python
from mist_vla.models import RiskPredictor

predictor = RiskPredictor(
    input_dim=768,
    hidden_dim=256,
    output_dim=7  # 7 DoF for robot
)

# Train predictor
predictor.train(
    train_data=train_loader,
    val_data=val_loader,
    epochs=50,
    learning_rate=1e-4
)

# Predict risk
risk_scores = predictor.predict(hidden_states)
```

**Parameters:**
- `input_dim`: Dimension of input features
- `hidden_dim`: Hidden layer dimension
- `output_dim`: Number of action dimensions

**Methods:**
- `train(train_data, val_data, epochs, learning_rate)`: Train the predictor
- `predict(hidden_states)`: Predict per-dimension risk scores
- `save(path)`: Save model checkpoint
- `load(path)`: Load model checkpoint

---

### Steering

#### `SteeringVectorExtractor`

Extract steering vectors from contrastive data.

```python
from mist_vla.steering import SteeringVectorExtractor

extractor = SteeringVectorExtractor(
    model=vla_model,
    layer_indices=[12, 18, 24]
)

# Extract vectors
steering_vectors = extractor.extract(
    safe_rollouts=safe_data,
    unsafe_rollouts=unsafe_data,
    method="mean_diff"  # or "pca", "caa"
)

# Save vectors
extractor.save_vectors(
    steering_vectors,
    "data/steering_vectors/layer_18.pt"
)
```

**Methods:**
- `extract(safe_rollouts, unsafe_rollouts, method)`: Extract steering vectors
- `save_vectors(vectors, path)`: Save to file
- `load_vectors(path)`: Load from file

---

### Evaluation

#### `Evaluator`

Comprehensive evaluation metrics.

```python
from mist_vla.evaluation import Evaluator

evaluator = Evaluator(
    env_name="libero_spatial",
    model=vla_model
)

# Run evaluation
results = evaluator.evaluate(
    n_episodes=100,
    steering_vectors=steering_vecs,
    steering_coefficients=[0.0, 0.5, 1.0, 2.0]
)

# Get metrics
print(f"Success rate: {results['success_rate']:.2%}")
print(f"Collision rate: {results['collision_rate']:.2%}")
print(f"Recovery rate: {results['recovery_rate']:.2%}")
```

**Methods:**
- `evaluate(n_episodes, steering_vectors, steering_coefficients)`: Run evaluation
- `compute_metrics(rollouts)`: Compute metrics from rollouts
- `save_results(results, path)`: Save evaluation results

---

## Configuration

Configuration files are located in `mist-vla/configs/`.

### Example Configuration

```yaml
# config.yaml
data_collection:
  env_name: libero_spatial
  n_success: 50
  n_failure: 50
  max_attempts: 10
  camera_res: 256

training:
  batch_size: 32
  epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-5

steering:
  layers: [12, 18, 24]
  method: mean_diff
  coefficients: [0.0, 0.5, 1.0, 1.5, 2.0]

evaluation:
  n_episodes: 100
  metrics: [success_rate, collision_rate, recovery_rate]
```

---

## Utilities

### Collision Detection

```python
from mist_vla.utils import CollisionDetector

detector = CollisionDetector(env)

# Check for collisions
collision_info = detector.detect(sim_state)
has_collision = collision_info['has_collision']
collision_dims = collision_info['dimensions']  # Which joints collided
```

### Data Processing

```python
from mist_vla.utils import process_rollouts

# Process raw rollouts
processed_data = process_rollouts(
    rollout_dir="data/rollouts",
    output_dir="data/processed",
    label_failures=True
)
```

---

## Command-Line Interface

Most functionality is accessible via command-line scripts:

```bash
# Data collection
python scripts/collect_failure_data.py \
    --env libero_spatial \
    --model moojink/openvla-7b-oft-finetuned-libero-spatial \
    --n_success 50 \
    --n_failure 50

# Training
python scripts/train_risk_predictor.py \
    --data_dir data/rollouts \
    --output_dir experiments/risk_predictor

# Evaluation
python scripts/run_evaluation.py \
    --env libero_spatial \
    --model moojink/openvla-7b-oft-finetuned-libero-spatial \
    --steering_vectors data/steering_vectors
```

---

## Error Handling

All modules raise descriptive exceptions:

```python
from mist_vla.exceptions import (
    DataCollectionError,
    ModelLoadError,
    SteeringError
)

try:
    model = VLAModel.load("invalid_model_name")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
```

---

## Advanced Usage

See the [examples directory](../mist-vla/notebooks/) for Jupyter notebooks with detailed examples.
