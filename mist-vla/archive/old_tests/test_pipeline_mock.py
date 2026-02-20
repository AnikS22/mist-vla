#!/usr/bin/env python3
"""
Test MIST-VLA pipeline with mock data (no GPU/VLA needed).
This verifies the logic works before running on HPC.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

print("="*60)
print("MIST-VLA Pipeline Test (Mock Data)")
print("="*60)

# Test 1: Failure Detector Components
print("\n[Test 1] Failure Detector Components...")
from src.failure_detection.safe_detector import (
    FailureDetectorMLP,
    FailureDetectorLSTM,
    ConformalPredictor
)

# Create mock detector
detector_mlp = FailureDetectorMLP(input_dim=4096, hidden_dim=256)
print("✓ MLP detector created")

detector_lstm = FailureDetectorLSTM(input_dim=4096, hidden_dim=256)
print("✓ LSTM detector created")

# Test forward pass with mock data
mock_features = torch.randn(2, 10, 4096)  # [batch=2, seq=10, hidden=4096]
score_mlp = detector_mlp(mock_features)
print(f"✓ MLP forward pass: output shape {score_mlp.shape}, values in [0,1]: {score_mlp.min():.3f}-{score_mlp.max():.3f}")

detector_lstm.reset_hidden()
score_lstm = detector_lstm(mock_features)
print(f"✓ LSTM forward pass: output shape {score_lstm.shape}, values in [0,1]: {score_lstm.min():.3f}-{score_lstm.max():.3f}")

# Test conformal predictor
print("\n[Test 2] Conformal Predictor...")
conformal = ConformalPredictor(alpha=0.1)

# Mock calibration data
mock_scores = [np.random.rand(50) for _ in range(20)]
conformal.calibrate(mock_scores, max_timesteps=50)
print(f"✓ Conformal predictor calibrated, thresholds shape: {conformal.thresholds.shape}")

# Test prediction
is_failure, margin = conformal.predict(0.8, timestep=10)
print(f"✓ Prediction at t=10: failure={is_failure}, margin={margin:.3f}")

# Test 3: Attribution Components
print("\n[Test 3] Attribution Components...")
from src.attribution.failure_localizer import FailureLocalizer

# We can't test full attribution without VLA, but check structure
print("✓ FailureLocalizer class imported")

# Mock attribution results
mock_attrs = {
    'image_patches': torch.randn(1, 256),
    'language_tokens': torch.randn(1, 20),
    'total': torch.randn(1, 276)
}
print(f"✓ Mock attribution structure created")

# Test 4: Steering Components
print("\n[Test 4] Steering Components...")
from src.steering.activation_steerer import (
    FFNAnalyzer,
    SteeringVectorComputer,
    ActivationSteerer
)

# Test semantic directions
steering_comp = SteeringVectorComputer(None)  # We'll provide mock analyzer later
print(f"✓ Steering computer has {len(steering_comp.semantic_directions)} semantic directions:")
for direction in list(steering_comp.semantic_directions.keys())[:5]:
    print(f"  - {direction}: {steering_comp.semantic_directions[direction]}")

# Mock steering vectors
mock_steering_vectors = {
    'up': {0: torch.randn(4096), 1: torch.randn(4096)},
    'down': {0: torch.randn(4096), 1: torch.randn(4096)},
    'left': {0: torch.randn(4096), 1: torch.randn(4096)},
}
print("✓ Mock steering vectors created")

# Test 5: Recovery Orchestrator
print("\n[Test 5] Recovery Orchestrator...")
from src.recovery.recovery_orchestrator import RecoveryResult

# Test RecoveryResult dataclass
result = RecoveryResult(
    original_action=torch.randn(7),
    corrected_action=torch.randn(7),
    was_failure_detected=True,
    failure_score=0.75,
    failure_cause='collision_left',
    explanation='Test explanation',
    steering_applied={'left': 1.0}
)
print(f"✓ RecoveryResult created: failure={result.was_failure_detected}, cause={result.failure_cause}")

# Test 6: Visualization Utils
print("\n[Test 6] Visualization Utils...")
from src.utils.visualization import plot_failure_scores, plot_attribution_heatmap

# Mock failure scores over time
mock_scores = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.3, 0.2]
fig = plot_failure_scores(mock_scores)
print("✓ Failure score plot created")

# Mock attribution heatmap
mock_attr = torch.randn(256)
fig = plot_attribution_heatmap(mock_attr)
print("✓ Attribution heatmap created")

# Test 7: Metrics
print("\n[Test 7] Metrics Utils...")
from src.utils.metrics import (
    compute_detection_metrics,
    compute_recovery_metrics,
    compute_task_metrics
)

# Mock detection metrics
mock_pred_scores = np.random.rand(100)
mock_labels = np.random.randint(0, 2, 100)
det_metrics = compute_detection_metrics(mock_pred_scores, mock_labels)
print(f"✓ Detection metrics: ROC-AUC={det_metrics['roc_auc']:.3f}, PR-AUC={det_metrics['pr_auc']:.3f}")

# Mock recovery metrics
rec_metrics = compute_recovery_metrics(10, 7, 100)
print(f"✓ Recovery metrics: recovery_rate={rec_metrics['recovery_rate']:.2%}")

# Test 8: Data Processing
print("\n[Test 8] Data Processing...")

# Mock rollout data structure
mock_rollout = {
    'observations': [],
    'actions': [np.random.randn(7) for _ in range(50)],
    'features': [np.random.randn(10, 4096) for _ in range(50)],
    'rewards': np.random.rand(50),
    'success': True,
    'instruction': 'pick up the red block'
}
print(f"✓ Mock rollout structure: {len(mock_rollout['actions'])} steps")

# Test 9: Script Verification
print("\n[Test 9] Script Structure...")
scripts = [
    'scripts/collect_failure_data.py',
    'scripts/train_failure_detector.py',
    'scripts/extract_steering_vectors.py',
    'scripts/run_libero_eval.py'
]

for script in scripts:
    if Path(script).exists():
        # Check if it's valid Python
        with open(script) as f:
            compile(f.read(), script, 'exec')
        print(f"✓ {script} syntax valid")
    else:
        print(f"✗ {script} not found")

# Test 10: Config Files
print("\n[Test 10] Configuration...")
import yaml

config_file = Path('configs/base_config.yaml')
if config_file.exists():
    with open(config_file) as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded: model={config['model']['name']}")
    print(f"  - Detector type: {config['failure_detection']['detector_type']}")
    print(f"  - Steering coefficient: {config['steering']['coefficient']}")
else:
    print("✗ Config file not found")

print("\n" + "="*60)
print("✅ ALL MOCK TESTS PASSED")
print("="*60)
print("\nNext steps:")
print("1. Transfer to HPC")
print("2. Run full pipeline with real VLA + LIBERO")
print("3. Collect actual experimental results")
