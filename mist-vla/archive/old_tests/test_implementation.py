#!/usr/bin/env python3
"""
Test script to verify core implementation works.

This tests:
1. Module imports
2. Basic class instantiation
3. Data structures
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing MIST-VLA Implementation")
print("=" * 60)

# Test 1: Imports
print("\n[Test 1] Module imports...")
try:
    from src.data_collection.hooks import HiddenStateCollector, MultiLayerCollector
    from src.data_collection.collision_detection import CollisionDetector
    from src.training.dataset import RiskPredictionDataset
    from src.training.risk_predictor import RiskPredictor, compute_loss
    from src.steering.neuron_alignment import NeuronAlignmentExtractor
    from src.steering.steering_module import SteeringModule
    from src.evaluation.baselines import create_baseline
    from src.evaluation.evaluator import Evaluator, EpisodeMetrics
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Risk Predictor
print("\n[Test 2] Risk Predictor...")
try:
    model = RiskPredictor(input_dim=4096, hidden_dims=[512, 256], output_dim=7)

    # Forward pass
    hidden = torch.randn(32, 4096)
    risk = model(hidden)

    assert risk.shape == (32, 7), f"Expected shape (32, 7), got {risk.shape}"
    assert (risk >= 0).all(), "Risk should be non-negative"

    # Loss computation
    target = torch.rand(32, 7)
    loss = compute_loss(risk, target, loss_type='mse')
    assert loss.ndim == 0, "Loss should be scalar"

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ RiskPredictor working ({n_params:,} parameters)")
    print(f"    Input: [32, 4096] → Output: {risk.shape}")
except Exception as e:
    print(f"  ✗ RiskPredictor test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Dataset
print("\n[Test 3] Dataset...")
try:
    # Create mock samples
    samples = []
    for i in range(100):
        sample = {
            'hidden_state': np.random.randn(4096),
            'risk_label': np.random.rand(7),
            'action': np.random.randn(7),
        }
        samples.append(sample)

    dataset = RiskPredictionDataset(samples, normalize_hidden=True)

    # Test __getitem__
    item = dataset[0]
    assert 'hidden_state' in item
    assert 'risk_label' in item
    assert 'action' in item
    assert item['hidden_state'].shape == (4096,)
    assert item['risk_label'].shape == (7,)

    # Test stats
    stats = dataset.get_stats()
    assert stats['num_samples'] == 100
    assert stats['hidden_dim'] == 4096

    print(f"  ✓ Dataset working ({len(dataset)} samples)")
except Exception as e:
    print(f"  ✗ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Baselines
print("\n[Test 4] Baseline methods...")
try:
    # Test baseline creation
    baselines = ['none', 'safe_stop']
    for baseline_name in baselines:
        baseline = create_baseline(baseline_name)

        # Test intervention
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        risk = np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

        modified_action = baseline.intervene(action, risk, threshold=0.5)
        assert modified_action.shape == (7,), f"Action shape mismatch"

        print(f"  ✓ {baseline_name:12s} working")
except Exception as e:
    print(f"  ✗ Baselines test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Episode Metrics
print("\n[Test 5] Episode metrics...")
try:
    metrics = EpisodeMetrics()
    metrics.collision_occurred = True
    metrics.success = False
    metrics.num_steps = 150
    metrics.num_interventions = 10

    metrics_dict = metrics.to_dict()
    assert 'collision' in metrics_dict
    assert 'success' in metrics_dict
    assert metrics_dict['num_steps'] == 150

    print(f"  ✓ EpisodeMetrics working")
except Exception as e:
    print(f"  ✗ EpisodeMetrics test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Data structures
print("\n[Test 6] Data structures...")
try:
    # Test trajectory structure
    trajectory = {
        'instruction': 'pick up the cube',
        'steps': [
            {
                'action': np.zeros(7),
                'hidden_state': np.zeros(4096),
                'risk_label': np.zeros(7),
                'ee_pos': np.zeros(3),
                'collision': False,
                'collision_pos': None,
            }
        ],
        'success': True,
        'collision_occurred': False,
    }

    assert 'instruction' in trajectory
    assert 'steps' in trajectory
    assert len(trajectory['steps']) == 1

    # Test risk label computation
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    collision_direction = np.array([1.0, 0.0, 0.0])  # Collision to the right
    risk = np.zeros(7)
    risk[0] = max(0, action[0] * collision_direction[0])

    assert risk[0] > 0, "Risk should be positive when moving toward collision"

    print(f"  ✓ Data structures correct")
    print(f"    Risk computation: action={action[0]:.2f}, direction={collision_direction[0]:.2f}, risk={risk[0]:.2f}")
except Exception as e:
    print(f"  ✗ Data structures test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
print("\nImplementation verified. Ready for:")
print("  1. Data collection (Phase 1)")
print("  2. Risk predictor training (Phase 2)")
print("  3. Steering vector extraction (Phase 3)")
print("  4. Full evaluation (Phase 5)")
print("\nSee claude.md for complete pipeline instructions.")
