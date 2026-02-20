#!/usr/bin/env python3
"""
Phase 5: Run evaluation with all baselines.

This script:
1. Loads OpenVLA, risk predictor, and steering vectors
2. Evaluates 5 baselines:
   - none: No intervention
   - safe_stop: Stop when risk detected
   - random_steer: Random steering
   - generic_slow: Generic 'slow' steering
   - mist: MIST with opposition-based steering
3. Computes metrics: collision rate, success rate, recovery rate
4. Saves results and generates visualization
"""

import os
import sys
import pickle
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark

from src.data_collection.hooks import HiddenStateCollector
from src.data_collection.collision_detection import CollisionDetector
from src.training.risk_predictor import RiskPredictor
from src.steering.steering_module import SteeringModule
from src.evaluation.baselines import create_baseline
from src.evaluation.evaluator import Evaluator


def plot_results(results: dict, save_path: str):
    """
    Plot evaluation results.

    Args:
        results: Dictionary mapping baseline_name -> metrics
        save_path: Path to save plot
    """
    baselines = list(results.keys())
    metrics = ['collision_rate', 'success_rate', 'recovery_rate']
    metric_labels = ['Collision Rate', 'Success Rate', 'Recovery Rate']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]

        values = [results[b][metric] * 100 for b in baselines]  # Convert to percentage
        colors = ['red' if b == 'none' else 'yellow' if b in ['safe_stop', 'random_steer', 'generic_slow'] else 'green' for b in baselines]

        bars = ax.bar(baselines, values, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

        ax.set_ylabel(f'{label} (%)', fontsize=12)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(baselines, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Results plot saved to {save_path}")


def run_evaluation(
    risk_predictor_path: str,
    steering_vectors_path: str,
    output_dir: str,
    benchmark_name: str = 'libero_spatial',
    num_episodes_per_task: int = 10,
    max_steps: int = 200,
    risk_threshold: float = 0.5,
    beta: float = 1.0,
    baselines: list = ['none', 'safe_stop', 'random_steer', 'generic_slow', 'mist'],
    device: str = 'cuda'
):
    """
    Run evaluation with all baselines.

    Args:
        risk_predictor_path: Path to trained risk predictor
        steering_vectors_path: Path to steering vectors
        output_dir: Directory to save results
        benchmark_name: LIBERO benchmark name
        num_episodes_per_task: Episodes per task
        max_steps: Max steps per episode
        risk_threshold: Risk threshold
        beta: Steering strength
        baselines: List of baselines to evaluate
        device: Device for inference
    """
    print("=" * 60)
    print("Phase 5: Evaluation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load OpenVLA
    print("\n[1/6] Loading OpenVLA...")
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
    )
    model = model.to(device)
    model.eval()
    print("  ✓ Model loaded")

    # Load risk predictor
    print("\n[2/6] Loading risk predictor...")
    checkpoint = torch.load(risk_predictor_path, map_location=device)
    risk_predictor = RiskPredictor()
    risk_predictor.load_state_dict(checkpoint['model_state_dict'])
    risk_predictor = risk_predictor.to(device)
    risk_predictor.eval()
    print(f"  ✓ Risk predictor loaded (AUC: {checkpoint.get('mean_auc', 'N/A'):.4f})")

    # Load steering vectors
    print("\n[3/6] Loading steering vectors...")
    with open(steering_vectors_path, 'rb') as f:
        steering_data = pickle.load(f)
    steering_vectors = steering_data['steering_vectors']
    print("  ✓ Steering vectors loaded")

    # Load LIBERO
    print(f"\n[4/6] Loading LIBERO benchmark ({benchmark_name})...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[benchmark_name]()
    num_tasks = len(task_suite)
    print(f"  ✓ Loaded {num_tasks} tasks")

    # Create shared components
    hidden_collector = HiddenStateCollector(model)
    steering_module = SteeringModule(model, steering_vectors, target_layer=20, device=device)

    # Run evaluation for each baseline
    print(f"\n[5/6] Running evaluation...")
    print(f"  Baselines: {baselines}")
    print(f"  Episodes per task: {num_episodes_per_task}")
    print(f"  Total episodes: {num_tasks * num_episodes_per_task}")

    all_results = {}

    for baseline_name in baselines:
        print(f"\n--- Evaluating {baseline_name} ---")

        # Create baseline
        if baseline_name in ['none', 'safe_stop']:
            baseline = create_baseline(baseline_name)
            use_steering = None
        else:
            baseline = create_baseline(baseline_name, steering_vectors, target_layer=20)
            use_steering = steering_module

        # Create dummy env for detector
        dummy_env = task_suite.make_env(task_id=0)
        collision_detector = CollisionDetector(dummy_env)
        dummy_env.close()

        # Create evaluator
        evaluator = Evaluator(
            model=model,
            processor=processor,
            risk_predictor=risk_predictor,
            baseline_method=baseline,
            hidden_collector=hidden_collector,
            collision_detector=collision_detector,
            steering_module=use_steering,
            device=device
        )

        # Run evaluation
        try:
            results = evaluator.evaluate(
                task_suite=task_suite,
                num_episodes_per_task=num_episodes_per_task,
                max_steps=max_steps,
                risk_threshold=risk_threshold,
                beta=beta
            )
            all_results[baseline_name] = results

            # Print results
            print(f"\n  Results for {baseline_name}:")
            print(f"    Collision rate: {results['collision_rate']:.2%}")
            print(f"    Success rate: {results['success_rate']:.2%}")
            print(f"    Recovery rate: {results['recovery_rate']:.2%}")
            print(f"    Avg interventions: {results['avg_interventions']:.2f}")

        except Exception as e:
            print(f"\n  ✗ Error evaluating {baseline_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    print(f"\n[6/6] Saving results...")
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        json_results = {}
        for baseline, metrics in all_results.items():
            json_results[baseline] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
        json.dump(json_results, f, indent=2)
    print(f"  ✓ Results saved to {results_path}")

    # Generate plot
    plot_path = os.path.join(output_dir, 'results.png')
    plot_results(all_results, plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"\n{'Baseline':<15} {'Collision↓':<12} {'Success↑':<12} {'Recovery↑':<12}")
    print("-" * 60)

    for baseline in baselines:
        if baseline in all_results:
            r = all_results[baseline]
            print(f"{baseline:<15} {r['collision_rate']:>10.1%}  {r['success_rate']:>10.1%}  {r['recovery_rate']:>10.1%}")

    # Find best method
    if 'mist' in all_results:
        mist_results = all_results['mist']
        print(f"\n{'='*60}")
        print(f"MIST Performance:")
        print(f"  Collision Rate: {mist_results['collision_rate']:.2%}")
        print(f"  Success Rate: {mist_results['success_rate']:.2%}")
        print(f"  Recovery Rate: {mist_results['recovery_rate']:.2%}")

        # Compare to baseline
        if 'none' in all_results:
            none_results = all_results['none']
            improvement = {
                'collision': (none_results['collision_rate'] - mist_results['collision_rate']) / none_results['collision_rate'],
                'success': (mist_results['success_rate'] - none_results['success_rate']) / (none_results['success_rate'] + 1e-8),
            }
            print(f"\nImprovement over vanilla VLA:")
            print(f"  Collision reduction: {improvement['collision']:+.1%}")
            print(f"  Success improvement: {improvement['success']:+.1%}")

    print("\n" + "=" * 60)
    print("✅ Phase 5 Complete - All Implementation Done!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with all baselines")
    parser.add_argument('--risk-predictor', type=str, default='models/risk_predictor/best_model.pt',
                        help='Path to trained risk predictor')
    parser.add_argument('--steering-vectors', type=str, default='data/phase3/steering_vectors.pkl',
                        help='Path to steering vectors')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                        help='Directory to save results')
    parser.add_argument('--benchmark', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90'],
                        help='LIBERO benchmark')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Episodes per task')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Max steps per episode')
    parser.add_argument('--risk-threshold', type=float, default=0.5,
                        help='Risk threshold for intervention')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Steering strength')
    parser.add_argument('--baselines', type=str, nargs='+',
                        default=['none', 'safe_stop', 'random_steer', 'generic_slow', 'mist'],
                        help='Baselines to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')

    args = parser.parse_args()

    run_evaluation(
        risk_predictor_path=args.risk_predictor,
        steering_vectors_path=args.steering_vectors,
        output_dir=args.output_dir,
        benchmark_name=args.benchmark,
        num_episodes_per_task=args.num_episodes,
        max_steps=args.max_steps,
        risk_threshold=args.risk_threshold,
        beta=args.beta,
        baselines=args.baselines,
        device=args.device,
    )


if __name__ == "__main__":
    main()
