"""
Evaluate MIST-VLA on LIBERO benchmark.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# LIBERO imports
import libero.libero.envs
from libero.libero import benchmark

# MIST imports
from src.models.hooked_openvla import HookedOpenVLA
from src.failure_detection.safe_detector import SAFEDetector
from src.attribution.failure_localizer import FailureLocalizer
from src.steering.activation_steerer import (
    FFNAnalyzer,
    SteeringVectorComputer,
    ActivationSteerer
)
from src.recovery.recovery_orchestrator import MISTRecoveryOrchestrator, MISTEvaluator


def evaluate_libero(
    task_suite: str,
    model_path: str,
    detector_path: str,
    steering_vectors_path: str,
    n_episodes: int = 50,
    max_steps: int = 200,
    save_dir: str = "results"
):
    """
    Evaluate MIST on LIBERO task suite.
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load components
    print("Loading model...")
    hooked_vla = HookedOpenVLA(model_path)

    print("Loading failure detector...")
    detector = SAFEDetector(hooked_vla)
    detector.detector.load_state_dict(torch.load(detector_path))

    # Load conformal calibration
    calib_path = detector_path.replace("detector", "conformal_calibration")
    calib = torch.load(calib_path)
    detector.conformal.thresholds = calib['thresholds']

    print("Loading steering vectors...")
    steering_vectors = torch.load(steering_vectors_path)

    # Create MIST components
    localizer = FailureLocalizer(hooked_vla, detector)
    steerer = ActivationSteerer(hooked_vla, steering_vectors)

    orchestrator = MISTRecoveryOrchestrator(
        hooked_vla,
        detector,
        localizer,
        steerer
    )

    evaluator = MISTEvaluator(orchestrator)

    # Load benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[task_suite]()

    results = {
        'task_suite': task_suite,
        'n_episodes': n_episodes,
        'per_task_results': {},
        'aggregate': {}
    }

    all_successes = []
    all_failures_detected = []
    all_recoveries = []

    for task_id in tqdm(range(len(suite.tasks)), desc="Tasks"):
        env = suite.make_env(task_id)
        instruction = suite.get_task_instruction(task_id)

        task_results = []

        for episode in range(n_episodes // len(suite.tasks)):
            metrics = evaluator.evaluate_task(env, instruction, max_steps)
            task_results.append(metrics)

            all_successes.append(metrics['success'])
            all_failures_detected.append(metrics['n_failures_detected'])
            all_recoveries.append(metrics['n_recoveries_successful'])

        results['per_task_results'][task_id] = {
            'instruction': instruction,
            'success_rate': np.mean([r['success'] for r in task_results]),
            'avg_steps': np.mean([r['n_steps'] for r in task_results]),
            'avg_failures_detected': np.mean([r['n_failures_detected'] for r in task_results]),
        }

    # Aggregate results
    results['aggregate'] = {
        'overall_success_rate': np.mean(all_successes),
        'avg_failures_per_episode': np.mean(all_failures_detected),
        'avg_recoveries_per_episode': np.mean(all_recoveries),
        'recovery_rate': np.sum(all_recoveries) / max(np.sum(all_failures_detected), 1)
    }

    # Save results
    with open(save_path / f"{task_suite}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results for {task_suite} ===")
    print(f"Overall Success Rate: {results['aggregate']['overall_success_rate']:.2%}")
    print(f"Avg Failures Detected: {results['aggregate']['avg_failures_per_episode']:.2f}")
    print(f"Recovery Rate: {results['aggregate']['recovery_rate']:.2%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", default="libero_spatial")
    parser.add_argument("--model_path", default="openvla/openvla-7b")
    parser.add_argument("--detector_path", default="checkpoints/best_detector.pt")
    parser.add_argument("--steering_path", default="data/steering_vectors/all_vectors.pt")
    parser.add_argument("--n_episodes", type=int, default=50)
    args = parser.parse_args()

    evaluate_libero(
        args.task_suite,
        args.model_path,
        args.detector_path,
        args.steering_path,
        args.n_episodes
    )
