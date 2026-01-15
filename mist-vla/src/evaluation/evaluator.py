"""
Evaluation harness for MIST-VLA.

This module evaluates VLA performance with different intervention methods
and computes key metrics:
- Collision rate: % of episodes with collisions
- Success rate: % of episodes completing task
- Recovery rate: % of risky situations recovered from
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from PIL import Image


class EpisodeMetrics:
    """Metrics for a single episode."""

    def __init__(self):
        self.collision_occurred = False
        self.success = False
        self.num_steps = 0
        self.num_interventions = 0
        self.num_risky_steps = 0
        self.max_risk = 0.0
        self.collisions_prevented = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'collision': self.collision_occurred,
            'success': self.success,
            'num_steps': self.num_steps,
            'num_interventions': self.num_interventions,
            'num_risky_steps': self.num_risky_steps,
            'max_risk': self.max_risk,
            'collisions_prevented': self.collisions_prevented,
        }


class Evaluator:
    """
    Evaluation harness for VLA with risk prediction and steering.
    """

    def __init__(
        self,
        model,
        processor,
        risk_predictor,
        baseline_method,
        hidden_collector,
        collision_detector,
        steering_module=None,
        device='cuda'
    ):
        """
        Initialize evaluator.

        Args:
            model: OpenVLA model
            processor: OpenVLA processor
            risk_predictor: Risk predictor model
            baseline_method: Baseline intervention method
            hidden_collector: HiddenStateCollector instance
            collision_detector: CollisionDetector instance
            steering_module: Optional SteeringModule for steering baselines
            device: Device for inference
        """
        self.model = model
        self.processor = processor
        self.risk_predictor = risk_predictor
        self.baseline = baseline_method
        self.hidden_collector = hidden_collector
        self.collision_detector = collision_detector
        self.steering_module = steering_module
        self.device = device

    def run_episode(
        self,
        env,
        instruction: str,
        max_steps: int = 200,
        risk_threshold: float = 0.5,
        beta: float = 1.0
    ) -> EpisodeMetrics:
        """
        Run a single episode with intervention.

        Args:
            env: LIBERO environment
            instruction: Task instruction
            max_steps: Maximum steps
            risk_threshold: Risk threshold for intervention
            beta: Steering strength

        Returns:
            EpisodeMetrics with results
        """
        metrics = EpisodeMetrics()

        obs = env.reset()
        done = False
        step_count = 0

        # Context managers for hooks
        contexts = [self.hidden_collector]
        if self.steering_module is not None:
            contexts.append(self.steering_module)

        # Enter all contexts
        for ctx in contexts:
            ctx.__enter__()

        try:
            while not done and step_count < max_steps:
                # Get observation image
                image = self._extract_image(obs)

                # Predict action with risk prediction
                self.hidden_collector.clear()
                if self.steering_module is not None:
                    self.steering_module.clear_steering()

                # Process inputs
                inputs = self.processor(images=image, text=instruction, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Extract action
                action = self._extract_action(outputs)

                # Get hidden state and predict risk
                hidden_state = self.hidden_collector.get_last_layer()
                with torch.no_grad():
                    risk_vector = self.risk_predictor(hidden_state)
                risk_vector = risk_vector.cpu().numpy()[0]

                # Track metrics
                max_risk = risk_vector.max()
                metrics.max_risk = max(metrics.max_risk, max_risk)
                if max_risk > risk_threshold:
                    metrics.num_risky_steps += 1

                # Apply intervention
                if self.baseline.should_intervene(risk_vector, risk_threshold):
                    metrics.num_interventions += 1

                    # Apply baseline-specific intervention
                    if self.baseline.name == 'safe_stop':
                        action = self.baseline.intervene(action, risk_vector, risk_threshold)

                    elif self.baseline.name == 'random_steer' and self.steering_module is not None:
                        concept = self.baseline.get_steering_concept()
                        if concept is not None:
                            self.steering_module.set_steering(concept, beta=beta)
                            # Re-run inference with steering
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                            action = self._extract_action(outputs)

                    elif self.baseline.name == 'generic_slow' and self.steering_module is not None:
                        concept = self.baseline.get_steering_concept(risk_vector, risk_threshold)
                        if concept is not None:
                            self.steering_module.set_steering(concept, beta=beta)
                            # Re-run inference with steering
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                            action = self._extract_action(outputs)

                    elif self.baseline.name == 'mist' and self.steering_module is not None:
                        concept = self.baseline.get_steering_concept(risk_vector, action, risk_threshold)
                        if concept is not None:
                            self.steering_module.set_steering(concept, beta=beta)
                            # Re-run inference with steering
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                            action = self._extract_action(outputs)

                # Check for collision before action
                had_collision_before, _ = self.collision_detector.check_collision()

                # Execute action
                obs, reward, done, info = env.step(action)

                # Check for collision after action
                has_collision, _ = self.collision_detector.check_collision()

                if has_collision and not had_collision_before:
                    metrics.collision_occurred = True

                step_count += 1

        finally:
            # Exit all contexts
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

        # Record final metrics
        metrics.num_steps = step_count
        metrics.success = info.get('success', False) or info.get('is_success', False)

        return metrics

    def _extract_image(self, obs):
        """Extract image from observation."""
        if 'agentview_image' in obs:
            image = obs['agentview_image']
        elif 'image' in obs:
            image = obs['image']
        else:
            image_keys = [k for k in obs.keys() if 'image' in k.lower()]
            if image_keys:
                image = obs[image_keys[0]]
            else:
                raise ValueError(f"No image found in observation")

        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))

        return image

    def _extract_action(self, outputs):
        """Extract action from model outputs."""
        if hasattr(outputs, 'action'):
            action_tensor = outputs.action
        elif hasattr(outputs, 'logits'):
            action_tensor = outputs.logits[:, -1, :7]
        else:
            raise ValueError("Cannot extract action from outputs")

        return action_tensor.cpu().numpy()[0]

    def evaluate(
        self,
        task_suite,
        num_episodes_per_task: int = 10,
        max_steps: int = 200,
        risk_threshold: float = 0.5,
        beta: float = 1.0
    ) -> Dict:
        """
        Evaluate on multiple tasks.

        Args:
            task_suite: LIBERO task suite
            num_episodes_per_task: Episodes per task
            max_steps: Max steps per episode
            risk_threshold: Risk threshold
            beta: Steering strength

        Returns:
            Dictionary with aggregated results
        """
        all_metrics = []
        num_tasks = len(task_suite)

        pbar = tqdm(total=num_tasks * num_episodes_per_task, desc=f"Evaluating {self.baseline.name}")

        for task_id in range(num_tasks):
            instruction = task_suite.get_task_instruction(task_id)

            for episode in range(num_episodes_per_task):
                env = task_suite.make_env(task_id=task_id)
                self.collision_detector = self.collision_detector.__class__(env)

                try:
                    metrics = self.run_episode(
                        env=env,
                        instruction=instruction,
                        max_steps=max_steps,
                        risk_threshold=risk_threshold,
                        beta=beta
                    )
                    all_metrics.append(metrics.to_dict())

                except Exception as e:
                    print(f"\nError in task {task_id} episode {episode}: {e}")

                finally:
                    env.close()

                pbar.update(1)

        pbar.close()

        # Compute aggregated metrics
        results = self._compute_aggregate_metrics(all_metrics)
        results['baseline'] = self.baseline.name
        results['num_episodes'] = len(all_metrics)

        return results

    def _compute_aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Compute aggregate metrics."""
        if not all_metrics:
            return {}

        collision_rate = np.mean([m['collision'] for m in all_metrics])
        success_rate = np.mean([m['success'] for m in all_metrics])
        avg_steps = np.mean([m['num_steps'] for m in all_metrics])
        avg_interventions = np.mean([m['num_interventions'] for m in all_metrics])
        avg_risky_steps = np.mean([m['num_risky_steps'] for m in all_metrics])

        # Recovery rate: success despite risk
        risky_episodes = [m for m in all_metrics if m['num_risky_steps'] > 0]
        if risky_episodes:
            recovery_rate = np.mean([m['success'] for m in risky_episodes])
        else:
            recovery_rate = 0.0

        return {
            'collision_rate': collision_rate,
            'success_rate': success_rate,
            'recovery_rate': recovery_rate,
            'avg_steps': avg_steps,
            'avg_interventions': avg_interventions,
            'avg_risky_steps': avg_risky_steps,
        }
