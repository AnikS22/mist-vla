"""
Main orchestrator that coordinates detection, attribution, and steering.
"""

import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    original_action: torch.Tensor
    corrected_action: torch.Tensor
    was_failure_detected: bool
    failure_score: float
    failure_cause: Optional[str]
    explanation: Optional[str]
    steering_applied: Dict[str, float]


class MISTRecoveryOrchestrator:
    """
    Main MIST-VLA system that orchestrates failure detection,
    attribution, and activation steering for real-time recovery.
    """

    def __init__(
        self,
        hooked_vla,
        failure_detector,
        failure_localizer,
        activation_steerer,
        steering_coefficient: float = 1.0,
        fusion_alpha: float = 0.5
    ):
        self.vla = hooked_vla
        self.detector = failure_detector
        self.localizer = failure_localizer
        self.steerer = activation_steerer

        self.steering_coefficient = steering_coefficient
        self.fusion_alpha = fusion_alpha  # Blend factor for action fusion

        # Mapping from attribution results to steering causes
        self.attribution_to_cause = {
            ('visual', 'bottom-left'): 'collision_left',
            ('visual', 'bottom-right'): 'collision_right',
            ('visual', 'bottom-center'): 'collision_forward',
            ('visual', 'center'): 'grip_miss',
            ('language', None): 'stuck',  # Language confusion often means stuck
        }

    def reset(self):
        """Reset for new episode."""
        self.detector.reset()
        self.steerer.clear_steering()

    def step(
        self,
        image,
        instruction: str,
        previous_action: Optional[torch.Tensor] = None
    ) -> RecoveryResult:
        """
        Execute one step of the MIST recovery pipeline.

        1. Run failure detection
        2. If failure detected: attribute cause
        3. Map cause to steering direction
        4. Apply activation steering
        5. Generate corrected action
        6. Fuse with original if needed

        Returns:
            RecoveryResult with all relevant information
        """
        # Step 1: Get original action (without steering)
        self.steerer.clear_steering()
        original_action = self._get_action(image, instruction)

        # Step 2: Run failure detection
        is_failure, score, margin = self.detector.predict_failure(image, instruction)

        result = RecoveryResult(
            original_action=original_action,
            corrected_action=original_action.clone(),
            was_failure_detected=is_failure,
            failure_score=score,
            failure_cause=None,
            explanation=None,
            steering_applied={}
        )

        if not is_failure:
            return result

        # Step 3: Attribute failure cause
        attribution_result = self.localizer.identify_failure_cause(image, instruction)

        result.explanation = attribution_result['explanation']

        # Step 4: Map attribution to steering cause
        cause = self._map_attribution_to_cause(attribution_result)
        result.failure_cause = cause

        if cause is None:
            # Cannot determine cause, return original action
            return result

        # Step 5: Apply activation steering
        self.steerer.apply_steering(
            cause,
            coefficient=self.steering_coefficient
        )

        result.steering_applied = {cause: self.steering_coefficient}

        # Step 6: Generate corrected action with steering
        corrected_action = self._get_action(image, instruction)

        # Step 7: Fuse actions for smooth motion
        fused_action = self._fuse_actions(
            original_action,
            corrected_action,
            alpha=self.fusion_alpha
        )

        result.corrected_action = fused_action

        # Clean up
        self.steerer.clear_steering()

        return result

    def _get_action(self, image, instruction: str) -> torch.Tensor:
        """Generate action from VLA."""
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.vla.processor(prompt, image).to(self.vla.device)

        with torch.no_grad():
            outputs = self.vla.model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False
            )

        action_tokens = outputs[:, inputs['input_ids'].shape[1]:]

        # Decode action tokens to continuous values
        # This is model-specific
        action = self._decode_action(action_tokens)

        return action

    def _decode_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Decode action tokens to continuous action values."""
        # OpenVLA uses 256-bin discretization
        # Each token represents a bin in [-1, 1]
        action_values = (action_tokens.float() - 128) / 128
        return action_values

    def _map_attribution_to_cause(self, attribution_result: Dict) -> Optional[str]:
        """Map attribution analysis to a steering cause."""
        cause_type = attribution_result['cause_type']

        if cause_type == 'visual':
            # Find dominant visual region
            top_patches = attribution_result['top_visual_patches']
            if top_patches:
                row, col = top_patches[0]['spatial_region']

                # Map to spatial region
                if row > 10:
                    vertical = 'bottom'
                elif row < 5:
                    vertical = 'top'
                else:
                    vertical = 'center'

                if col < 5:
                    horizontal = 'left'
                elif col > 10:
                    horizontal = 'right'
                else:
                    horizontal = 'center'

                region = f'{vertical}-{horizontal}'

                return self.attribution_to_cause.get(
                    ('visual', region),
                    'stuck'  # Default
                )

        elif cause_type == 'language':
            return self.attribution_to_cause.get(('language', None), 'stuck')

        return 'stuck'  # Default fallback

    def _fuse_actions(
        self,
        original: torch.Tensor,
        corrected: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Fuse original and corrected actions for smooth motion.

        Uses weighted average with optional constraint enforcement.
        """
        # Simple linear interpolation
        fused = alpha * corrected + (1 - alpha) * original

        # Clamp to valid action range
        fused = torch.clamp(fused, -1, 1)

        return fused


class MISTEvaluator:
    """
    Evaluation harness for MIST-VLA on LIBERO benchmark.
    """

    def __init__(self, orchestrator: MISTRecoveryOrchestrator):
        self.orchestrator = orchestrator

    def evaluate_task(
        self,
        env,
        instruction: str,
        max_steps: int = 200
    ) -> Dict:
        """
        Evaluate MIST on a single task.

        Returns:
            Dictionary with metrics:
            - success: bool
            - n_steps: int
            - n_failures_detected: int
            - n_recoveries_successful: int
            - failure_causes: List[str]
        """
        self.orchestrator.reset()

        obs = env.reset()
        done = False
        step = 0

        metrics = {
            'success': False,
            'n_steps': 0,
            'n_failures_detected': 0,
            'n_recoveries_successful': 0,
            'failure_causes': [],
            'failure_scores': []
        }

        while not done and step < max_steps:
            image = obs['agentview_image']

            # Run MIST pipeline
            result = self.orchestrator.step(image, instruction)

            metrics['failure_scores'].append(result.failure_score)

            if result.was_failure_detected:
                metrics['n_failures_detected'] += 1
                if result.failure_cause:
                    metrics['failure_causes'].append(result.failure_cause)

            # Execute action
            action = result.corrected_action.cpu().numpy()
            obs, reward, done, info = env.step(action)

            # Check if recovery was successful
            if result.was_failure_detected and reward > 0:
                metrics['n_recoveries_successful'] += 1

            step += 1

        metrics['success'] = info.get('success', False)
        metrics['n_steps'] = step

        return metrics
