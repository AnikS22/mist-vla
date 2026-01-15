"""
Baseline methods for comparison.

This module implements 5 baseline methods:
1. none: No intervention, vanilla VLA
2. safe_stop: Stop when risk predicted (set actions to zero)
3. random_steer: Apply random steering when risk detected
4. generic_slow: Apply 'slow' steering when risk detected
5. mist: MIST with opposition-based steering (our method)
"""

import numpy as np
import torch
from typing import Dict, Optional
from abc import ABC, abstractmethod


class BaselineMethod(ABC):
    """Abstract base class for baseline methods."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def intervene(
        self,
        action: np.ndarray,
        risk_vector: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Apply intervention to action.

        Args:
            action: Original action [7]
            risk_vector: Predicted risk [7]
            threshold: Risk threshold

        Returns:
            Modified action [7]
        """
        pass

    def should_intervene(self, risk_vector: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if intervention is needed."""
        return risk_vector.max() > threshold


class NoInterventionBaseline(BaselineMethod):
    """Baseline 1: No intervention (vanilla VLA)."""

    def __init__(self):
        super().__init__('none')

    def intervene(
        self,
        action: np.ndarray,
        risk_vector: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Return action unchanged."""
        return action.copy()


class SafeStopBaseline(BaselineMethod):
    """Baseline 2: Stop when risk detected."""

    def __init__(self):
        super().__init__('safe_stop')

    def intervene(
        self,
        action: np.ndarray,
        risk_vector: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Set actions to zero if risk detected."""
        if self.should_intervene(risk_vector, threshold):
            # Stop: set all actions to zero
            modified_action = np.zeros_like(action)
            # Keep gripper action
            modified_action[6] = action[6]
            return modified_action
        return action.copy()


class RandomSteerBaseline(BaselineMethod):
    """Baseline 3: Apply random steering when risk detected."""

    def __init__(self, steering_vectors: Dict, target_layer: int = 20):
        super().__init__('random_steer')
        self.steering_vectors = steering_vectors
        self.target_layer = target_layer

        # Get list of available concepts
        if target_layer in steering_vectors:
            self.concepts = [
                concept for concept, vec in steering_vectors[target_layer].items()
                if vec is not None
            ]
        else:
            self.concepts = []

    def intervene(
        self,
        action: np.ndarray,
        risk_vector: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Apply random steering concept."""
        # This method needs to be used with SteeringModule
        # For evaluation, we just mark that steering should be applied
        # The actual steering is handled by the evaluation harness
        return action.copy()

    def get_steering_concept(self) -> Optional[str]:
        """Get random steering concept."""
        if not self.concepts:
            return None
        return np.random.choice(self.concepts)


class GenericSlowBaseline(BaselineMethod):
    """Baseline 4: Apply 'slow' steering when risk detected."""

    def __init__(self, steering_vectors: Dict, target_layer: int = 20):
        super().__init__('generic_slow')
        self.steering_vectors = steering_vectors
        self.target_layer = target_layer

    def intervene(
        self,
        action: np.ndarray,
        risk_vector: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Apply slow steering."""
        # Steering is handled by evaluation harness
        return action.copy()

    def get_steering_concept(self, risk_vector: np.ndarray, threshold: float = 0.5) -> Optional[str]:
        """Get steering concept."""
        if self.should_intervene(risk_vector, threshold):
            return 'slow'
        return None


class MISTBaseline(BaselineMethod):
    """Baseline 5: MIST with opposition-based steering (our method)."""

    OPPOSITION_MAPPING = {
        0: ('left', 'right'),
        1: ('backward', 'forward'),
        2: ('down', 'up'),
    }

    def __init__(self, steering_vectors: Dict, target_layer: int = 20):
        super().__init__('mist')
        self.steering_vectors = steering_vectors
        self.target_layer = target_layer

    def intervene(
        self,
        action: np.ndarray,
        risk_vector: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Apply opposition-based steering."""
        # Steering is handled by evaluation harness
        return action.copy()

    def get_steering_concept(
        self,
        risk_vector: np.ndarray,
        action: np.ndarray,
        threshold: float = 0.5
    ) -> Optional[str]:
        """
        Get steering concept using opposition logic.

        Args:
            risk_vector: Predicted risk [7]
            action: Action being taken [7]
            threshold: Risk threshold

        Returns:
            Steering concept name or None
        """
        if not self.should_intervene(risk_vector, threshold):
            return None

        # Find dimension with highest risk
        max_risk_dim = np.argmax(risk_vector)
        max_risk = risk_vector[max_risk_dim]

        if max_risk < threshold:
            return None

        # Check if this dimension has opposition mapping
        if max_risk_dim not in self.OPPOSITION_MAPPING:
            # Fallback to slow
            if self.target_layer in self.steering_vectors:
                vectors = self.steering_vectors[self.target_layer]
                if 'slow' in vectors and vectors['slow'] is not None:
                    return 'slow'
            return None

        # Get opposition concepts
        neg_concept, pos_concept = self.OPPOSITION_MAPPING[max_risk_dim]

        # Choose concept based on action direction
        action_value = action[max_risk_dim]
        if action_value > 0:
            # Moving positive → steer negative
            concept = neg_concept
        else:
            # Moving negative → steer positive
            concept = pos_concept

        # Verify concept exists
        if self.target_layer in self.steering_vectors:
            vectors = self.steering_vectors[self.target_layer]
            if concept in vectors and vectors[concept] is not None:
                return concept

        # Fallback to slow
        if 'slow' in vectors and vectors['slow'] is not None:
            return 'slow'

        return None


def create_baseline(
    name: str,
    steering_vectors: Optional[Dict] = None,
    target_layer: int = 20
) -> BaselineMethod:
    """
    Factory function to create baseline method.

    Args:
        name: Baseline name ('none', 'safe_stop', 'random_steer', 'generic_slow', 'mist')
        steering_vectors: Steering vectors (required for steering baselines)
        target_layer: Target layer for steering

    Returns:
        BaselineMethod instance
    """
    if name == 'none':
        return NoInterventionBaseline()
    elif name == 'safe_stop':
        return SafeStopBaseline()
    elif name == 'random_steer':
        if steering_vectors is None:
            raise ValueError("steering_vectors required for random_steer baseline")
        return RandomSteerBaseline(steering_vectors, target_layer)
    elif name == 'generic_slow':
        if steering_vectors is None:
            raise ValueError("steering_vectors required for generic_slow baseline")
        return GenericSlowBaseline(steering_vectors, target_layer)
    elif name == 'mist':
        if steering_vectors is None:
            raise ValueError("steering_vectors required for mist baseline")
        return MISTBaseline(steering_vectors, target_layer)
    else:
        raise ValueError(f"Unknown baseline: {name}")


# Example usage
if __name__ == "__main__":
    print("Example usage of baseline methods:")
    print("""
    import numpy as np
    from src.evaluation.baselines import create_baseline

    # Create baselines
    none = create_baseline('none')
    safe_stop = create_baseline('safe_stop')

    # Example intervention
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    risk = np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    modified_action_none = none.intervene(action, risk)
    modified_action_stop = safe_stop.intervene(action, risk)

    print(f"Original action: {action}")
    print(f"None baseline: {modified_action_none}")
    print(f"Safe stop baseline: {modified_action_stop}")
    """)
