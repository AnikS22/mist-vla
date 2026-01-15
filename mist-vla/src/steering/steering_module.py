"""
Steering module with opposition-based logic.

This module implements:
1. Steering injection hooks (Phase 4.1)
2. Opposition-based steering logic (Phase 4.2)

Opposition mapping:
- If moving right (action[0] > 0) is risky → apply 'left' steering
- If moving left (action[0] < 0) is risky → apply 'right' steering
- Similar for other dimensions
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import numpy as np


class SteeringModule:
    """
    Implements activation steering with opposition-based risk mitigation.

    Usage:
        steerer = SteeringModule(model, steering_vectors_path)
        steerer.set_steering_from_risk(risk_vector, action)
        with steerer:
            action = model.predict_action(image, instruction)
    """

    # Opposition mapping for each dimension
    OPPOSITION_MAPPING = {
        0: ('left', 'right'),       # x dimension
        1: ('backward', 'forward'), # y dimension
        2: ('down', 'up'),          # z dimension
        # Rotational dimensions don't have clear oppositions yet
        # Could be extended with rotation-specific concepts
    }

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, Dict[str, torch.Tensor]],
        target_layer: int = 20,
        device='cuda'
    ):
        """
        Initialize steering module.

        Args:
            model: OpenVLA model
            steering_vectors: Dict mapping layer_idx -> {concept: vector}
            target_layer: Layer to inject steering
            device: Device for computation
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.target_layer = target_layer
        self.device = device

        # Active steering vector
        self.active_steering = None
        self.beta = 0.0  # Steering strength

        # Hook handle
        self.hook_handle = None
        self._active = False

    def _steering_hook(self, module, input, output):
        """Hook function to inject steering into activations."""
        if not self._active or self.active_steering is None:
            return output

        # Output is typically a tuple (hidden_state, ...) or just hidden_state
        if isinstance(output, tuple):
            hidden = output[0]
            other_outputs = output[1:]
        else:
            hidden = output
            other_outputs = ()

        # Add steering vector
        # hidden: [batch, seq_len, hidden_dim]
        # steering: [hidden_dim]
        steering = self.active_steering.to(hidden.device, hidden.dtype)
        hidden = hidden + self.beta * steering.view(1, 1, -1)

        # Return modified output
        if other_outputs:
            return (hidden,) + other_outputs
        else:
            return hidden

    def register_hook(self):
        """Register steering hook on target layer."""
        if self.hook_handle is not None:
            return  # Already registered

        layer = self.model.language_model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._steering_hook)

    def remove_hook(self):
        """Remove steering hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def set_steering(
        self,
        concept: str,
        beta: float = 1.0,
        layer_idx: Optional[int] = None
    ):
        """
        Set steering to a specific concept.

        Args:
            concept: Concept name ('left', 'right', 'up', etc.)
            beta: Steering strength
            layer_idx: Layer to use (default: self.target_layer)
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        if layer_idx not in self.steering_vectors:
            raise ValueError(f"No steering vectors for layer {layer_idx}")

        vectors = self.steering_vectors[layer_idx]
        if concept not in vectors or vectors[concept] is None:
            raise ValueError(f"No steering vector for concept '{concept}' in layer {layer_idx}")

        self.active_steering = vectors[concept]
        self.beta = beta

    def set_steering_from_risk(
        self,
        risk_vector: np.ndarray,
        action: np.ndarray,
        beta: float = 1.0,
        threshold: float = 0.5,
        layer_idx: Optional[int] = None
    ):
        """
        Set steering based on risk prediction using opposition logic.

        Opposition logic:
        - If risk_i > threshold and action_i > 0 → apply negative direction steering
        - If risk_i > threshold and action_i < 0 → apply positive direction steering

        Args:
            risk_vector: Predicted risk [7]
            action: Action being taken [7]
            beta: Steering strength
            threshold: Risk threshold
            layer_idx: Layer to use
        """
        if layer_idx is None:
            layer_idx = self.target_layer

        if layer_idx not in self.steering_vectors:
            raise ValueError(f"No steering vectors for layer {layer_idx}")

        vectors = self.steering_vectors[layer_idx]

        # Find dimension with highest risk
        max_risk_dim = np.argmax(risk_vector)
        max_risk = risk_vector[max_risk_dim]

        if max_risk < threshold:
            # No risk, no steering
            self.active_steering = None
            self.beta = 0.0
            return

        # Check if this dimension has opposition mapping
        if max_risk_dim not in self.OPPOSITION_MAPPING:
            # No opposition concept available
            # Could use 'slow' or 'stop' as fallback
            if 'slow' in vectors and vectors['slow'] is not None:
                self.set_steering('slow', beta=beta, layer_idx=layer_idx)
            else:
                self.active_steering = None
                self.beta = 0.0
            return

        # Get opposition concepts
        neg_concept, pos_concept = self.OPPOSITION_MAPPING[max_risk_dim]

        # Choose concept based on action direction
        action_value = action[max_risk_dim]
        if action_value > 0:
            # Moving in positive direction → steer negative
            concept = neg_concept
        else:
            # Moving in negative direction → steer positive
            concept = pos_concept

        # Apply steering
        if concept in vectors and vectors[concept] is not None:
            self.set_steering(concept, beta=beta, layer_idx=layer_idx)
        else:
            # Fallback to slow
            if 'slow' in vectors and vectors['slow'] is not None:
                self.set_steering('slow', beta=beta * 0.5, layer_idx=layer_idx)
            else:
                self.active_steering = None
                self.beta = 0.0

    def clear_steering(self):
        """Clear active steering."""
        self.active_steering = None
        self.beta = 0.0

    def __enter__(self):
        """Context manager entry - register hook and activate."""
        self.register_hook()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deactivate."""
        self._active = False
        return False

    def get_steering_info(self) -> Dict:
        """Get current steering information."""
        return {
            'active': self._active,
            'steering_set': self.active_steering is not None,
            'beta': self.beta,
            'target_layer': self.target_layer,
        }


class MultiLayerSteering:
    """
    Steering module that injects across multiple layers.

    Useful for stronger or more robust steering.
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, Dict[str, torch.Tensor]],
        target_layers: List[int] = [16, 20, 24],
        device='cuda'
    ):
        """
        Initialize multi-layer steering.

        Args:
            model: OpenVLA model
            steering_vectors: Dict mapping layer_idx -> {concept: vector}
            target_layers: Layers to inject steering
            device: Device for computation
        """
        self.steerers = []
        for layer_idx in target_layers:
            steerer = SteeringModule(
                model, steering_vectors,
                target_layer=layer_idx,
                device=device
            )
            self.steerers.append(steerer)

    def set_steering(self, concept: str, beta: float = 1.0):
        """Set steering for all layers."""
        for steerer in self.steerers:
            try:
                steerer.set_steering(concept, beta=beta)
            except ValueError:
                # Layer may not have this concept
                pass

    def set_steering_from_risk(
        self,
        risk_vector: np.ndarray,
        action: np.ndarray,
        beta: float = 1.0,
        threshold: float = 0.5
    ):
        """Set steering from risk for all layers."""
        for steerer in self.steerers:
            steerer.set_steering_from_risk(
                risk_vector, action, beta, threshold
            )

    def clear_steering(self):
        """Clear steering for all layers."""
        for steerer in self.steerers:
            steerer.clear_steering()

    def __enter__(self):
        for steerer in self.steerers:
            steerer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for steerer in self.steerers:
            steerer.__exit__(exc_type, exc_val, exc_tb)
        return False


# Example usage
if __name__ == "__main__":
    print("Example usage of SteeringModule:")
    print("""
    import pickle
    from transformers import AutoModelForVision2Seq
    from src.steering.steering_module import SteeringModule

    # Load model
    model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")

    # Load steering vectors
    with open('data/phase3/steering_vectors.pkl', 'rb') as f:
        data = pickle.load(f)
    steering_vectors = data['steering_vectors']

    # Create steering module
    steerer = SteeringModule(model, steering_vectors, target_layer=20)

    # Example 1: Manual steering
    steerer.set_steering('left', beta=1.0)
    with steerer:
        action = model.predict_action(image, instruction)

    # Example 2: Risk-based steering
    risk_vector = np.array([0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # High x risk
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])       # Moving right

    steerer.set_steering_from_risk(risk_vector, action, beta=1.0)
    # This will apply 'left' steering because action[0] > 0 and risk[0] > threshold

    with steerer:
        new_action = model.predict_action(image, instruction)
    """)
