"""
Unified interface for different VLA models (OpenVLA, pi0).
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
from abc import ABC, abstractmethod


class VLAWrapper(ABC):
    """Abstract base class for VLA model wrappers."""

    @abstractmethod
    def get_action(self, image, instruction: str) -> torch.Tensor:
        """Get action from VLA given image and instruction."""
        pass

    @abstractmethod
    def get_action_with_features(
        self,
        image,
        instruction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and latent features."""
        pass

    @abstractmethod
    def get_last_layer_features(self, image, instruction: str) -> torch.Tensor:
        """Extract features from the last layer for failure detection."""
        pass


class OpenVLAWrapper(VLAWrapper):
    """Wrapper for OpenVLA model."""

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: str = "cuda"
    ):
        from .hooked_openvla import HookedOpenVLA

        self.model = HookedOpenVLA(model_name, device)
        self.device = device

    def get_action(self, image, instruction: str) -> torch.Tensor:
        """Get action from OpenVLA."""
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.model.processor(prompt, image).to(self.device)

        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False
            )

        action_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        action = self._decode_action(action_tokens)

        return action

    def get_action_with_features(
        self,
        image,
        instruction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and latent features."""
        action_tokens, cache = self.model.run_with_cache(image, instruction)
        action = self._decode_action(action_tokens)

        # Get last layer features
        n_layers = len([k for k in cache if 'hook_mlp_out' in k])
        last_layer_key = f"blocks.{n_layers-1}.hook_mlp_out"
        features = cache[last_layer_key]

        return action, features

    def get_last_layer_features(self, image, instruction: str) -> torch.Tensor:
        """Extract features from the last transformer layer."""
        return self.model.get_last_layer_features(image, instruction)

    def _decode_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Decode action tokens to continuous values."""
        # OpenVLA uses 256-bin discretization
        # Each token represents a bin in [-1, 1]
        action_values = (action_tokens.float() - 128) / 128
        return action_values


def create_vla_wrapper(
    model_type: str = "openvla",
    model_name: Optional[str] = None,
    device: str = "cuda"
) -> VLAWrapper:
    """
    Factory function to create appropriate VLA wrapper.

    Args:
        model_type: Type of VLA model ('openvla' or 'pi0')
        model_name: Specific model name/path
        device: Device to load model on

    Returns:
        VLAWrapper instance
    """
    if model_type.lower() == "openvla":
        model_name = model_name or "openvla/openvla-7b"
        return OpenVLAWrapper(model_name, device)
    elif model_type.lower() == "pi0":
        # TODO: Implement pi0 wrapper when needed
        raise NotImplementedError("pi0 wrapper not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
