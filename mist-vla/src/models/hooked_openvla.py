"""
OpenVLA wrapper with hook points for mechanistic interpretability.
Based on TransformerLens patterns adapted for VLA architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, List, Callable, Optional, Tuple
from collections import defaultdict


class HookPoint(nn.Module):
    """Minimal hook point for capturing and modifying activations."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.hooks: List[Callable] = []

    def add_hook(self, hook_fn: Callable):
        self.hooks.append(hook_fn)

    def clear_hooks(self):
        self.hooks = []

    def forward(self, x):
        for hook_fn in self.hooks:
            x = hook_fn(x, self)
        return x


class HookedOpenVLA(nn.Module):
    """
    OpenVLA model with hook points at every FFN layer output.
    Enables latent space monitoring and activation steering.
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: str = "cuda"
    ):
        super().__init__()

        # Load base model
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to(device)

        self.device = device
        self.hook_points: Dict[str, HookPoint] = {}
        self.activation_cache: Dict[str, torch.Tensor] = {}

        # Install hook points
        self._install_hooks()

    def _install_hooks(self):
        """Install hook points at FFN outputs in each transformer layer."""

        # Get the language model backbone (Llama-2 in OpenVLA)
        llm = self.model.language_model

        for layer_idx, layer in enumerate(llm.model.layers):
            # Hook at FFN (MLP) output
            hook_name = f"blocks.{layer_idx}.hook_mlp_out"
            hook_point = HookPoint(hook_name)
            self.hook_points[hook_name] = hook_point

            # Wrap the MLP forward
            original_mlp = layer.mlp

            def make_hooked_mlp(orig_mlp, hp):
                class HookedMLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.mlp = orig_mlp
                        self.hook_point = hp

                    def forward(self, x):
                        out = self.mlp(x)
                        return self.hook_point(out)

                return HookedMLP()

            layer.mlp = make_hooked_mlp(original_mlp, hook_point)

            # Also hook residual stream
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            self.hook_points[resid_hook_name] = HookPoint(resid_hook_name)

    def add_caching_hooks(self):
        """Add hooks that cache activations for analysis."""
        self.activation_cache.clear()

        for name, hook_point in self.hook_points.items():
            def make_cache_hook(cache_name):
                def cache_hook(activation, hp):
                    self.activation_cache[cache_name] = activation.detach().clone()
                    return activation
                return cache_hook

            hook_point.add_hook(make_cache_hook(name))

    def add_steering_hook(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        coefficient: float = 1.0
    ):
        """Add a hook that steers activations at a specific layer."""

        hook_name = f"blocks.{layer_idx}.hook_mlp_out"

        def steering_hook(activation, hp):
            # Add steering vector scaled by coefficient
            return activation + coefficient * steering_vector.to(activation.device)

        self.hook_points[hook_name].add_hook(steering_hook)

    def clear_all_hooks(self):
        """Remove all hooks from all hook points."""
        for hook_point in self.hook_points.values():
            hook_point.clear_hooks()
        self.activation_cache.clear()

    def run_with_cache(
        self,
        image,
        instruction: str,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and return both output and cached activations.
        """
        self.add_caching_hooks()

        # Process inputs
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=7,  # 7 action dimensions
                do_sample=False
            )

        # Get action tokens
        action = outputs[:, inputs['input_ids'].shape[1]:]

        # Return actions and cached activations
        cache = dict(self.activation_cache)
        self.clear_all_hooks()

        return action, cache

    def get_last_layer_features(
        self,
        image,
        instruction: str
    ) -> torch.Tensor:
        """
        Extract features from the last transformer layer.
        This is what SAFE uses for failure detection.
        """
        _, cache = self.run_with_cache(image, instruction)

        # Get last layer's residual stream
        n_layers = len([k for k in cache if 'hook_mlp_out' in k])
        last_layer_key = f"blocks.{n_layers-1}.hook_mlp_out"

        return cache[last_layer_key]
