"""
Hidden state collection hooks for OpenVLA.

This module implements hooks to collect hidden states from OpenVLA's transformer
layers during inference. These hidden states are used for:
1. Training the per-dimension risk predictor (Phase 2)
2. Extracting neuron-token alignments (Phase 3)
"""

import torch
from typing import Dict, List, Optional


class HiddenStateCollector:
    """
    Collects hidden states from OpenVLA transformer layers during inference.

    Usage:
        collector = HiddenStateCollector(model)
        with collector:
            action = model.predict_action(image, instruction)
            hidden_state = collector.get_last_layer()
    """

    def __init__(self, model, layers: Optional[List[int]] = None):
        """
        Initialize the hidden state collector.

        Args:
            model: OpenVLA model instance
            layers: Optional list of layer indices to collect. If None, collects all layers.
        """
        self.model = model
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self.hooks = []
        self.layers = layers
        self._active = False

    def _save_hidden(self, layer_idx: int, module, input, output):
        """Hook function to save hidden states."""
        if self._active:
            # Output is typically a tuple (hidden_state, ...) or just hidden_state
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Store detached copy to avoid memory issues
            self.hidden_states[layer_idx] = hidden.detach()

    def register_hooks(self):
        """Register forward hooks on transformer layers."""
        # OpenVLA structure: model.language_model.model.layers
        # Each layer is a transformer block
        layers = self.model.language_model.model.layers

        for i, layer in enumerate(layers):
            # Skip if we're only collecting specific layers
            if self.layers is not None and i not in self.layers:
                continue

            hook = layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._save_hidden(idx, m, inp, out)
            )
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear stored hidden states."""
        self.hidden_states = {}

    def get_layer(self, layer_idx: int) -> torch.Tensor:
        """
        Get hidden state from a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Hidden state tensor of shape [batch, seq_len, hidden_dim]
        """
        if layer_idx not in self.hidden_states:
            raise KeyError(f"Layer {layer_idx} not found in collected states")
        return self.hidden_states[layer_idx]

    def get_last_layer(self, pool: str = "mean") -> torch.Tensor:
        """
        Get final layer hidden state, pooled over action tokens.

        Args:
            pool: Pooling method - "mean", "last", or "first"

        Returns:
            Pooled hidden state of shape [batch, hidden_dim]
        """
        if not self.hidden_states:
            raise ValueError("No hidden states collected. Did you run inference?")

        # Get the last layer
        last_layer_idx = max(self.hidden_states.keys())
        hidden = self.hidden_states[last_layer_idx]  # [batch, seq_len, hidden_dim]

        # Pool over sequence dimension
        if pool == "mean":
            return hidden.mean(dim=1)  # [batch, hidden_dim]
        elif pool == "last":
            return hidden[:, -1, :]  # [batch, hidden_dim]
        elif pool == "first":
            return hidden[:, 0, :]  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pool}")

    def get_all_layers(self) -> Dict[int, torch.Tensor]:
        """
        Get hidden states from all collected layers.

        Returns:
            Dictionary mapping layer index to hidden state tensor
        """
        return self.hidden_states.copy()

    def __enter__(self):
        """Context manager entry - register hooks and activate collection."""
        self.register_hooks()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - deactivate and clean up."""
        self._active = False
        # Don't remove hooks here - allow reuse
        return False


class MultiLayerCollector:
    """
    Collects hidden states from multiple specific layers.
    Useful for analyzing which layer is best for risk prediction.
    """

    def __init__(self, model, layer_indices: List[int]):
        """
        Initialize multi-layer collector.

        Args:
            model: OpenVLA model instance
            layer_indices: List of layer indices to collect from
        """
        self.collector = HiddenStateCollector(model, layers=layer_indices)
        self.layer_indices = layer_indices

    def __enter__(self):
        return self.collector.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.collector.__exit__(exc_type, exc_val, exc_tb)

    def get_layer_features(self, layer_idx: int, pool: str = "mean") -> torch.Tensor:
        """
        Get pooled features from a specific layer.

        Args:
            layer_idx: Layer index
            pool: Pooling method

        Returns:
            Pooled hidden state of shape [batch, hidden_dim]
        """
        hidden = self.collector.get_layer(layer_idx)

        if pool == "mean":
            return hidden.mean(dim=1)
        elif pool == "last":
            return hidden[:, -1, :]
        elif pool == "first":
            return hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pool}")


# Example usage
if __name__ == "__main__":
    print("Example usage of HiddenStateCollector:")
    print("""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from src.data_collection.hooks import HiddenStateCollector

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )

    # Collect hidden states during inference
    collector = HiddenStateCollector(model)
    with collector:
        # Run inference
        inputs = processor(images=image, text=instruction)
        outputs = model(**inputs)

        # Get hidden state from last layer
        hidden = collector.get_last_layer()  # [1, 4096]
        print(f"Hidden state shape: {hidden.shape}")
    """)
