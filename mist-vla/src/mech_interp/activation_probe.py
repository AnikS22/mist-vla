"""
Lightweight mechanistic-interpretability hooks and analysis helpers.
Not connected to the rollout pipeline yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class ProbeResult:
    layer_idx: int
    token_idx: Optional[int]
    feature: torch.Tensor


class ActivationProbe:
    """
    Collects per-layer activations for post-hoc analysis.

    Usage (manual):
        probe = ActivationProbe(model, layers=[-1])
        with probe:
            _ = model(**inputs)
        results = probe.get_results()
    """

    def __init__(self, model, layers: Optional[List[int]] = None) -> None:
        self.model = model
        self.layers = layers
        self._active = False
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._results: List[ProbeResult] = []

    def _save(self, layer_idx: int, module, inp, out) -> None:
        if not self._active:
            return
        hidden = out[0] if isinstance(out, tuple) else out
        self._results.append(ProbeResult(layer_idx=layer_idx, token_idx=None, feature=hidden.detach()))

    def register_hooks(self) -> None:
        self.clear()
        if not hasattr(self.model, "language_model"):
            return
        layers = self.model.language_model.model.layers
        for i, layer in enumerate(layers):
            if self.layers is not None and i not in self.layers:
                continue
            handle = layer.register_forward_hook(lambda m, inp, out, idx=i: self._save(idx, m, inp, out))
            self._hooks.append(handle)

    def clear(self) -> None:
        self._results = []

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_results(self) -> List[ProbeResult]:
        return list(self._results)

    def __enter__(self):
        self.register_hooks()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        return False
