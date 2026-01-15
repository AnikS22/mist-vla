"""
Data collection module for MIST-VLA.

This module provides tools for:
- Collecting hidden states from OpenVLA during inference
- Detecting collisions in MuJoCo simulation
- Collecting rollout data with collision labels
- Computing per-dimension risk labels
"""

from .hooks import HiddenStateCollector, MultiLayerCollector
from .collision_detection import CollisionDetector

__all__ = [
    'HiddenStateCollector',
    'MultiLayerCollector',
    'CollisionDetector',
]
