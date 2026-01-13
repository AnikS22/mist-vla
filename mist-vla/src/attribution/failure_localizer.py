"""
Attribution methods to identify WHY the VLA is failing.
Uses Integrated Gradients and token-level analysis.
"""

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, Saliency, LayerIntegratedGradients
from typing import Dict, List, Tuple, Optional
import numpy as np


class FailureLocalizer:
    """
    Localizes failure causes using attribution methods.
    Identifies which input tokens/modalities are responsible for failure.
    """

    def __init__(self, hooked_vla, failure_detector):
        self.vla = hooked_vla
        self.detector = failure_detector

        # Create attribution methods
        self.ig = IntegratedGradients(self._failure_score_fn)
        self.saliency = Saliency(self._failure_score_fn)

    def _failure_score_fn(self, features: torch.Tensor) -> torch.Tensor:
        """Wrapper to get failure score as differentiable output."""
        return self.detector.detector(features)

    def attribute_failure(
        self,
        image,
        instruction: str,
        method: str = "integrated_gradients",
        n_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribution for failure prediction.

        Returns:
            Dictionary with attributions for different input components:
            - 'image_patches': Attribution for each image patch
            - 'language_tokens': Attribution for each language token
            - 'proprioception': Attribution for proprioceptive inputs (if any)
        """
        # Get features with gradient tracking
        features = self.vla.get_last_layer_features(image, instruction)
        features.requires_grad_(True)

        # Compute baseline (zero features)
        baseline = torch.zeros_like(features)

        if method == "integrated_gradients":
            attributions = self.ig.attribute(
                features,
                baselines=baseline,
                n_steps=n_steps,
                return_convergence_delta=False
            )
        elif method == "saliency":
            attributions = self.saliency.attribute(features)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Parse attributions into input components
        # Note: This depends on how inputs are tokenized
        result = self._parse_attributions(attributions, image, instruction)

        return result

    def _parse_attributions(
        self,
        attributions: torch.Tensor,
        image,
        instruction: str
    ) -> Dict[str, torch.Tensor]:
        """
        Parse raw attributions into semantic components.

        For OpenVLA:
        - First N tokens are image patches (N = 256 for 224x224 / 14x14)
        - Remaining tokens are language + action tokens
        """
        # Get sequence positions
        # This is model-specific - adjust based on your VLA

        n_image_tokens = 256  # Standard for OpenVLA
        n_language_tokens = len(self.vla.processor.tokenizer.encode(instruction))

        # Sum attributions over hidden dimension
        attr_per_token = attributions.abs().sum(dim=-1)  # [batch, seq_len]

        if attr_per_token.dim() == 1:
            attr_per_token = attr_per_token.unsqueeze(0)

        result = {
            'image_patches': attr_per_token[:, :n_image_tokens],
            'language_tokens': attr_per_token[:, n_image_tokens:n_image_tokens + n_language_tokens],
            'total': attr_per_token
        }

        return result

    def identify_failure_cause(
        self,
        image,
        instruction: str,
        threshold_percentile: float = 90
    ) -> Dict[str, any]:
        """
        High-level function to identify the cause of failure.

        Returns:
            Dictionary with:
            - 'cause_type': 'visual', 'language', 'proprioception', 'mixed'
            - 'top_visual_patches': Indices of most attributed image regions
            - 'top_language_tokens': Most attributed language tokens
            - 'explanation': Human-readable failure explanation
        """
        # Get attributions
        attrs = self.attribute_failure(image, instruction)

        # Compute attribution mass per modality
        image_attr_total = attrs['image_patches'].sum().item()
        language_attr_total = attrs['language_tokens'].sum().item()
        total_attr = image_attr_total + language_attr_total

        image_ratio = image_attr_total / total_attr
        language_ratio = language_attr_total / total_attr

        # Determine cause type
        if image_ratio > 0.7:
            cause_type = 'visual'
        elif language_ratio > 0.7:
            cause_type = 'language'
        else:
            cause_type = 'mixed'

        # Find top attributed tokens
        threshold = np.percentile(attrs['total'].cpu().numpy(), threshold_percentile)
        top_indices = (attrs['total'] > threshold).nonzero()

        # Parse top visual patches into spatial regions
        top_visual = []
        for idx in top_indices:
            if idx[1] < 256:  # Image patch
                patch_idx = idx[1].item()
                row = patch_idx // 16
                col = patch_idx % 16
                top_visual.append({
                    'patch_idx': patch_idx,
                    'spatial_region': (row, col),
                    'attribution': attrs['image_patches'][0, patch_idx].item()
                })

        # Sort by attribution
        top_visual = sorted(top_visual, key=lambda x: x['attribution'], reverse=True)[:5]

        # Generate explanation
        explanation = self._generate_explanation(
            cause_type,
            image_ratio,
            top_visual,
            instruction
        )

        return {
            'cause_type': cause_type,
            'image_attribution_ratio': image_ratio,
            'language_attribution_ratio': language_ratio,
            'top_visual_patches': top_visual,
            'explanation': explanation,
            'raw_attributions': attrs
        }

    def _generate_explanation(
        self,
        cause_type: str,
        image_ratio: float,
        top_visual: List[Dict],
        instruction: str
    ) -> str:
        """Generate human-readable failure explanation."""

        if cause_type == 'visual':
            # Map spatial regions to semantic descriptions
            regions = [v['spatial_region'] for v in top_visual]

            # Simple heuristic mapping
            region_descriptions = []
            for row, col in regions:
                if row < 5:
                    vertical = "top"
                elif row > 10:
                    vertical = "bottom"
                else:
                    vertical = "middle"

                if col < 5:
                    horizontal = "left"
                elif col > 10:
                    horizontal = "right"
                else:
                    horizontal = "center"

                region_descriptions.append(f"{vertical}-{horizontal}")

            return (
                f"Failure appears to be caused by visual confusion in the "
                f"{', '.join(set(region_descriptions))} region(s) of the image. "
                f"The model is attending strongly ({image_ratio:.1%}) to visual "
                f"features but may be misinterpreting the scene."
            )

        elif cause_type == 'language':
            return (
                f"Failure appears to be caused by language understanding issues. "
                f"The model is focusing heavily ({1-image_ratio:.1%}) on language "
                f"tokens in instruction: '{instruction}'. "
                f"Consider rephrasing or the instruction may be ambiguous."
            )

        else:
            return (
                f"Failure has mixed causes: {image_ratio:.1%} visual, "
                f"{1-image_ratio:.1%} language. The model may be confused about "
                f"the relationship between the instruction and visual scene."
            )
