"""
Activation steering for VLAs based on mechanistic interpretability.
Modifies FFN activations to correct failure-inducing behaviors.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans


class FFNAnalyzer:
    """
    Analyzes FFN (MLP) weight matrices to find semantic directions.
    Based on the VLA mechanistic interpretability paper (Häon et al. 2025).
    """

    def __init__(self, hooked_vla):
        self.vla = hooked_vla

        # Extract FFN weights
        self.ffn_weights = self._extract_ffn_weights()

        # Get token embeddings for projection
        self.token_embeddings = self._get_token_embeddings()

    def _extract_ffn_weights(self) -> Dict[int, torch.Tensor]:
        """Extract output projection weights from each FFN layer."""
        weights = {}

        llm = self.vla.model.language_model

        for layer_idx, layer in enumerate(llm.model.layers):
            # Get down_proj (W_out in standard transformer notation)
            down_proj = layer.mlp.down_proj.weight.data  # [hidden_dim, intermediate_dim]
            weights[layer_idx] = down_proj.T  # Transpose for easier analysis

        return weights

    def _get_token_embeddings(self) -> torch.Tensor:
        """Get the model's token embedding matrix."""
        return self.vla.model.language_model.model.embed_tokens.weight.data

    def project_to_token_space(
        self,
        layer_idx: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Project FFN vectors onto token embedding basis.

        Returns:
            (top_tokens_per_neuron, top_token_strings)
        """
        ffn_weights = self.ffn_weights[layer_idx]  # [intermediate_dim, hidden_dim]
        token_emb = self.token_embeddings  # [vocab_size, hidden_dim]

        # Compute similarity: each FFN neuron vs each token embedding
        # Using cosine similarity
        ffn_norm = ffn_weights / ffn_weights.norm(dim=-1, keepdim=True)
        token_norm = token_emb / token_emb.norm(dim=-1, keepdim=True)

        similarity = ffn_norm @ token_norm.T  # [intermediate_dim, vocab_size]

        # Get top-k tokens for each neuron
        top_k = 10
        top_values, top_indices = similarity.topk(top_k, dim=-1)

        return top_indices, top_values

    def cluster_neurons_by_semantics(
        self,
        layer_idx: int,
        n_clusters: int = 20,
        semantic_filter: Optional[List[str]] = None
    ) -> Dict[str, List[int]]:
        """
        Cluster FFN neurons by their semantic alignment.

        Args:
            layer_idx: Which transformer layer to analyze
            n_clusters: Number of semantic clusters
            semantic_filter: Optional list of target semantic categories

        Returns:
            Dictionary mapping semantic labels to neuron indices
        """
        top_indices, top_values = self.project_to_token_space(layer_idx)

        # Use top token indices as features for clustering
        features = top_indices[:, :5].float().cpu().numpy()  # Top 5 tokens per neuron

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Group neurons by cluster
        clusters = {}
        for neuron_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(neuron_idx)

        # Label clusters based on top tokens
        labeled_clusters = {}
        tokenizer = self.vla.processor.tokenizer

        for cluster_id, neuron_indices in clusters.items():
            # Get most common top tokens in this cluster
            all_top_tokens = top_indices[neuron_indices, 0].tolist()
            most_common = max(set(all_top_tokens), key=all_top_tokens.count)
            label = tokenizer.decode([most_common]).strip()

            # Clean up label
            if not label or label.isspace():
                label = f"cluster_{cluster_id}"

            labeled_clusters[label] = neuron_indices

        return labeled_clusters


class SteeringVectorComputer:
    """
    Computes steering vectors for specific semantic directions.
    """

    def __init__(self, ffn_analyzer: FFNAnalyzer):
        self.analyzer = ffn_analyzer

        # Pre-defined semantic directions for failure recovery
        self.semantic_directions = {
            'slower': ['slow', 'careful', 'gentle', 'cautious'],
            'faster': ['fast', 'quick', 'rapid', 'swift'],
            'up': ['up', 'raise', 'lift', 'above'],
            'down': ['down', 'lower', 'below', 'beneath'],
            'left': ['left', 'leftward'],
            'right': ['right', 'rightward'],
            'retract': ['back', 'retreat', 'withdraw', 'retract'],
            'extend': ['forward', 'extend', 'reach'],
            'open': ['open', 'release', 'ungrasp'],
            'close': ['close', 'grip', 'grasp', 'hold']
        }

    def compute_steering_vector(
        self,
        direction: str,
        layer_idx: int,
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute a steering vector for a semantic direction.

        Args:
            direction: One of the pre-defined semantic directions
            layer_idx: Which layer to create steering vector for
            aggregation: How to combine multiple neurons ('mean' or 'pca')

        Returns:
            Steering vector of shape [hidden_dim]
        """
        if direction not in self.semantic_directions:
            raise ValueError(f"Unknown direction: {direction}")

        target_tokens = self.semantic_directions[direction]

        # Find neurons aligned with these tokens
        clusters = self.analyzer.cluster_neurons_by_semantics(layer_idx)

        relevant_neurons = []
        for label, neuron_indices in clusters.items():
            for token in target_tokens:
                if token.lower() in label.lower():
                    relevant_neurons.extend(neuron_indices)
                    break

        if not relevant_neurons:
            # Fall back to direct token matching
            tokenizer = self.analyzer.vla.processor.tokenizer
            for token in target_tokens:
                token_id = tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    relevant_neurons.append(token_id[0] % self.analyzer.ffn_weights[layer_idx].shape[0])

        # Get FFN weight vectors for relevant neurons
        ffn_weights = self.analyzer.ffn_weights[layer_idx]  # [intermediate, hidden]

        relevant_indices = list(set(relevant_neurons))[:50]  # Limit to avoid noise
        relevant_vectors = ffn_weights[relevant_indices]  # [n_neurons, hidden]

        if aggregation == 'mean':
            steering_vector = relevant_vectors.mean(dim=0)
        elif aggregation == 'pca':
            # Use first principal component
            _, _, V = torch.svd(relevant_vectors)
            steering_vector = V[:, 0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Normalize
        steering_vector = steering_vector / steering_vector.norm()

        return steering_vector

    def compute_all_steering_vectors(
        self,
        layers: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Pre-compute steering vectors for all directions and layers.

        Returns:
            Nested dict: direction -> layer_idx -> steering_vector
        """
        if layers is None:
            layers = list(self.analyzer.ffn_weights.keys())

        all_vectors = {}

        for direction in self.semantic_directions:
            all_vectors[direction] = {}
            for layer_idx in layers:
                try:
                    vector = self.compute_steering_vector(direction, layer_idx)
                    all_vectors[direction][layer_idx] = vector
                except Exception as e:
                    print(f"Warning: Could not compute {direction} for layer {layer_idx}: {e}")

        return all_vectors


class ActivationSteerer:
    """
    Applies activation steering to correct VLA behavior in real-time.
    """

    def __init__(
        self,
        hooked_vla,
        steering_vectors: Dict[str, Dict[int, torch.Tensor]]
    ):
        self.vla = hooked_vla
        self.steering_vectors = steering_vectors

        # Mapping from failure causes to steering directions
        self.cause_to_steering = {
            'collision_left': ['right', 'retract'],
            'collision_right': ['left', 'retract'],
            'collision_forward': ['retract', 'up'],
            'too_fast': ['slower'],
            'too_slow': ['faster'],
            'grip_miss': ['down', 'close'],
            'overshoot': ['slower', 'retract'],
            'stuck': ['up', 'retract']
        }

    def apply_steering(
        self,
        cause: str,
        coefficient: float = 1.0,
        layers: Optional[List[int]] = None
    ):
        """
        Apply steering vectors to correct for a failure cause.

        Args:
            cause: Identified failure cause (from cause_to_steering)
            coefficient: Scaling factor for steering strength
            layers: Which layers to steer (default: last 8 layers)
        """
        if cause not in self.cause_to_steering:
            print(f"Warning: Unknown cause '{cause}', no steering applied")
            return

        directions = self.cause_to_steering[cause]

        if layers is None:
            # Default: last 8 layers (where action semantics are strongest)
            n_layers = len(list(self.steering_vectors.values())[0])
            layers = list(range(n_layers - 8, n_layers))

        # Clear any existing steering hooks
        self.vla.clear_all_hooks()

        # Apply steering for each direction and layer
        for direction in directions:
            if direction not in self.steering_vectors:
                continue

            for layer_idx in layers:
                if layer_idx not in self.steering_vectors[direction]:
                    continue

                steering_vector = self.steering_vectors[direction][layer_idx]
                self.vla.add_steering_hook(
                    layer_idx,
                    steering_vector,
                    coefficient=coefficient
                )

    def clear_steering(self):
        """Remove all steering hooks."""
        self.vla.clear_all_hooks()
