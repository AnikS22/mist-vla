"""
Token-neuron alignment extraction.

This module projects FFN neurons onto the token embedding space to identify
neurons that activate for specific directional concepts (left, right, up, down, etc.).

Method:
1. Extract FFN weight matrices W_out from each layer
2. Project neurons onto token embedding space: v_neuron = W_out[i] @ W_unembed
3. Find tokens with highest cosine similarity to each neuron
4. Identify semantic concepts (left, right, up, down, slow, fast, stop)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer
from tqdm import tqdm


class NeuronAlignmentExtractor:
    """
    Extract token-neuron alignments from OpenVLA.

    This finds which FFN neurons correspond to semantic concepts like
    'left', 'right', 'up', 'down', 'slow', 'fast', 'stop'.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize alignment extractor.

        Args:
            model: OpenVLA model
            tokenizer: OpenVLA tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Semantic concepts to search for
        self.directional_concepts = {
            'left': ['left', 'leftward', 'leftwards', 'towards left', 'to the left'],
            'right': ['right', 'rightward', 'rightwards', 'towards right', 'to the right'],
            'up': ['up', 'upward', 'upwards', 'above', 'higher'],
            'down': ['down', 'downward', 'downwards', 'below', 'lower'],
            'forward': ['forward', 'forwards', 'ahead', 'front'],
            'backward': ['backward', 'backwards', 'back', 'behind'],
            'slow': ['slow', 'slowly', 'slower', 'careful', 'gently'],
            'fast': ['fast', 'quickly', 'rapid', 'swift'],
            'stop': ['stop', 'halt', 'freeze', 'pause', 'wait'],
        }

    def get_token_embeddings(self) -> torch.Tensor:
        """
        Get token embeddings from the model.

        Returns:
            Token embeddings [vocab_size, embed_dim]
        """
        # OpenVLA structure: model.language_model.model.embed_tokens
        embed_layer = self.model.language_model.model.embed_tokens
        vocab_size = embed_layer.num_embeddings
        embed_dim = embed_layer.embedding_dim

        # Get all token embeddings
        token_ids = torch.arange(vocab_size, device=self.device)
        embeddings = embed_layer(token_ids)  # [vocab_size, embed_dim]

        return embeddings

    def get_unembedding_matrix(self) -> torch.Tensor:
        """
        Get unembedding matrix (output projection).

        Returns:
            Unembedding matrix [embed_dim, vocab_size]
        """
        # OpenVLA structure: model.language_model.lm_head
        lm_head = self.model.language_model.lm_head
        W_unembed = lm_head.weight.T  # [embed_dim, vocab_size]

        return W_unembed

    def extract_neuron_projections(self, layer_idx: int) -> torch.Tensor:
        """
        Extract neuron projections for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Neuron projections onto token space [n_neurons, vocab_size]
        """
        # Get FFN output weights
        layer = self.model.language_model.model.layers[layer_idx]

        # FFN structure: typically layer.mlp.down_proj or layer.mlp.fc2
        if hasattr(layer.mlp, 'down_proj'):
            W_out = layer.mlp.down_proj.weight  # [hidden_dim, ffn_dim]
        elif hasattr(layer.mlp, 'fc2'):
            W_out = layer.mlp.fc2.weight
        else:
            raise ValueError(f"Cannot find FFN output weights in layer {layer_idx}")

        # Get unembedding matrix
        W_unembed = self.get_unembedding_matrix()  # [hidden_dim, vocab_size]

        # Project neurons onto token space
        # neuron_projections[i, j] = similarity between neuron i and token j
        neuron_projections = W_out.T @ W_unembed  # [ffn_dim, vocab_size]

        return neuron_projections

    def find_concept_neurons(
        self,
        layer_idx: int,
        top_k: int = 20,
        threshold: float = 0.1
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find neurons that align with semantic concepts.

        Args:
            layer_idx: Layer index to analyze
            top_k: Number of top tokens to consider per neuron
            threshold: Minimum alignment score threshold

        Returns:
            Dictionary mapping concept -> [(neuron_idx, score), ...]
        """
        print(f"\nAnalyzing layer {layer_idx}...")

        # Get neuron projections
        projections = self.extract_neuron_projections(layer_idx)  # [n_neurons, vocab_size]
        n_neurons = projections.shape[0]

        # Normalize for cosine similarity
        projections_norm = projections / (projections.norm(dim=1, keepdim=True) + 1e-8)

        # Tokenize all concept words
        concept_token_ids = {}
        for concept, words in self.directional_concepts.items():
            token_ids = []
            for word in words:
                # Tokenize with space prefix (common in language models)
                for prefix in ['', ' ', '\n']:
                    text = prefix + word
                    toks = self.tokenizer.encode(text, add_special_tokens=False)
                    token_ids.extend(toks)

            concept_token_ids[concept] = list(set(token_ids))
            print(f"  {concept}: {len(concept_token_ids[concept])} tokens")

        # Find neurons for each concept
        concept_neurons = {}

        for concept, token_ids in concept_token_ids.items():
            if not token_ids:
                concept_neurons[concept] = []
                continue

            # Get projections for these tokens
            concept_proj = projections_norm[:, token_ids]  # [n_neurons, n_concept_tokens]

            # Average across tokens for this concept
            concept_scores = concept_proj.mean(dim=1)  # [n_neurons]

            # Get top neurons
            top_neurons = []
            for neuron_idx in range(n_neurons):
                score = concept_scores[neuron_idx].item()
                if score > threshold:
                    top_neurons.append((neuron_idx, score))

            # Sort by score
            top_neurons.sort(key=lambda x: x[1], reverse=True)
            concept_neurons[concept] = top_neurons[:top_k]

            print(f"  {concept}: {len(top_neurons)} neurons found (showing top {min(len(top_neurons), 5)})")
            for neuron_idx, score in top_neurons[:5]:
                print(f"    Neuron {neuron_idx}: {score:.4f}")

        return concept_neurons

    def extract_all_layers(
        self,
        layers: Optional[List[int]] = None,
        top_k: int = 20,
        threshold: float = 0.1
    ) -> Dict[int, Dict[str, List[Tuple[int, float]]]]:
        """
        Extract concept neurons from multiple layers.

        Args:
            layers: List of layer indices (None = all layers)
            top_k: Number of top neurons per concept
            threshold: Minimum alignment score

        Returns:
            Dictionary mapping layer_idx -> concept_neurons
        """
        if layers is None:
            n_layers = len(self.model.language_model.model.layers)
            layers = list(range(n_layers))

        results = {}
        for layer_idx in tqdm(layers, desc="Extracting alignments"):
            concept_neurons = self.find_concept_neurons(
                layer_idx,
                top_k=top_k,
                threshold=threshold
            )
            results[layer_idx] = concept_neurons

        return results

    def get_neuron_activation_vector(
        self,
        layer_idx: int,
        neuron_idx: int,
        scale: float = 1.0
    ) -> torch.Tensor:
        """
        Get the activation vector for a specific neuron.

        This is the direction in hidden state space that corresponds
        to activating this neuron.

        Args:
            layer_idx: Layer index
            neuron_idx: Neuron index
            scale: Scaling factor

        Returns:
            Activation vector [hidden_dim]
        """
        layer = self.model.language_model.model.layers[layer_idx]

        # Get FFN output weights for this neuron
        if hasattr(layer.mlp, 'down_proj'):
            W_out = layer.mlp.down_proj.weight  # [hidden_dim, ffn_dim]
        elif hasattr(layer.mlp, 'fc2'):
            W_out = layer.mlp.fc2.weight
        else:
            raise ValueError(f"Cannot find FFN output weights")

        # Get activation vector for this neuron
        activation_vector = W_out[:, neuron_idx] * scale  # [hidden_dim]

        return activation_vector


# Example usage
if __name__ == "__main__":
    print("Example usage of NeuronAlignmentExtractor:")
    print("""
    from transformers import AutoModelForVision2Seq, AutoTokenizer
    from src.steering.neuron_alignment import NeuronAlignmentExtractor

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("openvla/openvla-7b")

    # Create extractor
    extractor = NeuronAlignmentExtractor(model, tokenizer)

    # Find concept neurons in layer 16
    concept_neurons = extractor.find_concept_neurons(layer_idx=16)

    # Get activation vector for 'left' concept
    left_neurons = concept_neurons['left']
    if left_neurons:
        neuron_idx, score = left_neurons[0]
        steering_vector = extractor.get_neuron_activation_vector(16, neuron_idx)
        print(f"Steering vector shape: {steering_vector.shape}")
    """)
