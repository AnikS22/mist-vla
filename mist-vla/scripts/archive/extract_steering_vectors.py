#!/usr/bin/env python3
"""
Phase 3.2 & 3.3: Find semantic concept neurons and compute steering vectors.

This script:
1. Loads OpenVLA model
2. Extracts token-neuron alignments
3. Finds neurons for directional concepts (left, right, up, down, etc.)
4. Computes steering vectors by aggregating top neuron activations
5. Saves steering vectors for Phase 4 (steering implementation)
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.steering.neuron_alignment import NeuronAlignmentExtractor


def compute_steering_vectors(
    concept_neurons: dict,
    extractor: NeuronAlignmentExtractor,
    layer_idx: int,
    top_n: int = 5,
    scale: float = 1.0
) -> dict:
    """
    Compute steering vectors for each concept.

    Args:
        concept_neurons: Dictionary mapping concept -> [(neuron_idx, score), ...]
        extractor: NeuronAlignmentExtractor instance
        layer_idx: Layer index
        top_n: Number of top neurons to aggregate
        scale: Scaling factor for steering vectors

    Returns:
        Dictionary mapping concept -> steering_vector
    """
    steering_vectors = {}

    for concept, neurons in concept_neurons.items():
        if not neurons:
            print(f"  ! No neurons found for '{concept}'")
            steering_vectors[concept] = None
            continue

        # Take top N neurons
        top_neurons = neurons[:top_n]

        # Aggregate their activation vectors
        vectors = []
        for neuron_idx, score in top_neurons:
            vec = extractor.get_neuron_activation_vector(
                layer_idx, neuron_idx, scale=score
            )
            vectors.append(vec)

        # Average and scale
        steering_vector = torch.stack(vectors).mean(dim=0) * scale

        # Normalize
        steering_vector = steering_vector / (steering_vector.norm() + 1e-8)

        steering_vectors[concept] = steering_vector.cpu()

        print(f"  ✓ {concept:12s}: aggregated {len(top_neurons)} neurons, norm={steering_vector.norm():.4f}")

    return steering_vectors


def extract_steering_vectors(
    output_path,
    layers=[16, 20, 24],
    top_k=20,
    threshold=0.1,
    top_n_aggregate=5,
    scale=1.0,
    device='cuda'
):
    """
    Extract steering vectors from OpenVLA.

    Args:
        output_path: Path to save steering vectors
        layers: List of layers to analyze
        top_k: Number of top neurons to find per concept
        threshold: Minimum alignment score
        top_n_aggregate: Number of top neurons to aggregate for steering
        scale: Scaling factor for steering vectors
        device: Device for computation
    """
    print("=" * 60)
    print("Phase 3: Extract Steering Vectors")
    print("=" * 60)

    # Load model and tokenizer
    print("\n[1/4] Loading OpenVLA model and tokenizer...")
    from transformers import AutoModelForVision2Seq, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )
    print("  ✓ Tokenizer loaded")

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
    )
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model loaded on {device}")

    # Create extractor
    print("\n[2/4] Creating neuron alignment extractor...")
    extractor = NeuronAlignmentExtractor(model, tokenizer, device=device)
    print("  ✓ Extractor created")

    # Extract concept neurons for each layer
    print(f"\n[3/4] Extracting concept neurons from layers {layers}...")
    all_concept_neurons = {}
    all_steering_vectors = {}

    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")

        # Find concept neurons
        concept_neurons = extractor.find_concept_neurons(
            layer_idx=layer_idx,
            top_k=top_k,
            threshold=threshold
        )
        all_concept_neurons[layer_idx] = concept_neurons

        # Compute steering vectors
        print(f"\nComputing steering vectors for layer {layer_idx}...")
        steering_vectors = compute_steering_vectors(
            concept_neurons=concept_neurons,
            extractor=extractor,
            layer_idx=layer_idx,
            top_n=top_n_aggregate,
            scale=scale
        )
        all_steering_vectors[layer_idx] = steering_vectors

    # Analyze steering vector norms
    print("\n[4/4] Analyzing steering vectors...")
    for layer_idx, steering_vectors in all_steering_vectors.items():
        print(f"\nLayer {layer_idx}:")
        for concept, vector in steering_vectors.items():
            if vector is not None:
                norm = vector.norm().item()
                print(f"  {concept:12s}: norm={norm:.4f}")
            else:
                print(f"  {concept:12s}: None")

    # Check for non-trivial norms
    print("\n[5/5] Validating steering vectors...")
    valid_count = 0
    total_count = 0

    for layer_idx, steering_vectors in all_steering_vectors.items():
        for concept, vector in steering_vectors.items():
            total_count += 1
            if vector is not None and vector.norm().item() > 0.01:
                valid_count += 1

    print(f"  Valid vectors: {valid_count}/{total_count} ({valid_count/total_count:.1%})")

    if valid_count < total_count * 0.5:
        print("  ⚠️  WARNING: Less than 50% of steering vectors are valid")
        print("     Consider adjusting threshold or collecting different layers")
    else:
        print("  ✓ Steering vectors have non-trivial norms")

    # Save steering vectors
    print(f"\n[6/6] Saving steering vectors to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        'steering_vectors': all_steering_vectors,
        'concept_neurons': all_concept_neurons,
        'metadata': {
            'layers': layers,
            'top_k': top_k,
            'threshold': threshold,
            'top_n_aggregate': top_n_aggregate,
            'scale': scale,
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  ✓ Data saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.2f} MB")

    print("\n" + "=" * 60)
    print("✅ Phase 3 Complete!")
    print("=" * 60)
    print("\nSteering vectors extracted! Next steps:")
    print("  1. Implement steering injection hooks (Phase 4.1)")
    print("  2. Implement opposition-based steering logic (Phase 4.2)")
    print("\nNext command:")
    print("  python scripts/implement_steering.py")


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors")
    parser.add_argument('--output', type=str, default='data/phase3/steering_vectors.pkl',
                        help='Output path for steering vectors')
    parser.add_argument('--layers', type=int, nargs='+', default=[16, 20, 24],
                        help='Layers to analyze')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top neurons per concept')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Minimum alignment score')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of neurons to aggregate for steering')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scaling factor for steering vectors')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation')

    args = parser.parse_args()

    extract_steering_vectors(
        output_path=args.output,
        layers=args.layers,
        top_k=args.top_k,
        threshold=args.threshold,
        top_n_aggregate=args.top_n,
        scale=args.scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
