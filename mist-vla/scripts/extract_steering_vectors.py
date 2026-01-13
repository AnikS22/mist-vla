"""
Extract and save steering vectors for all semantic directions.
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from src.models.hooked_openvla import HookedOpenVLA
from src.steering.activation_steerer import FFNAnalyzer, SteeringVectorComputer


def extract_all_steering_vectors(
    model_name: str = "openvla/openvla-7b",
    save_path: str = "data/steering_vectors",
    device: str = "cuda"
):
    """
    Extract steering vectors for all semantic directions and layers.

    Args:
        model_name: VLA model to use
        save_path: Directory to save vectors
        device: Device to run on
    """

    print(f"Loading model: {model_name}")
    hooked_vla = HookedOpenVLA(model_name, device)

    print("Analyzing FFN layers...")
    ffn_analyzer = FFNAnalyzer(hooked_vla)

    print("Computing steering vectors...")
    steering_computer = SteeringVectorComputer(ffn_analyzer)

    # Compute vectors for all directions and layers
    all_vectors = steering_computer.compute_all_steering_vectors()

    # Save vectors
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_file = save_dir / "all_vectors.pt"
    torch.save(all_vectors, output_file)

    print(f"Saved steering vectors to {output_file}")

    # Print summary
    print("\nSteering Vector Summary:")
    for direction, layer_vectors in all_vectors.items():
        print(f"  {direction}: {len(layer_vectors)} layers")

    return all_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--save_path", default="data/steering_vectors")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extract_all_steering_vectors(
        args.model,
        args.save_path,
        args.device
    )
