"""
Dataset classes for per-dimension risk prediction.

This module provides PyTorch datasets for training the risk predictor.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional


class RiskPredictionDataset(Dataset):
    """
    Dataset for training per-dimension risk predictor.

    Each sample contains:
    - hidden_state: [hidden_dim] - Hidden state from OpenVLA
    - risk_label: [7] - Per-dimension risk labels
    - action: [7] - Action taken
    """

    def __init__(self, samples: List[Dict], normalize_hidden: bool = True):
        """
        Initialize dataset.

        Args:
            samples: List of sample dictionaries
            normalize_hidden: Whether to normalize hidden states
        """
        self.samples = samples
        self.normalize_hidden = normalize_hidden

        # Extract data
        self.hidden_states = np.array([s['hidden_state'] for s in samples])
        self.risk_labels = np.array([s['risk_label'] for s in samples])
        self.actions = np.array([s['action'] for s in samples])

        # Compute normalization stats if needed
        if normalize_hidden:
            self.hidden_mean = self.hidden_states.mean(axis=0)
            self.hidden_std = self.hidden_states.std(axis=0) + 1e-8
        else:
            self.hidden_mean = 0.0
            self.hidden_std = 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get hidden state and normalize
        hidden = self.hidden_states[idx]
        if self.normalize_hidden:
            hidden = (hidden - self.hidden_mean) / self.hidden_std

        # Get risk label
        risk = self.risk_labels[idx]

        # Get action
        action = self.actions[idx]

        return {
            'hidden_state': torch.FloatTensor(hidden),
            'risk_label': torch.FloatTensor(risk),
            'action': torch.FloatTensor(action),
        }

    def get_stats(self):
        """Get dataset statistics."""
        return {
            'num_samples': len(self.samples),
            'hidden_dim': self.hidden_states.shape[1],
            'risk_dims': self.risk_labels.shape[1],
            'positive_rate': (self.risk_labels > 0).mean(axis=0).tolist(),
            'mean_risk': self.risk_labels.mean(axis=0).tolist(),
            'max_risk': self.risk_labels.max(axis=0).tolist(),
        }


def create_dataloaders(
    dataset: RiskPredictionDataset,
    train_ratio: float = 0.8,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create train and validation dataloaders.

    Args:
        dataset: RiskPredictionDataset instance
        train_ratio: Ratio of training data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split dataset
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def balance_dataset(
    samples: List[Dict],
    target_positive_ratio: float = 0.3,
    seed: int = 42
) -> List[Dict]:
    """
    Balance dataset by oversampling positive examples.

    Args:
        samples: List of sample dictionaries
        target_positive_ratio: Target ratio of positive samples
        seed: Random seed

    Returns:
        Balanced list of samples
    """
    np.random.seed(seed)

    # Separate positive and negative samples
    # Positive = any dimension has risk > 0
    positive_samples = []
    negative_samples = []

    for sample in samples:
        if sample['risk_label'].sum() > 0:
            positive_samples.append(sample)
        else:
            negative_samples.append(sample)

    n_pos = len(positive_samples)
    n_neg = len(negative_samples)

    print(f"Original distribution: {n_pos} positive, {n_neg} negative")
    print(f"  Positive ratio: {n_pos/(n_pos+n_neg):.2%}")

    # Calculate how many positives we need
    n_total = len(samples)
    n_pos_target = int(n_total * target_positive_ratio)

    if n_pos < n_pos_target:
        # Oversample positive examples
        n_oversample = n_pos_target - n_pos
        oversampled = np.random.choice(
            positive_samples,
            size=n_oversample,
            replace=True
        ).tolist()
        balanced_samples = positive_samples + oversampled + negative_samples
    else:
        # Undersample negative examples
        n_neg_target = int(n_pos / target_positive_ratio) - n_pos
        undersampled = np.random.choice(
            negative_samples,
            size=n_neg_target,
            replace=False
        ).tolist()
        balanced_samples = positive_samples + undersampled

    # Shuffle
    np.random.shuffle(balanced_samples)

    n_pos_final = sum(1 for s in balanced_samples if s['risk_label'].sum() > 0)
    print(f"Balanced distribution: {n_pos_final} positive, {len(balanced_samples)-n_pos_final} negative")
    print(f"  Positive ratio: {n_pos_final/len(balanced_samples):.2%}")

    return balanced_samples


# Example usage
if __name__ == "__main__":
    print("Example usage of RiskPredictionDataset:")
    print("""
    import pickle
    from src.training.dataset import RiskPredictionDataset, create_dataloaders

    # Load labeled data
    with open('data/phase1/labeled_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Create dataset
    dataset = RiskPredictionDataset(data['dataset'])
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset stats: {dataset.get_stats()}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset,
        train_ratio=0.8,
        batch_size=256
    )

    # Iterate
    for batch in train_loader:
        hidden = batch['hidden_state']  # [batch, hidden_dim]
        risk = batch['risk_label']      # [batch, 7]
        action = batch['action']        # [batch, 7]
        break
    """)
