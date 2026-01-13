"""
Train the SAFE-style failure detector on collected rollouts.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.failure_detection.safe_detector import (
    FailureDetectorMLP,
    FailureDetectorLSTM,
    ConformalPredictor
)


class FailureDataset(Dataset):
    """Dataset for failure detection training."""

    def __init__(self, rollouts, max_seq_len=50):
        self.data = []

        for rollout in rollouts:
            features = np.stack(rollout['features'])
            label = 0 if rollout['success'] else 1

            # Split into chunks if too long
            for i in range(0, len(features), max_seq_len):
                chunk = features[i:i + max_seq_len]
                self.data.append({
                    'features': torch.tensor(chunk, dtype=torch.float32),
                    'label': torch.tensor(label, dtype=torch.float32)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(item['features'].shape[0] for item in batch)

    features = []
    labels = []

    for item in batch:
        feat = item['features']
        if feat.shape[0] < max_len:
            padding = torch.zeros(max_len - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
        features.append(feat)
        labels.append(item['label'])

    return {
        'features': torch.stack(features),
        'labels': torch.stack(labels)
    }


def train_detector(
    success_rollouts,
    failure_rollouts,
    detector_type='mlp',
    hidden_dim=4096,
    epochs=50,
    batch_size=32,
    lr=1e-4
):
    """Train failure detector."""

    # Create dataset
    all_rollouts = success_rollouts + failure_rollouts
    train_rollouts, val_rollouts = train_test_split(
        all_rollouts, test_size=0.2, random_state=42
    )

    train_dataset = FailureDataset(train_rollouts)
    val_dataset = FailureDataset(val_rollouts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # Create detector
    if detector_type == 'mlp':
        detector = FailureDetectorMLP(hidden_dim)
    else:
        detector = FailureDetectorLSTM(hidden_dim)

    detector = detector.cuda()

    optimizer = torch.optim.AdamW(detector.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        detector.train()
        train_loss = 0

        for batch in train_loader:
            features = batch['features'].cuda()
            labels = batch['labels'].cuda()

            optimizer.zero_grad()

            if detector_type == 'lstm':
                detector.reset_hidden()

            preds = detector(features).squeeze()
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        detector.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].cuda()
                labels = batch['labels'].cuda()

                if detector_type == 'lstm':
                    detector.reset_hidden()

                preds = detector(features).squeeze()
                loss = criterion(preds, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(detector.state_dict(), "checkpoints/best_detector.pt")

    return detector


def calibrate_conformal(detector, success_rollouts, alpha=0.1):
    """Calibrate conformal predictor on success rollouts."""

    detector.eval()
    conformal = ConformalPredictor(alpha=alpha)

    # Get scores for all success rollouts
    success_scores = []

    with torch.no_grad():
        for rollout in success_rollouts:
            features = torch.tensor(
                np.stack(rollout['features']),
                dtype=torch.float32
            ).cuda()

            scores = detector(features).squeeze().cpu().numpy()
            success_scores.append(scores)

    conformal.calibrate(success_scores)

    # Save calibration
    torch.save({
        'thresholds': conformal.thresholds,
        'alpha': conformal.alpha
    }, "checkpoints/conformal_calibration.pt")

    return conformal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/rollouts")
    parser.add_argument("--detector_type", default="mlp")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)

    # Load rollouts
    data_dir = Path(args.data_dir)

    with open(data_dir / "success_rollouts.pkl", "rb") as f:
        success_rollouts = pickle.load(f)

    with open(data_dir / "failure_rollouts.pkl", "rb") as f:
        failure_rollouts = pickle.load(f)

    # Train detector
    detector = train_detector(
        success_rollouts,
        failure_rollouts,
        detector_type=args.detector_type,
        epochs=args.epochs
    )

    # Calibrate conformal
    conformal = calibrate_conformal(detector, success_rollouts)

    print("Training complete!")
