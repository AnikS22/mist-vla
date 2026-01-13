"""
SAFE-style failure detector that operates on VLA internal features.
Predicts failure probability and enables early stopping/recovery.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np


class FailureDetectorMLP(nn.Module):
    """Simple MLP for failure score prediction."""

    def __init__(
        self,
        input_dim: int = 4096,  # OpenVLA hidden dim
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent features [batch, seq_len, hidden_dim] or [batch, hidden_dim]

        Returns:
            Failure score in [0, 1]
        """
        if x.dim() == 3:
            # Average pool over sequence
            x = x.mean(dim=1)
        return self.net(x)


class FailureDetectorLSTM(nn.Module):
    """LSTM-based detector for sequential failure prediction."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.hidden = None

    def reset_hidden(self):
        self.hidden = None

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Features [batch, seq_len, hidden_dim] or [batch, hidden_dim]

        Returns:
            Failure score in [0, 1]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # Use last timestep
        last_hidden = lstm_out[:, -1, :]
        score = self.classifier(last_hidden)

        if return_hidden:
            return score, last_hidden
        return score


class ConformalPredictor:
    """
    Functional conformal prediction for time-varying failure thresholds.
    Calibrates thresholds to achieve desired false positive rate.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (default 10% FPR)
        """
        self.alpha = alpha
        self.calibration_scores: List[np.ndarray] = []
        self.thresholds: Optional[np.ndarray] = None

    def calibrate(
        self,
        success_rollout_scores: List[np.ndarray],
        max_timesteps: int = 200
    ):
        """
        Calibrate thresholds on successful rollouts.

        Args:
            success_rollout_scores: List of score arrays from successful episodes
            max_timesteps: Maximum episode length
        """
        # Pad/truncate to uniform length
        padded_scores = []
        for scores in success_rollout_scores:
            if len(scores) < max_timesteps:
                # Pad with last value
                padded = np.pad(scores, (0, max_timesteps - len(scores)),
                               mode='edge')
            else:
                padded = scores[:max_timesteps]
            padded_scores.append(padded)

        # Stack into array [n_rollouts, max_timesteps]
        all_scores = np.stack(padded_scores, axis=0)

        # Compute quantile threshold at each timestep
        # Higher quantile = more conservative (later detection)
        quantile = 1 - self.alpha
        self.thresholds = np.quantile(all_scores, quantile, axis=0)

    def predict(
        self,
        score: float,
        timestep: int
    ) -> Tuple[bool, float]:
        """
        Predict whether current state indicates failure.

        Returns:
            (is_failure, margin) where margin is score - threshold
        """
        if self.thresholds is None:
            raise ValueError("Must call calibrate() first")

        timestep = min(timestep, len(self.thresholds) - 1)
        threshold = self.thresholds[timestep]

        is_failure = score > threshold
        margin = score - threshold

        return is_failure, margin


class SAFEDetector:
    """
    Complete SAFE-style failure detection system.
    Combines feature extraction, score prediction, and conformal calibration.
    """

    def __init__(
        self,
        hooked_vla,  # HookedOpenVLA or similar
        detector_type: str = "mlp",  # or "lstm"
        hidden_dim: int = 4096,
        alpha: float = 0.1
    ):
        self.vla = hooked_vla

        # Create detector
        if detector_type == "mlp":
            self.detector = FailureDetectorMLP(hidden_dim)
        else:
            self.detector = FailureDetectorLSTM(hidden_dim)

        self.detector.to(hooked_vla.device)
        self.conformal = ConformalPredictor(alpha)

        # Running state for LSTM
        self.timestep = 0

    def reset(self):
        """Reset for new episode."""
        self.timestep = 0
        if hasattr(self.detector, 'reset_hidden'):
            self.detector.reset_hidden()

    def extract_features(self, image, instruction: str) -> torch.Tensor:
        """Extract last-layer features from VLA."""
        return self.vla.get_last_layer_features(image, instruction)

    def predict_failure(
        self,
        image,
        instruction: str
    ) -> Tuple[bool, float, float]:
        """
        Run full failure prediction pipeline.

        Returns:
            (is_failure, score, margin)
        """
        # Extract features
        features = self.extract_features(image, instruction)

        # Get failure score
        with torch.no_grad():
            score = self.detector(features).item()

        # Apply conformal prediction
        is_failure, margin = self.conformal.predict(score, self.timestep)

        self.timestep += 1

        return is_failure, score, margin

    def train_detector(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        lr: float = 1e-4
    ):
        """Train the failure detector on collected rollouts."""

        optimizer = torch.optim.AdamW(self.detector.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.detector.train()
            train_loss = 0

            for features, labels in train_loader:
                features = features.to(self.vla.device)
                labels = labels.to(self.vla.device)

                optimizer.zero_grad()
                preds = self.detector(features)
                loss = criterion(preds.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.detector.eval()
            val_loss = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.vla.device)
                    labels = labels.to(self.vla.device)

                    preds = self.detector(features)
                    loss = criterion(preds.squeeze(), labels.float())
                    val_loss += loss.item()

            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Val Loss = {val_loss/len(val_loader):.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.detector.state_dict(), "best_detector.pt")
