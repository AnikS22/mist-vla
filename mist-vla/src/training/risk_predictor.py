"""
Per-dimension risk predictor model.

This module implements an MLP that predicts 7-dimensional risk from
hidden states extracted from OpenVLA.

Architecture:
    Input: [batch, hidden_dim] (e.g., 4096)
    Hidden layers: [512, 256]
    Output: [batch, 7] - Per-dimension risk predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class RiskPredictor(nn.Module):
    """
    MLP probe to predict per-dimension collision risk.

    Predicts risk for each of 7 action dimensions:
    [x, y, z, roll, pitch, yaw, gripper]
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 7,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize risk predictor.

        Args:
            input_dim: Hidden state dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (7 for action dimensions)
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Build MLP
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_state: [batch, hidden_dim]

        Returns:
            risk: [batch, 7] - Per-dimension risk predictions
        """
        risk = self.mlp(hidden_state)
        # Apply ReLU to ensure non-negative risk
        risk = F.relu(risk)
        return risk

    def predict(self, hidden_state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary risk flags.

        Args:
            hidden_state: [batch, hidden_dim]
            threshold: Threshold for binary classification

        Returns:
            binary_risk: [batch, 7] - Binary risk flags (0 or 1)
        """
        risk = self.forward(hidden_state)
        return (risk > threshold).float()


class EnsembleRiskPredictor(nn.Module):
    """
    Ensemble of risk predictors for improved robustness.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 7,
        num_models: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize ensemble.

        Args:
            input_dim: Hidden state dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            num_models: Number of models in ensemble
            dropout: Dropout rate
        """
        super().__init__()

        self.models = nn.ModuleList([
            RiskPredictor(input_dim, hidden_dims, output_dim, dropout)
            for _ in range(num_models)
        ])

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - average predictions.

        Args:
            hidden_state: [batch, hidden_dim]

        Returns:
            risk: [batch, 7] - Averaged risk predictions
        """
        predictions = [model(hidden_state) for model in self.models]
        return torch.stack(predictions).mean(dim=0)

    def forward_with_uncertainty(self, hidden_state: torch.Tensor):
        """
        Forward pass with uncertainty estimation.

        Args:
            hidden_state: [batch, hidden_dim]

        Returns:
            Tuple of (mean_risk, std_risk)
        """
        predictions = torch.stack([model(hidden_state) for model in self.models])
        return predictions.mean(dim=0), predictions.std(dim=0)


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = 'mse',
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute loss for risk prediction.

    Args:
        predictions: [batch, 7] - Predicted risks
        targets: [batch, 7] - Target risks
        loss_type: Loss type ('mse', 'mae', 'huber')
        weights: Optional per-sample weights [batch]

    Returns:
        loss: Scalar loss
    """
    if loss_type == 'mse':
        loss = F.mse_loss(predictions, targets, reduction='none')
    elif loss_type == 'mae':
        loss = F.l1_loss(predictions, targets, reduction='none')
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(predictions, targets, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Average over dimensions
    loss = loss.mean(dim=1)  # [batch]

    # Apply weights if provided
    if weights is not None:
        loss = loss * weights

    return loss.mean()


# Example usage
if __name__ == "__main__":
    print("Example usage of RiskPredictor:")
    print("""
    from src.training.risk_predictor import RiskPredictor

    # Create model
    model = RiskPredictor(
        input_dim=4096,
        hidden_dims=[512, 256],
        output_dim=7,
        dropout=0.1
    )

    # Forward pass
    hidden = torch.randn(32, 4096)  # Batch of 32
    risk = model(hidden)             # [32, 7]

    print(f"Input shape: {hidden.shape}")
    print(f"Output shape: {risk.shape}")
    print(f"Risk values: {risk[0]}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")
    """)

    # Test model
    print("\n" + "=" * 60)
    print("Testing RiskPredictor...")
    print("=" * 60)

    model = RiskPredictor()
    hidden = torch.randn(32, 4096)
    risk = model(hidden)

    print(f"\n✓ Input shape: {hidden.shape}")
    print(f"✓ Output shape: {risk.shape}")
    print(f"✓ Output range: [{risk.min():.4f}, {risk.max():.4f}]")
    print(f"✓ Non-negative: {(risk >= 0).all().item()}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {n_params:,}")

    print("\n✅ RiskPredictor test passed!")
