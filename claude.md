MIST-VLA: Mechanistic Interpretability for Steering and Transparent VLA Failure Recovery
Complete Implementation Blueprint
Project Codename: MIST-VLA (Mechanistic Interpretability Steering for Transparent VLA Recovery)
Core Innovation: A snap-on module that uses mechanistic interpretability to both explain why a VLA is failing (via latent space analysis and attribution) and intervene to fix it (via activation steering) — all without retraining the base model.

PART 1: REPOSITORY SETUP
1.1 Required GitHub Repositories to Clone
# Create project directory
mkdir -p ~/vla-failure-recovery && cd ~/vla-failure-recovery

# ========== CORE VLA MODELS ==========

# OpenVLA - Primary VLA model (PyTorch, well-documented, most interpretability work done here)
git clone https://github.com/openvla/openvla.git

# OpenVLA-OFT - Optimized fine-tuning variant (faster inference, better performance)
git clone https://github.com/moojink/openvla-oft.git

# Physical Intelligence's OpenPi - pi0 and pi0-FAST models (JAX-based)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Alternative: Community PyTorch reimplementation of pi0
git clone https://github.com/allenzren/open-pi-zero.git

# ========== SIMULATION BENCHMARKS ==========

# LIBERO - Primary benchmark (used by SAFE, FailSafe, Mech-Interp paper)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# LIBERO-PRO - Extended benchmark with perturbations (exposes failure modes)
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git

# SimplerEnv - Alternative evaluation environment
git clone https://github.com/simpler-env/SimplerEnv.git

# ========== BASELINE METHODS ==========

# FailSafe - Failure generation and recovery baseline
git clone https://github.com/Jimntu/FailSafe_code.git

# Note: SAFE code should be at https://vla-safe.github.io/ - check for release
# If not available, you'll implement based on paper methodology

# ========== INTERPRETABILITY TOOLS ==========

# TransformerLens - Core mechanistic interpretability library
git clone https://github.com/TransformerLensOrg/TransformerLens.git

# Captum - PyTorch attribution methods (Integrated Gradients, Saliency)
# Install via pip: pip install captum

# SAELens - Sparse Autoencoder tools (optional, for advanced analysis)
git clone https://github.com/jbloomAus/SAELens.git

1.2 Environment Setup
# Create main conda environment
conda create -n mist-vla python=3.10 -y
conda activate mist-vla

# Core dependencies
pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10
pip install flash-attn==2.5.5 --no-build-isolation

# Interpretability tools
pip install transformer-lens captum

# Simulation
pip install robosuite mujoco gymnasium

# For LIBERO
cd LIBERO && pip install -r requirements.txt && pip install -e . && cd ..

# For OpenVLA
cd openvla && pip install -e . && cd ..

# Utilities
pip install wandb matplotlib seaborn scikit-learn tqdm einops

# For OpenPi (JAX-based pi0) - separate environment recommended
# conda create -n openpi python=3.10 && conda activate openpi
# See openpi/README.md for JAX installation


PART 2: SYSTEM ARCHITECTURE
2.1 High-Level Architecture Diagram
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MIST-VLA SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌─────────────────────────────────────────────────┐    │
│  │   RGB Image  │───▶│                  BASE VLA                        │    │
│  │  + Language  │    │            (OpenVLA / pi0-FAST)                  │    │
│  │  Instruction │    │                                                  │    │
│  └──────────────┘    │  ┌─────────────────────────────────────────┐    │    │
│                      │  │         Transformer Backbone             │    │    │
│                      │  │  ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐   │    │    │
│                      │  │  │ L1  │─│ L2  │─│ ... │─...─│ L32 │   │    │    │
│                      │  │  └──┬──┘ └──┬──┘ └──┬──┘      └──┬──┘   │    │    │
│                      │  │     │       │       │            │       │    │    │
│                      │  │     ▼       ▼       ▼            ▼       │    │    │
│                      │  │  ┌─────────────────────────────────┐    │    │    │
│                      │  │  │    Hook Points (FFN outputs)     │    │    │    │
│                      │  │  └─────────────┬───────────────────┘    │    │    │
│                      │  └────────────────┼────────────────────────┘    │    │
│                      └───────────────────┼─────────────────────────────┘    │
│                                          │                                   │
│                    ┌─────────────────────┼─────────────────────────┐        │
│                    │     MIST MODULE (SNAP-ON)                      │        │
│                    │                     │                          │        │
│  ┌─────────────────┴─────────────────────┴──────────────────────────┴──┐    │
│  │                                                                      │    │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────┐ │    │
│  │  │  FAILURE DETECTOR  │  │  FAILURE LOCALIZER │  │   ACTIVATION   │ │    │
│  │  │    (SAFE-style)    │  │   (Attribution)    │  │    STEERER     │ │    │
│  │  │                    │  │                    │  │                │ │    │
│  │  │ • Latent monitor   │  │ • Integrated Grads │  │ • FFN neuron   │ │    │
│  │  │ • MLP/LSTM head    │  │ • Saliency maps    │  │   injection    │ │    │
│  │  │ • Failure score    │  │ • Token attribution│  │ • Steering     │ │    │
│  │  │ • Conformal pred   │  │ • "Why failing?"   │  │   vectors      │ │    │
│  │  └─────────┬──────────┘  └─────────┬──────────┘  └───────┬────────┘ │    │
│  │            │                       │                      │          │    │
│  │            ▼                       ▼                      ▼          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐│    │
│  │  │                    RECOVERY ORCHESTRATOR                         ││    │
│  │  │  1. Monitor failure score continuously                          ││    │
│  │  │  2. When score > threshold: attribute failure cause             ││    │
│  │  │  3. Map cause to semantic steering direction                    ││    │
│  │  │  4. Apply activation steering to FFN neurons                    ││    │
│  │  │  5. Fuse corrected action with original trajectory              ││    │
│  │  └─────────────────────────────────────────────────────────────────┘│    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                          │                                   │
│                                          ▼                                   │
│                               ┌──────────────────┐                          │
│                               │  Corrected Action │                          │
│                               │   + Explanation   │                          │
│                               └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘

2.2 Directory Structure
mist-vla/
├── configs/
│   ├── base_config.yaml           # Default hyperparameters
│   ├── openvla_config.yaml        # OpenVLA-specific settings
│   ├── pi0_config.yaml            # pi0-FAST specific settings
│   └── libero_tasks.yaml          # Task definitions
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vla_wrapper.py         # Unified interface for OpenVLA/pi0
│   │   ├── hooked_openvla.py      # OpenVLA with hook points
│   │   └── hooked_pi0.py          # pi0 with hook points (JAX version)
│   │
│   ├── failure_detection/
│   │   ├── __init__.py
│   │   ├── latent_monitor.py      # Extract latent features from VLA
│   │   ├── failure_classifier.py  # MLP/LSTM failure score predictor
│   │   ├── conformal_predictor.py # Time-varying threshold calibration
│   │   └── safe_detector.py       # Full SAFE-style detector
│   │
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── integrated_gradients.py # IG for VLA actions
│   │   ├── saliency.py            # Gradient saliency maps
│   │   ├── token_attribution.py   # Which tokens caused failure
│   │   └── failure_localizer.py   # Combined attribution analysis
│   │
│   ├── steering/
│   │   ├── __init__.py
│   │   ├── ffn_analyzer.py        # Extract and cluster FFN vectors
│   │   ├── steering_vectors.py    # Compute steering directions
│   │   ├── activation_injector.py # Runtime activation modification
│   │   └── semantic_mapper.py     # Map failure causes to steering
│   │
│   ├── recovery/
│   │   ├── __init__.py
│   │   ├── recovery_orchestrator.py # Main recovery pipeline
│   │   ├── action_fusion.py       # Blend original + corrected actions
│   │   └── explanation_generator.py # Human-readable failure explanations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── hooks.py               # PyTorch hook utilities
│       ├── visualization.py       # Plotting and analysis
│       └── metrics.py             # Evaluation metrics
│
├── scripts/
│   ├── train_failure_detector.py  # Train SAFE-style detector
│   ├── extract_steering_vectors.py # Pre-compute steering directions
│   ├── run_libero_eval.py         # Evaluate on LIBERO benchmark
│   ├── collect_failure_data.py    # Generate failure trajectories
│   └── ablation_studies.py        # Component ablations
│
├── data/
│   ├── steering_vectors/          # Pre-computed steering directions
│   ├── failure_rollouts/          # Collected failure trajectories
│   └── calibration_sets/          # Conformal prediction calibration
│
├── experiments/
│   ├── exp001_baseline/           # SAFE + FailSafe baselines
│   ├── exp002_steering_only/      # Steering without attribution
│   ├── exp003_full_mist/          # Full MIST system
│   └── exp004_ablations/          # Component ablations
│
├── notebooks/
│   ├── 01_explore_vla_internals.ipynb
│   ├── 02_failure_zone_analysis.ipynb
│   ├── 03_steering_vector_discovery.ipynb
│   └── 04_intervention_experiments.ipynb
│
├── tests/
│   ├── test_failure_detection.py
│   ├── test_attribution.py
│   └── test_steering.py
│
├── requirements.txt
├── setup.py
└── README.md


PART 3: IMPLEMENTATION DETAILS
3.1 Hooked OpenVLA Wrapper
# src/models/hooked_openvla.py
"""
OpenVLA wrapper with hook points for mechanistic interpretability.
Based on TransformerLens patterns adapted for VLA architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Dict, List, Callable, Optional, Tuple
from collections import defaultdict


class HookPoint(nn.Module):
    """Minimal hook point for capturing and modifying activations."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.hooks: List[Callable] = []
        
    def add_hook(self, hook_fn: Callable):
        self.hooks.append(hook_fn)
        
    def clear_hooks(self):
        self.hooks = []
        
    def forward(self, x):
        for hook_fn in self.hooks:
            x = hook_fn(x, self)
        return x


class HookedOpenVLA(nn.Module):
    """
    OpenVLA model with hook points at every FFN layer output.
    Enables latent space monitoring and activation steering.
    """
    
    def __init__(
        self, 
        model_name: str = "openvla/openvla-7b",
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load base model
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to(device)
        
        self.device = device
        self.hook_points: Dict[str, HookPoint] = {}
        self.activation_cache: Dict[str, torch.Tensor] = {}
        
        # Install hook points
        self._install_hooks()
        
    def _install_hooks(self):
        """Install hook points at FFN outputs in each transformer layer."""
        
        # Get the language model backbone (Llama-2 in OpenVLA)
        llm = self.model.language_model
        
        for layer_idx, layer in enumerate(llm.model.layers):
            # Hook at FFN (MLP) output
            hook_name = f"blocks.{layer_idx}.hook_mlp_out"
            hook_point = HookPoint(hook_name)
            self.hook_points[hook_name] = hook_point
            
            # Wrap the MLP forward
            original_mlp = layer.mlp
            
            def make_hooked_mlp(orig_mlp, hp):
                class HookedMLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.mlp = orig_mlp
                        self.hook_point = hp
                        
                    def forward(self, x):
                        out = self.mlp(x)
                        return self.hook_point(out)
                        
                return HookedMLP()
            
            layer.mlp = make_hooked_mlp(original_mlp, hook_point)
            
            # Also hook residual stream
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            self.hook_points[resid_hook_name] = HookPoint(resid_hook_name)
            
    def add_caching_hooks(self):
        """Add hooks that cache activations for analysis."""
        self.activation_cache.clear()
        
        for name, hook_point in self.hook_points.items():
            def make_cache_hook(cache_name):
                def cache_hook(activation, hp):
                    self.activation_cache[cache_name] = activation.detach().clone()
                    return activation
                return cache_hook
            
            hook_point.add_hook(make_cache_hook(name))
            
    def add_steering_hook(
        self, 
        layer_idx: int, 
        steering_vector: torch.Tensor,
        coefficient: float = 1.0
    ):
        """Add a hook that steers activations at a specific layer."""
        
        hook_name = f"blocks.{layer_idx}.hook_mlp_out"
        
        def steering_hook(activation, hp):
            # Add steering vector scaled by coefficient
            return activation + coefficient * steering_vector.to(activation.device)
        
        self.hook_points[hook_name].add_hook(steering_hook)
        
    def clear_all_hooks(self):
        """Remove all hooks from all hook points."""
        for hook_point in self.hook_points.values():
            hook_point.clear_hooks()
        self.activation_cache.clear()
        
    def run_with_cache(
        self, 
        image, 
        instruction: str,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run forward pass and return both output and cached activations.
        """
        self.add_caching_hooks()
        
        # Process inputs
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=7,  # 7 action dimensions
                do_sample=False
            )
        
        # Get action tokens
        action = outputs[:, inputs['input_ids'].shape[1]:]
        
        # Return actions and cached activations
        cache = dict(self.activation_cache)
        self.clear_all_hooks()
        
        return action, cache
        
    def get_last_layer_features(
        self, 
        image, 
        instruction: str
    ) -> torch.Tensor:
        """
        Extract features from the last transformer layer.
        This is what SAFE uses for failure detection.
        """
        _, cache = self.run_with_cache(image, instruction)
        
        # Get last layer's residual stream
        n_layers = len([k for k in cache if 'hook_mlp_out' in k])
        last_layer_key = f"blocks.{n_layers-1}.hook_mlp_out"
        
        return cache[last_layer_key]

3.2 Failure Detector (SAFE-style)
# src/failure_detection/safe_detector.py
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

3.3 Failure Attribution/Localization
# src/attribution/failure_localizer.py
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

3.4 Activation Steering
# src/steering/activation_steerer.py
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

3.5 Recovery Orchestrator
# src/recovery/recovery_orchestrator.py
"""
Main orchestrator that coordinates detection, attribution, and steering.
"""

import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    original_action: torch.Tensor
    corrected_action: torch.Tensor
    was_failure_detected: bool
    failure_score: float
    failure_cause: Optional[str]
    explanation: Optional[str]
    steering_applied: Dict[str, float]


class MISTRecoveryOrchestrator:
    """
    Main MIST-VLA system that orchestrates failure detection,
    attribution, and activation steering for real-time recovery.
    """
    
    def __init__(
        self,
        hooked_vla,
        failure_detector,
        failure_localizer,
        activation_steerer,
        steering_coefficient: float = 1.0,
        fusion_alpha: float = 0.5
    ):
        self.vla = hooked_vla
        self.detector = failure_detector
        self.localizer = failure_localizer
        self.steerer = activation_steerer
        
        self.steering_coefficient = steering_coefficient
        self.fusion_alpha = fusion_alpha  # Blend factor for action fusion
        
        # Mapping from attribution results to steering causes
        self.attribution_to_cause = {
            ('visual', 'bottom-left'): 'collision_left',
            ('visual', 'bottom-right'): 'collision_right',
            ('visual', 'bottom-center'): 'collision_forward',
            ('visual', 'center'): 'grip_miss',
            ('language', None): 'stuck',  # Language confusion often means stuck
        }
        
    def reset(self):
        """Reset for new episode."""
        self.detector.reset()
        self.steerer.clear_steering()
        
    def step(
        self,
        image,
        instruction: str,
        previous_action: Optional[torch.Tensor] = None
    ) -> RecoveryResult:
        """
        Execute one step of the MIST recovery pipeline.
        
        1. Run failure detection
        2. If failure detected: attribute cause
        3. Map cause to steering direction
        4. Apply activation steering
        5. Generate corrected action
        6. Fuse with original if needed
        
        Returns:
            RecoveryResult with all relevant information
        """
        # Step 1: Get original action (without steering)
        self.steerer.clear_steering()
        original_action = self._get_action(image, instruction)
        
        # Step 2: Run failure detection
        is_failure, score, margin = self.detector.predict_failure(image, instruction)
        
        result = RecoveryResult(
            original_action=original_action,
            corrected_action=original_action.clone(),
            was_failure_detected=is_failure,
            failure_score=score,
            failure_cause=None,
            explanation=None,
            steering_applied={}
        )
        
        if not is_failure:
            return result
            
        # Step 3: Attribute failure cause
        attribution_result = self.localizer.identify_failure_cause(image, instruction)
        
        result.explanation = attribution_result['explanation']
        
        # Step 4: Map attribution to steering cause
        cause = self._map_attribution_to_cause(attribution_result)
        result.failure_cause = cause
        
        if cause is None:
            # Cannot determine cause, return original action
            return result
            
        # Step 5: Apply activation steering
        self.steerer.apply_steering(
            cause,
            coefficient=self.steering_coefficient
        )
        
        result.steering_applied = {cause: self.steering_coefficient}
        
        # Step 6: Generate corrected action with steering
        corrected_action = self._get_action(image, instruction)
        
        # Step 7: Fuse actions for smooth motion
        fused_action = self._fuse_actions(
            original_action, 
            corrected_action,
            alpha=self.fusion_alpha
        )
        
        result.corrected_action = fused_action
        
        # Clean up
        self.steerer.clear_steering()
        
        return result
        
    def _get_action(self, image, instruction: str) -> torch.Tensor:
        """Generate action from VLA."""
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.vla.processor(prompt, image).to(self.vla.device)
        
        with torch.no_grad():
            outputs = self.vla.model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False
            )
            
        action_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        
        # Decode action tokens to continuous values
        # This is model-specific
        action = self._decode_action(action_tokens)
        
        return action
        
    def _decode_action(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Decode action tokens to continuous action values."""
        # OpenVLA uses 256-bin discretization
        # Each token represents a bin in [-1, 1]
        action_values = (action_tokens.float() - 128) / 128
        return action_values
        
    def _map_attribution_to_cause(self, attribution_result: Dict) -> Optional[str]:
        """Map attribution analysis to a steering cause."""
        cause_type = attribution_result['cause_type']
        
        if cause_type == 'visual':
            # Find dominant visual region
            top_patches = attribution_result['top_visual_patches']
            if top_patches:
                row, col = top_patches[0]['spatial_region']
                
                # Map to spatial region
                if row > 10:
                    vertical = 'bottom'
                elif row < 5:
                    vertical = 'top'
                else:
                    vertical = 'center'
                    
                if col < 5:
                    horizontal = 'left'
                elif col > 10:
                    horizontal = 'right'
                else:
                    horizontal = 'center'
                    
                region = f'{vertical}-{horizontal}'
                
                return self.attribution_to_cause.get(
                    ('visual', region),
                    'stuck'  # Default
                )
                
        elif cause_type == 'language':
            return self.attribution_to_cause.get(('language', None), 'stuck')
            
        return 'stuck'  # Default fallback
        
    def _fuse_actions(
        self,
        original: torch.Tensor,
        corrected: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Fuse original and corrected actions for smooth motion.
        
        Uses weighted average with optional constraint enforcement.
        """
        # Simple linear interpolation
        fused = alpha * corrected + (1 - alpha) * original
        
        # Clamp to valid action range
        fused = torch.clamp(fused, -1, 1)
        
        return fused


class MISTEvaluator:
    """
    Evaluation harness for MIST-VLA on LIBERO benchmark.
    """
    
    def __init__(self, orchestrator: MISTRecoveryOrchestrator):
        self.orchestrator = orchestrator
        
    def evaluate_task(
        self,
        env,
        instruction: str,
        max_steps: int = 200
    ) -> Dict:
        """
        Evaluate MIST on a single task.
        
        Returns:
            Dictionary with metrics:
            - success: bool
            - n_steps: int
            - n_failures_detected: int
            - n_recoveries_successful: int
            - failure_causes: List[str]
        """
        self.orchestrator.reset()
        
        obs = env.reset()
        done = False
        step = 0
        
        metrics = {
            'success': False,
            'n_steps': 0,
            'n_failures_detected': 0,
            'n_recoveries_successful': 0,
            'failure_causes': [],
            'failure_scores': []
        }
        
        while not done and step < max_steps:
            image = obs['agentview_image']
            
            # Run MIST pipeline
            result = self.orchestrator.step(image, instruction)
            
            metrics['failure_scores'].append(result.failure_score)
            
            if result.was_failure_detected:
                metrics['n_failures_detected'] += 1
                if result.failure_cause:
                    metrics['failure_causes'].append(result.failure_cause)
                    
            # Execute action
            action = result.corrected_action.cpu().numpy()
            obs, reward, done, info = env.step(action)
            
            # Check if recovery was successful
            if result.was_failure_detected and reward > 0:
                metrics['n_recoveries_successful'] += 1
                
            step += 1
            
        metrics['success'] = info.get('success', False)
        metrics['n_steps'] = step
        
        return metrics


PART 4: TRAINING AND EVALUATION PIPELINE
4.1 Data Collection Script
# scripts/collect_failure_data.py
"""
Collect successful and failed rollouts for training the failure detector.
Based on FailSafe methodology for generating diverse failure cases.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Import LIBERO
import libero.libero.envs
from libero.libero import benchmark

# Import VLA
from openvla import OpenVLAPolicy


def collect_rollouts(
    env_name: str,
    policy,
    n_success: int = 100,
    n_failure: int = 100,
    max_steps: int = 200,
    save_dir: str = "data/rollouts"
):
    """
    Collect both successful and failed rollouts.
    
    Failure generation strategies:
    1. Natural failures from policy
    2. Injected perturbations (translation, rotation, no-op)
    3. OOD scenarios (new objects, positions)
    """
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[env_name]()
    
    success_rollouts = []
    failure_rollouts = []
    
    for task_id in range(len(task_suite.tasks)):
        env = task_suite.make_env(task_id)
        instruction = task_suite.get_task_instruction(task_id)
        
        # Collect successful rollouts
        while len(success_rollouts) < n_success:
            rollout = run_episode(env, policy, instruction, max_steps)
            
            if rollout['success']:
                success_rollouts.append(rollout)
                print(f"Success: {len(success_rollouts)}/{n_success}")
                
        # Collect natural failures
        while len(failure_rollouts) < n_failure // 2:
            rollout = run_episode(env, policy, instruction, max_steps)
            
            if not rollout['success']:
                failure_rollouts.append(rollout)
                print(f"Natural failure: {len(failure_rollouts)}/{n_failure}")
                
        # Collect injected failures
        while len(failure_rollouts) < n_failure:
            rollout = run_episode_with_perturbation(
                env, policy, instruction, max_steps
            )
            
            if not rollout['success']:
                failure_rollouts.append(rollout)
                print(f"Injected failure: {len(failure_rollouts)}/{n_failure}")
                
    # Save rollouts
    with open(save_path / "success_rollouts.pkl", "wb") as f:
        pickle.dump(success_rollouts, f)
        
    with open(save_path / "failure_rollouts.pkl", "wb") as f:
        pickle.dump(failure_rollouts, f)
        
    print(f"Saved {len(success_rollouts)} success and {len(failure_rollouts)} failure rollouts")
    
    return success_rollouts, failure_rollouts


def run_episode(env, policy, instruction, max_steps):
    """Run a single episode and collect trajectory data."""
    
    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'features': [],  # VLA latent features
        'rewards': [],
        'success': False,
        'instruction': instruction
    }
    
    for step in range(max_steps):
        image = obs['agentview_image']
        
        # Get action and features from policy
        action, features = policy.get_action_with_features(image, instruction)
        
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['features'].append(features.cpu().numpy())
        
        obs, reward, done, info = env.step(action)
        trajectory['rewards'].append(reward)
        
        if done:
            trajectory['success'] = info.get('success', False)
            break
            
    return trajectory


def run_episode_with_perturbation(env, policy, instruction, max_steps):
    """Run episode with injected perturbations to create failures."""
    
    obs = env.reset()
    trajectory = {
        'observations': [],
        'actions': [],
        'features': [],
        'rewards': [],
        'success': False,
        'instruction': instruction,
        'perturbation_type': None,
        'perturbation_step': None
    }
    
    # Choose perturbation type and timing
    perturbation_type = np.random.choice(['translation', 'rotation', 'noop'])
    perturbation_step = np.random.randint(10, max_steps // 2)
    
    trajectory['perturbation_type'] = perturbation_type
    trajectory['perturbation_step'] = perturbation_step
    
    for step in range(max_steps):
        image = obs['agentview_image']
        action, features = policy.get_action_with_features(image, instruction)
        
        # Apply perturbation
        if step >= perturbation_step and step < perturbation_step + 10:
            action = apply_perturbation(action, perturbation_type)
            
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['features'].append(features.cpu().numpy())
        
        obs, reward, done, info = env.step(action)
        trajectory['rewards'].append(reward)
        
        if done:
            trajectory['success'] = info.get('success', False)
            break
            
    return trajectory


def apply_perturbation(action, perturbation_type):
    """Apply perturbation to action."""
    
    if perturbation_type == 'translation':
        # Add random translation offset
        offset = np.random.uniform(-0.1, 0.1, size=3)
        action[:3] += offset
        
    elif perturbation_type == 'rotation':
        # Add random rotation offset
        offset = np.random.uniform(-0.2, 0.2, size=3)
        action[3:6] += offset
        
    elif perturbation_type == 'noop':
        # Zero out action
        action = np.zeros_like(action)
        
    return np.clip(action, -1, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="libero_spatial")
    parser.add_argument("--n_success", type=int, default=100)
    parser.add_argument("--n_failure", type=int, default=100)
    parser.add_argument("--save_dir", default="data/rollouts")
    args = parser.parse_args()
    
    # Load policy
    policy = OpenVLAPolicy("openvla/openvla-7b")
    
    collect_rollouts(
        args.env,
        policy,
        args.n_success,
        args.n_failure,
        save_dir=args.save_dir
    )

4.2 Training Script
# scripts/train_failure_detector.py
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

4.3 Evaluation Script
# scripts/run_libero_eval.py
"""
Evaluate MIST-VLA on LIBERO benchmark.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# LIBERO imports
import libero.libero.envs
from libero.libero import benchmark

# MIST imports
from src.models.hooked_openvla import HookedOpenVLA
from src.failure_detection.safe_detector import SAFEDetector
from src.attribution.failure_localizer import FailureLocalizer
from src.steering.activation_steerer import (
    FFNAnalyzer, 
    SteeringVectorComputer,
    ActivationSteerer
)
from src.recovery.recovery_orchestrator import MISTRecoveryOrchestrator, MISTEvaluator


def evaluate_libero(
    task_suite: str,
    model_path: str,
    detector_path: str,
    steering_vectors_path: str,
    n_episodes: int = 50,
    max_steps: int = 200,
    save_dir: str = "results"
):
    """
    Evaluate MIST on LIBERO task suite.
    """
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load components
    print("Loading model...")
    hooked_vla = HookedOpenVLA(model_path)
    
    print("Loading failure detector...")
    detector = SAFEDetector(hooked_vla)
    detector.detector.load_state_dict(torch.load(detector_path))
    
    # Load conformal calibration
    calib = torch.load(detector_path.replace("detector", "conformal_calibration"))
    detector.conformal.thresholds = calib['thresholds']
    
    print("Loading steering vectors...")
    steering_vectors = torch.load(steering_vectors_path)
    
    # Create MIST components
    localizer = FailureLocalizer(hooked_vla, detector)
    steerer = ActivationSteerer(hooked_vla, steering_vectors)
    
    orchestrator = MISTRecoveryOrchestrator(
        hooked_vla,
        detector,
        localizer,
        steerer
    )
    
    evaluator = MISTEvaluator(orchestrator)
    
    # Load benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[task_suite]()
    
    results = {
        'task_suite': task_suite,
        'n_episodes': n_episodes,
        'per_task_results': {},
        'aggregate': {}
    }
    
    all_successes = []
    all_failures_detected = []
    all_recoveries = []
    
    for task_id in tqdm(range(len(suite.tasks)), desc="Tasks"):
        env = suite.make_env(task_id)
        instruction = suite.get_task_instruction(task_id)
        
        task_results = []
        
        for episode in range(n_episodes // len(suite.tasks)):
            metrics = evaluator.evaluate_task(env, instruction, max_steps)
            task_results.append(metrics)
            
            all_successes.append(metrics['success'])
            all_failures_detected.append(metrics['n_failures_detected'])
            all_recoveries.append(metrics['n_recoveries_successful'])
            
        results['per_task_results'][task_id] = {
            'instruction': instruction,
            'success_rate': np.mean([r['success'] for r in task_results]),
            'avg_steps': np.mean([r['n_steps'] for r in task_results]),
            'avg_failures_detected': np.mean([r['n_failures_detected'] for r in task_results]),
        }
        
    # Aggregate results
    results['aggregate'] = {
        'overall_success_rate': np.mean(all_successes),
        'avg_failures_per_episode': np.mean(all_failures_detected),
        'avg_recoveries_per_episode': np.mean(all_recoveries),
        'recovery_rate': np.sum(all_recoveries) / max(np.sum(all_failures_detected), 1)
    }
    
    # Save results
    with open(save_path / f"{task_suite}_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n=== Results for {task_suite} ===")
    print(f"Overall Success Rate: {results['aggregate']['overall_success_rate']:.2%}")
    print(f"Avg Failures Detected: {results['aggregate']['avg_failures_per_episode']:.2f}")
    print(f"Recovery Rate: {results['aggregate']['recovery_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", default="libero_spatial")
    parser.add_argument("--model_path", default="openvla/openvla-7b")
    parser.add_argument("--detector_path", default="checkpoints/best_detector.pt")
    parser.add_argument("--steering_path", default="data/steering_vectors/all_vectors.pt")
    parser.add_argument("--n_episodes", type=int, default=50)
    args = parser.parse_args()
    
    evaluate_libero(
        args.task_suite,
        args.model_path,
        args.detector_path,
        args.steering_path,
        args.n_episodes
    )


PART 5: EXPERIMENT PLAN
5.1 Experimental Design
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXPERIMENT ROADMAP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: BASELINE ESTABLISHMENT (Week 1-2)                                 │
│  ───────────────────────────────────────────                                │
│  □ Reproduce SAFE baseline on LIBERO                                        │
│  □ Reproduce FailSafe baseline on LIBERO                                    │
│  □ Evaluate vanilla OpenVLA/pi0 (no intervention)                           │
│  □ Collect success/failure rollouts for detector training                   │
│                                                                             │
│  PHASE 2: COMPONENT DEVELOPMENT (Week 2-4)                                  │
│  ────────────────────────────────────────────                               │
│  □ Implement HookedOpenVLA wrapper                                          │
│  □ Train failure detector (MLP and LSTM variants)                           │
│  □ Calibrate conformal prediction                                           │
│  □ Implement attribution pipeline                                           │
│  □ Extract and cluster steering vectors                                     │
│                                                                             │
│  PHASE 3: INTEGRATION & ABLATIONS (Week 4-6)                                │
│  ───────────────────────────────────────────────                            │
│  □ Integrate all components into MIST orchestrator                          │
│  □ Ablation: Detection only (no steering)                                   │
│  □ Ablation: Steering only (no attribution)                                 │
│  □ Ablation: Random steering vs. attributed steering                        │
│  □ Ablation: Different layer ranges for steering                            │
│                                                                             │
│  PHASE 4: COMPREHENSIVE EVALUATION (Week 6-8)                               │
│  ─────────────────────────────────────────────                              │
│  □ Full LIBERO evaluation (all 5 suites)                                    │
│  □ LIBERO-PRO robustness evaluation                                         │
│  □ Cross-architecture: OpenVLA + pi0                                        │
│  □ Latency/overhead analysis                                                │
│  □ Failure case analysis and visualization                                  │
│                                                                             │
│  PHASE 5: PAPER WRITING (Week 8-10)                                         │
│  ───────────────────────────────────────                                    │
│  □ Write introduction and related work                                      │
│  □ Method section with architecture diagram                                 │
│  □ Experiments and results                                                  │
│  □ Analysis and ablations                                                   │
│  □ Conclusion and limitations                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

5.2 Metrics to Report
Metric
Description
Target
Success Rate
Task completion rate
Compare to baselines
Recovery Rate
% of detected failures that led to successful recovery
Novel metric
Detection Latency
Steps until failure detected
< 10 steps
Detection Accuracy
ROC-AUC for failure prediction
> 0.85
Explanation Quality
Human eval of failure explanations
Qualitative
Inference Overhead
Additional latency from MIST
< 5ms
Generalization
Performance on unseen tasks/objects
Key differentiator

5.3 Key Comparisons
MIST vs. SAFE: Detection + steering vs. detection only
MIST vs. FailSafe: Steering vs. VLM reasoning for recovery
MIST vs. FPC-VLA: Activation steering vs. supervisor model
MIST vs. SafeVLA: Snap-on vs. retrained safety

PART 6: KEY PAPER REFERENCES
Must-Cite Papers
SAFE (arXiv 2506.09937): Foundation for failure detection methodology
Mechanistic Interpretability for Steering VLAs (arXiv 2509.00328): Core steering technique
FailSafe (arXiv 2510.01642): Failure recovery baseline
FPC-VLA (arXiv 2509.04018): Supervisor-based correction baseline
SafeVLA (arXiv 2503.03480): Safety alignment baseline
OpenVLA (arXiv 2406.09246): Primary base model
π0/π0-FAST (arXiv 2410.24164): Secondary base model
LIBERO (arXiv 2306.03310): Primary benchmark
Additional References
TransformerLens documentation
Captum documentation
Integrated Gradients (Sundararajan et al. 2017)
Activation steering in LLMs (Turner et al. 2023)

PART 7: COMPUTE REQUIREMENTS
Minimum Setup
1x NVIDIA A100 40GB (or 2x RTX 4090)
64GB RAM
500GB SSD
Recommended Setup
2-4x NVIDIA A100 80GB
128GB RAM
1TB NVMe SSD
Cloud Options
Lambda Labs: ~$1.50/hr for A100 80GB
RunPod: ~$1.00/hr for A100 40GB
AWS: p4d.24xlarge for heavy training

GETTING STARTED CHECKLIST
# 1. Clone all repositories (see Part 1)

# 2. Set up environments
conda create -n mist-vla python=3.10
conda activate mist-vla
# Install dependencies (see Part 1)

# 3. Download models
# OpenVLA
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b')"

# 4. Download LIBERO data
cd LIBERO
python benchmark_scripts/download_libero_datasets.py
cd ..

# 5. Collect initial data
python scripts/collect_failure_data.py --env libero_spatial --n_success 100 --n_failure 100

# 6. Train detector
python scripts/train_failure_detector.py --detector_type mlp --epochs 50

# 7. Extract steering vectors
python scripts/extract_steering_vectors.py

# 8. Run evaluation
python scripts/run_libero_eval.py --task_suite libero_spatial

# 9. Celebrate! 🎉


END OF BLUEPRINT
This document provides a complete roadmap for implementing MIST-VLA. Follow the phases in order, and refer back to specific sections as needed. Good luck with your research!

