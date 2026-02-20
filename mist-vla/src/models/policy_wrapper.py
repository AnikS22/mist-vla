"""
Module 3 — Multi-Model Policy Adapter
=======================================

Provides a unified `.predict(obs, return_embedding=True)` interface for
multiple VLA architectures:

  ┌────────────┐   ┌────────────┐   ┌─────────────┐
  │  OpenVLA    │   │  OpenVLA   │   │   Octo      │
  │ (Standard)  │   │  (OFT)     │   │ (Diffusion) │
  └──────┬──────┘   └──────┬─────┘   └──────┬──────┘
         │                  │                 │
         └──────────┬───────┴─────────────────┘
                    │
            ┌───────▼────────┐
            │ PolicyAdapter  │
            │                │
            │ .predict(obs)  │
            │  → action (7,) │
            │  → embed (D,)  │
            └────────────────┘

The embedding is the key input to our SafetyMLP.

Usage:
    # OpenVLA (Llama backbone)
    policy = PolicyAdapter.from_openvla("openvla/openvla-7b")
    action, embed = policy.predict(obs, return_embedding=True)

    # OpenVLA-OFT (finetuned)
    policy = PolicyAdapter.from_openvla_oft(
        "moojink/openvla-7b-oft-finetuned-libero-spatial")
    action, embed = policy.predict(obs, return_embedding=True)

    # Octo (Diffusion transformer)
    policy = PolicyAdapter.from_octo("hf://rail-berkeley/octo-base-1.5")
    action, embed = policy.predict(obs, return_embedding=True)

Architecture agnosticism:
    The SafetyMLP only sees a flat embedding vector (D,). It does NOT
    know whether it came from a Llama decoder, a ViT encoder, or a
    diffusion transformer. This is the core of the universality claim.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class PolicyAdapter(abc.ABC):
    """Unified interface for VLA policies.

    Every adapter must expose:
        .predict(obs, return_embedding=True) → (action, embedding)
        .embedding_dim → int
        .model_name → str
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self._model_name = model_name
        self.device = device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the embedding vector (e.g. 4096 for Llama-7B)."""
        ...

    @abc.abstractmethod
    def predict(self, obs: Dict[str, Any],
                return_embedding: bool = True
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate an action from an observation.

        Args:
            obs:              standardized observation dict with at least
                              "image" (H,W,3 uint8) and "instruction" (str)
            return_embedding: if True, also return the bottleneck embedding

        Returns:
            action:    (7,) float32 — task-space action
            embedding: (D,) float32 — bottleneck hidden state (or None)
        """
        ...

    @abc.abstractmethod
    def close(self):
        """Release GPU memory."""
        ...

    # ── Factories ────────────────────────────────────────────────────────

    @staticmethod
    def from_openvla(model_name: str = "openvla/openvla-7b",
                     device: str = "cuda") -> "OpenVLAAdapter":
        """Create adapter for standard OpenVLA (Llama backbone)."""
        return OpenVLAAdapter(model_name=model_name, device=device)

    @staticmethod
    def from_openvla_oft(model_name: str,
                         unnorm_key: str = "libero_spatial_no_noops",
                         device: str = "cuda") -> "OpenVLAOFTAdapter":
        """Create adapter for OpenVLA-OFT (finetuned variant)."""
        return OpenVLAOFTAdapter(model_name=model_name,
                                 unnorm_key=unnorm_key,
                                 device=device)

    @staticmethod
    def from_octo(model_name: str = "hf://rail-berkeley/octo-base-1.5",
                  device: str = "cuda") -> "OctoAdapter":
        """Create adapter for Octo (diffusion transformer)."""
        return OctoAdapter(model_name=model_name, device=device)


# ──────────────────────────────────────────────────────────────────────────
# OpenVLA Adapter (Standard — Llama backbone)
# ──────────────────────────────────────────────────────────────────────────

class OpenVLAAdapter(PolicyAdapter):
    """Adapter for standard OpenVLA with Llama-2/3 backbone.

    Embedding: mean-pooled last-layer hidden state from the Llama decoder.
    Dimension: 4096 (Llama-7B) or 2048 (Llama-3B).
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self._wrapper = None
        self._embed_dim = None
        self._load_model()

    def _load_model(self):
        from src.models.vla_wrapper import OpenVLAWrapper
        self._wrapper = OpenVLAWrapper(
            model_name=self._model_name,
            device=self.device,
        )
        # Infer embedding dim
        self._embed_dim = self._wrapper.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def predict(self, obs: Dict[str, Any],
                return_embedding: bool = True
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        image = obs.get("image")
        instruction = obs.get("instruction", "")

        action, hidden = self._wrapper.get_action_with_features(
            image, instruction, obs=obs
        )

        action_np = action.detach().cpu().numpy().astype(np.float32)
        # Pad to 7 if needed
        if action_np.shape[0] < 7:
            action_np = np.pad(action_np, (0, 7 - action_np.shape[0]))
        action_np = action_np[:7]

        if return_embedding:
            embed_np = hidden.detach().cpu().numpy().astype(np.float32)
            if embed_np.ndim == 2:
                embed_np = embed_np.squeeze(0)
            return action_np, embed_np
        return action_np, None

    def close(self):
        if self._wrapper is not None:
            self._wrapper.close()
            self._wrapper = None


# ──────────────────────────────────────────────────────────────────────────
# OpenVLA-OFT Adapter (Finetuned — Llama backbone + action head)
# ──────────────────────────────────────────────────────────────────────────

class OpenVLAOFTAdapter(PolicyAdapter):
    """Adapter for OpenVLA-OFT (finetuned on specific benchmarks).

    Uses the OFT action head for more precise actions.
    Embedding: actions_hidden_states from the predict_action call.
    Dimension: typically 4096.
    """

    def __init__(self, model_name: str,
                 unnorm_key: str = "libero_spatial_no_noops",
                 device: str = "cuda"):
        super().__init__(model_name, device)
        self._unnorm_key = unnorm_key
        self._wrapper = None
        self._embed_dim = None
        self._load_model()

    def _load_model(self):
        from src.models.openvla_oft_wrapper import OpenVLAOFTWrapper
        self._wrapper = OpenVLAOFTWrapper(
            pretrained_checkpoint=self._model_name,
            unnorm_key=self._unnorm_key,
            device=self.device,
        )
        # Infer embedding dim
        self._embed_dim = self._wrapper.vla.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def predict(self, obs: Dict[str, Any],
                return_embedding: bool = True
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        image = obs.get("image")
        instruction = obs.get("instruction", "")

        action, hidden = self._wrapper.get_action_with_features(
            image, instruction, obs=obs
        )

        action_np = action.detach().cpu().numpy().astype(np.float32)
        if action_np.shape[0] < 7:
            action_np = np.pad(action_np, (0, 7 - action_np.shape[0]))
        action_np = action_np[:7]

        if return_embedding:
            embed_np = hidden.detach().cpu().numpy().astype(np.float32)
            if embed_np.ndim == 2:
                embed_np = embed_np.squeeze(0)
            return action_np, embed_np
        return action_np, None

    def close(self):
        if self._wrapper is not None:
            self._wrapper.close()
            self._wrapper = None


# ──────────────────────────────────────────────────────────────────────────
# Octo Adapter (Diffusion Transformer)
# ──────────────────────────────────────────────────────────────────────────

class OctoAdapter(PolicyAdapter):
    """Adapter for the Octo model (diffusion policy, transformer backbone).

    Octo architecture:
        Image → ViT → Tokens
        Language → Tokenizer → Tokens
        [Image Tokens | Language Tokens] → Transformer → Bottleneck → Diffusion Head

    Embedding extraction:
        We hook the transformer's last layer output (before the diffusion
        head) and mean-pool it. This is analogous to our Llama hidden state
        extraction in OpenVLA.

    Dimension: 512 (Octo-Base) or 256 (Octo-Small).

    Important: The SafetyMLP input_dim must match. When training on Octo
    embeddings, you need to either:
        1. Train a separate SafetyMLP with input_dim=512, or
        2. Project Octo embeddings to 4096-d via a learned projector
           (for cross-architecture universality).
    """

    def __init__(self, model_name: str = "hf://rail-berkeley/octo-base-1.5",
                 device: str = "cuda"):
        super().__init__(model_name, device)
        self._model = None
        self._embed_dim = None
        self._cached_embedding = None
        self._hook_handle = None
        self._load_model()

    def _load_model(self):
        """Load Octo model and register embedding extraction hook."""
        try:
            from octo.model.octo_model import OctoModel
        except ImportError:
            raise ImportError(
                "Octo not installed. Install with:\n"
                "  pip install octo\n"
                "or: git clone https://github.com/octo-models/octo"
            )

        self._model = OctoModel.load_pretrained(self._model_name)

        # Octo uses a JAX-based transformer. The bottleneck embedding
        # is extracted from the model's `transformer_outputs`.
        # For JAX models, we cannot use PyTorch hooks directly.
        # Instead, we capture the embedding during the predict call.

        # Infer embedding dim from model config
        if hasattr(self._model, 'config'):
            config = self._model.config
            if hasattr(config, 'model') and hasattr(config.model, 'transformer_kwargs'):
                tk = config.model.transformer_kwargs
                self._embed_dim = tk.get('token_embedding_size', 512)
            else:
                self._embed_dim = 512
        else:
            self._embed_dim = 512

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def predict(self, obs: Dict[str, Any],
                return_embedding: bool = True
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action and extract embedding from Octo.

        Octo's predict_action returns a dictionary. We also call the
        transformer directly to extract the bottleneck embedding.
        """
        import jax
        import jax.numpy as jnp

        image = obs.get("image")
        instruction = obs.get("instruction", "")

        # Prepare Octo observation format
        if isinstance(image, np.ndarray):
            # Ensure (1, H, W, 3) for batching
            if image.ndim == 3:
                image = image[np.newaxis]

        octo_obs = {
            "image_primary": jnp.array(image),
            "pad_mask": jnp.ones((1, 1)),
        }

        # Create task from language instruction
        task = self._model.create_tasks(texts=[instruction])

        # Get action
        action_output = self._model.sample_actions(
            octo_obs,
            task,
            rng=jax.random.PRNGKey(0),
        )
        action_np = np.array(action_output).squeeze().astype(np.float32)

        # Pad/trim to 7-DoF
        if action_np.shape[-1] < 7:
            action_np = np.pad(action_np, (0, 7 - action_np.shape[-1]))
        action_np = action_np[:7]

        # Extract embedding from transformer
        if return_embedding:
            embed = self._extract_embedding(octo_obs, task)
            return action_np, embed
        return action_np, None

    def _extract_embedding(self, octo_obs, task) -> np.ndarray:
        """Extract the bottleneck embedding from Octo's transformer.

        Octo's forward pass:
            1. Tokenize image + language → token sequence
            2. Pass through transformer layers
            3. Extract "readout" tokens (bottleneck)
            4. Feed readout tokens to action head (diffusion)

        We extract the readout tokens (step 3) as our embedding.
        This is Octo's equivalent of the Llama last-layer hidden state.
        """
        import jax.numpy as jnp

        try:
            # Run the transformer forward pass to get intermediate outputs
            transformer_outputs = self._model.run_transformer(
                octo_obs, task, timestep_pad_mask=jnp.ones((1, 1))
            )

            # The readout tokens are the bottleneck embedding
            # Shape: (batch, n_readout_tokens, embed_dim)
            if hasattr(transformer_outputs, 'keys'):
                # Dict-like output — get the readout key
                if "readout_action" in transformer_outputs:
                    readout = transformer_outputs["readout_action"]
                elif "readout" in transformer_outputs:
                    readout = transformer_outputs["readout"]
                else:
                    # Take the last key as the readout
                    readout = list(transformer_outputs.values())[-1]
            else:
                readout = transformer_outputs

            # Mean-pool over tokens → (embed_dim,)
            embed = np.array(readout).squeeze()
            if embed.ndim == 2:
                embed = embed.mean(axis=0)
            elif embed.ndim == 3:
                embed = embed.squeeze(0).mean(axis=0)

            return embed.astype(np.float32)

        except Exception as e:
            print(f"  Warning: Octo embedding extraction failed: {e}")
            return np.zeros(self._embed_dim, dtype=np.float32)

    def close(self):
        self._model = None
        import gc
        gc.collect()


# ──────────────────────────────────────────────────────────────────────────
# Embedding Projector (for cross-architecture universality)
# ──────────────────────────────────────────────────────────────────────────

class EmbeddingProjector(torch.nn.Module):
    """Projects embeddings from any VLA to a common dimension.

    This allows a SINGLE SafetyMLP to work across architectures:
        OpenVLA  (4096-d) ─┐
        Octo-Base (512-d) ─┤──→ Projector ──→ (D_common,) ──→ SafetyMLP
        Octo-Small (256-d)─┘

    The projector is a lightweight linear + LayerNorm layer.
    """

    def __init__(self, source_dims: Dict[str, int],
                 common_dim: int = 512):
        """
        Args:
            source_dims: {"openvla": 4096, "octo_base": 512, ...}
            common_dim:  target embedding dimension for the SafetyMLP
        """
        super().__init__()
        self.projectors = torch.nn.ModuleDict()
        self.common_dim = common_dim

        for name, dim in source_dims.items():
            self.projectors[name] = torch.nn.Sequential(
                torch.nn.Linear(dim, common_dim),
                torch.nn.LayerNorm(common_dim),
                torch.nn.GELU(),
            )

    def forward(self, embedding: torch.Tensor,
                source: str) -> torch.Tensor:
        """Project an embedding to the common space.

        Args:
            embedding: (B, D_source) embedding from a specific VLA
            source:    key identifying the VLA (e.g. "openvla", "octo_base")

        Returns:
            (B, D_common) projected embedding
        """
        if source not in self.projectors:
            raise ValueError(
                f"Unknown source '{source}'. "
                f"Available: {list(self.projectors.keys())}"
            )
        return self.projectors[source](embedding)
