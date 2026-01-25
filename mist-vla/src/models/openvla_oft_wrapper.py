"""
OpenVLA-OFT wrapper for action generation + hidden state extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Avoid importing prismatic at module import time (pulls in heavy deps like dlimp).
PROPRIO_DIM = 8


def _ensure_openvla_oft_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    oft_path = repo_root / "openvla-oft"
    if oft_path.exists() and str(oft_path) not in sys.path:
        sys.path.insert(0, str(oft_path))


class OpenVLAOFTWrapper:
    def __init__(
        self,
        pretrained_checkpoint: str,
        unnorm_key: str = "libero_spatial_no_noops",
        device: Optional[str] = None,
    ) -> None:
        _ensure_openvla_oft_on_path()

        # Some diffusers versions expect torch.xpu to exist.
        if not hasattr(torch, "xpu"):
            class _XPU:
                def empty_cache(self):
                    return None

                def device_count(self):
                    return 0

                def is_available(self):
                    return False

                def manual_seed(self, *args, **kwargs):
                    return None

                def manual_seed_all(self, *args, **kwargs):
                    return None

                def device(self, *args, **kwargs):
                    return None

                def current_device(self):
                    return 0

            torch.xpu = _XPU()

        # Ensure torch.load maps to CPU when CUDA is unavailable (OFT loads CUDA tensors).
        if not torch.cuda.is_available():
            _orig_torch_load = torch.load

            def _cpu_torch_load(*args, **kwargs):
                if "map_location" not in kwargs:
                    kwargs["map_location"] = torch.device("cpu")
                return _orig_torch_load(*args, **kwargs)

            torch.load = _cpu_torch_load

        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
            normalize_proprio,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.cfg = GenerateConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=1,
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            num_open_loop_steps=8,
            unnorm_key=unnorm_key,
        )

        self.vla = get_vla(self.cfg)
        self._resolve_unnorm_key()
        self.processor = get_processor(self.cfg)
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(
            self.cfg,
            llm_dim=self.vla.llm_dim,
            proprio_dim=PROPRIO_DIM,
        )
        self.get_vla_action = get_vla_action
        self.normalize_proprio = normalize_proprio

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                return Image.fromarray(image)
            return Image.fromarray((image * 255).astype(np.uint8))
        if torch.is_tensor(image):
            return Image.fromarray(image.detach().cpu().numpy())
        return image

    def _resolve_unnorm_key(self) -> None:
        unnorm_key = self.cfg.task_suite_name if hasattr(self.cfg, "task_suite_name") else "libero_spatial"
        if unnorm_key not in self.vla.norm_stats and f"{unnorm_key}_no_noops" in self.vla.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        if unnorm_key in self.vla.norm_stats:
            self.cfg.unnorm_key = unnorm_key

    def _build_inputs(self, image: Image.Image, instruction: str, obs: Optional[dict] = None):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, image)
        if "labels" in inputs:
            inputs.pop("labels")
        dtype = None
        if hasattr(self.vla, "llm_backbone") and hasattr(self.vla.llm_backbone, "half_precision_dtype"):
            dtype = self.vla.llm_backbone.half_precision_dtype
        elif hasattr(self.vla, "dtype"):
            dtype = self.vla.dtype
        else:
            dtype = torch.float32
        for key, val in inputs.items():
            if torch.is_tensor(val):
                if val.is_floating_point():
                    inputs[key] = val.to(self.device, dtype=dtype)
                else:
                    inputs[key] = val.to(self.device)
        return inputs

    def _extract_proprio(self, obs: Optional[dict]) -> Optional[np.ndarray]:
        if not obs:
            return None
        if "robot0_proprio-state" in obs:
            proprio = np.asarray(obs["robot0_proprio-state"], dtype=np.float32)
            if proprio.shape[0] == PROPRIO_DIM:
                return proprio
        if "state" in obs:
            proprio = np.asarray(obs["state"], dtype=np.float32)
            if proprio.shape[0] == PROPRIO_DIM:
                return proprio
        if all(k in obs for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")):
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
            eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
            gripper_scalar = float(gripper.mean()) if gripper.size else 0.0
            proprio = np.concatenate([eef_pos[:3], eef_quat[:4], np.array([gripper_scalar], dtype=np.float32)])
            if proprio.shape[0] == PROPRIO_DIM:
                return proprio
        return None

    def get_action_with_features(self, image, instruction: str, obs: Optional[dict] = None):
        image = self._to_pil(image)

        inputs = self._build_inputs(image, instruction, obs=obs)

        proprio = None
        if self.cfg.use_proprio:
            proprio = self._extract_proprio(obs)
            if proprio is not None:
                norm_stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
                proprio = self.normalize_proprio(proprio, norm_stats)

        actions, actions_hidden_states = self.vla.predict_action(
            **inputs,
            unnorm_key=self.cfg.unnorm_key,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            use_film=self.cfg.use_film,
        )
        action = torch.from_numpy(actions[0]).to(self.device)

        if actions_hidden_states is None:
            hidden_states = torch.zeros((1, self.vla.config.hidden_size), device=self.device)
        elif actions_hidden_states.ndim == 3:
            hidden_states = actions_hidden_states.mean(dim=1)
        else:
            hidden_states = actions_hidden_states

        return action, hidden_states.detach()

    def close(self) -> None:
        del self.vla
