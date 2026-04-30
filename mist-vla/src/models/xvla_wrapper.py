"""
LeRobot XVLA ("SmolVLA-like") wrapper for action generation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer


class XVLAWrapper:
    def __init__(self, model_name: str = "lerobot/xvla-base", device: str = "cuda") -> None:
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.policy = XVLAPolicy.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.policy.config.tokenizer_name)

    def _to_np_rgb(self, image) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            arr = image
        elif torch.is_tensor(image):
            arr = image.detach().cpu().numpy()
        else:
            arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _prep_img(self, image: np.ndarray, size: int) -> torch.Tensor:
        pil = Image.fromarray(image).resize((size, size), Image.BICUBIC)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        # CHW, ImageNet normalization (matches XVLA processor stack).
        chw = np.transpose(arr, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        chw = (chw - mean) / std
        return torch.from_numpy(chw).to(self.device)

    def _extract_state(self, obs: Optional[dict]) -> torch.Tensor:
        if obs and isinstance(obs, dict):
            if "robot0_proprio-state" in obs:
                s = np.asarray(obs["robot0_proprio-state"], dtype=np.float32).reshape(-1)
                if s.shape[0] >= 8:
                    return torch.from_numpy(s[:8]).to(self.device)
            if "state" in obs:
                s = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
                if s.shape[0] >= 8:
                    return torch.from_numpy(s[:8]).to(self.device)
        return torch.zeros(8, dtype=torch.float32, device=self.device)

    def get_action_with_features(self, image, instruction: str, obs: Optional[dict] = None):
        rgb = self._to_np_rgb(image)
        # Use a single camera view to keep sequence length within model limits.
        img3 = self._prep_img(rgb, 224).unsqueeze(0)
        toks = self.tokenizer(
            instruction,
            max_length=self.policy.config.tokenizer_max_length,
            padding=self.policy.config.pad_language_to,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        state = self._extract_state(obs).unsqueeze(0)
        batch = {
            "observation.language.tokens": toks,
            "observation.images.image3": img3,
            "observation.state": state,
        }
        with torch.no_grad():
            act = self.policy.select_action(batch)
        act = act.reshape(-1)
        # XVLA outputs padded action; take first 7 for xyz/rpy/gripper contract.
        action7 = act[:7].detach()
        features = action7.unsqueeze(0)
        return action7, features

    def close(self) -> None:
        del self.policy
