"""
Minimal OpenVLA wrapper for action generation + feature extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

from src.data_collection.hooks import HiddenStateCollector


class OpenVLAWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model: Optional[AutoModelForVision2Seq] = None,
        processor: Optional[AutoProcessor] = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.processor = processor or AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if model is None:
            # Try FlashAttention2; fall back to eager if unavailable.
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "eager"

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            ).to(device)
        else:
            self.model = model

        self.collector = HiddenStateCollector(self.model)
        self.collector.register_hooks()

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        if torch.is_tensor(image):
            return Image.fromarray(image.detach().cpu().numpy())
        return image

    def _prepare_inputs(self, image, instruction: str) -> dict:
        prompt = f"In: {instruction}\nOut:"
        image = self._to_pil(image)
        inputs = self.processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self.device, dtype=self.model.dtype
            )
        return inputs

    def _generate_action(self, inputs: dict) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
            )
        action_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        action = (action_tokens.float() - 128) / 128
        action = torch.clamp(action, -1, 1)
        return action[0]

    def _extract_features(self, inputs: dict) -> torch.Tensor:
        self.collector.clear()
        self.collector._active = True
        with torch.no_grad():
            _ = self.model(**inputs)
        self.collector._active = False
        return self.collector.get_last_layer(pool="mean")

    def get_action_with_features(self, image, instruction: str, obs: Optional[dict] = None):
        inputs = self._prepare_inputs(image, instruction)
        action = self._generate_action(inputs)
        features = self._extract_features(inputs)
        return action, features

    def close(self) -> None:
        self.collector.remove_hooks()
        del self.model


def create_vla_wrapper(
    model_type: str,
    model_name: str,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> OpenVLAWrapper:
    """
    Factory for VLA wrappers. Currently supports OpenVLA.
    """
    if model_type == "openvla_oft":
        from src.models.openvla_oft_wrapper import OpenVLAOFTWrapper

        return OpenVLAOFTWrapper(model_name, device=device)
    if model_type != "openvla":
        raise ValueError(f"Unsupported model_type: {model_type}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        if device == "cuda":
            # Turing GPUs (e.g., 2080 Ti) do not support bfloat16 well.
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    return OpenVLAWrapper(model_name, device=device, torch_dtype=torch_dtype)

