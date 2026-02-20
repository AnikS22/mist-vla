"""
HookedOpenVLA: load OpenVLA and expose hidden-state features.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from src.data_collection.hooks import HiddenStateCollector


class HookedOpenVLA:
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

    def get_last_layer_features(
        self,
        image,
        instruction: str,
        pool: str = "mean",
    ) -> torch.Tensor:
        self.collector.clear()
        self.collector._active = True
        inputs = self._prepare_inputs(image, instruction)
        with torch.no_grad():
            _ = self.model(**inputs)
        self.collector._active = False
        return self.collector.get_last_layer(pool=pool)

    def close(self) -> None:
        self.collector.remove_hooks()
        del self.model

