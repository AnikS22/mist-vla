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
        device_map: Optional[str] = None,
        model: Optional[AutoModelForVision2Seq] = None,
        processor: Optional[AutoProcessor] = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.device_map = device_map
        self.processor = processor or AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if model is None:
            # Try FlashAttention2 on Ampere+ GPUs; fall back to eager otherwise.
            attn_impl = "eager"
            if device.startswith("cuda") and torch.cuda.is_available():
                try:
                    major, _minor = torch.cuda.get_device_capability()
                    if major >= 8:
                        import flash_attn  # noqa: F401

                        attn_impl = "flash_attention_2"
                except Exception:
                    attn_impl = "eager"

            load_kwargs = dict(
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            )
            if device_map is not None and device_map.lower() != "none":
                load_kwargs["device_map"] = device_map
                load_kwargs["low_cpu_mem_usage"] = True

            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **load_kwargs,
            )
            if device_map is None or device_map.lower() == "none":
                self.model = self.model.to(device)
        else:
            self.model = model

        self.collector = HiddenStateCollector(self.model)
        self.collector.register_hooks()

    def _resolve_unnorm_key(self) -> Optional[str]:
        if not hasattr(self.model, "norm_stats"):
            return None
        norm_stats = getattr(self.model, "norm_stats", {})
        if not isinstance(norm_stats, dict) or not norm_stats:
            return None
        # Prefer commonly used keys for LIBERO / bridge evaluations.
        preferred = [
            "libero_spatial_no_noops",
            "libero_spatial",
            "bridge_orig",
        ]
        for k in preferred:
            if k in norm_stats:
                return k
        return next(iter(norm_stats.keys()))

    def _input_device(self) -> torch.device:
        if hasattr(self.model, "hf_device_map"):
            # Pick the first CUDA shard, else first listed device.
            vals = list(self.model.hf_device_map.values())
            for v in vals:
                if isinstance(v, str) and v.startswith("cuda"):
                    return torch.device(v)
            for v in vals:
                if isinstance(v, str):
                    return torch.device(v)
        return torch.device(self.device)

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        if torch.is_tensor(image):
            return Image.fromarray(image.detach().cpu().numpy())
        return image

    def _prepare_inputs(self, image, instruction: str) -> dict:
        # Match OpenVLA/OFT evaluation prompt style used in LIBERO pipelines.
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        image = self._to_pil(image)
        inputs = self.processor(prompt, image, return_tensors="pt")
        input_device = self._input_device()
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                input_device, dtype=self.model.dtype
            )
        return inputs

    def _generate_action(self, inputs: dict) -> torch.Tensor:
        # Preferred path: custom OpenVLA trust_remote_code API.
        if hasattr(self.model, "predict_action"):
            unnorm_key = self._resolve_unnorm_key()
            try:
                if unnorm_key is not None:
                    act = self.model.predict_action(**inputs, unnorm_key=unnorm_key)
                else:
                    act = self.model.predict_action(**inputs)
                action_np = np.asarray(act, dtype=np.float32).reshape(-1)
                return torch.from_numpy(action_np[:7]).to(self._input_device())
            except TypeError:
                # Some forks accept no unnorm_key; retry minimal signature.
                act = self.model.predict_action(**inputs)
                action_np = np.asarray(act, dtype=np.float32).reshape(-1)
                return torch.from_numpy(action_np[:7]).to(self._input_device())
            except Exception:
                # Fall back to token decoding path below.
                pass

        # Fallback path for models that do not expose predict_action.
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
    device_map: Optional[str] = None,
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
    return OpenVLAWrapper(
        model_name,
        device=device,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

