"""
Minimal OpenVLA wrapper for action generation + feature extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
import transformers
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

from src.data_collection.hooks import HiddenStateCollector


def _force_eager_attention() -> bool:
    """
    Use eager attention only for newer transformers stacks where OpenVLA dynamic modules
    can fail on SDPA capability checks. Keep default attention on the known-good 4.40.x stack.
    """
    try:
        major, minor, *_ = [int(x) for x in transformers.__version__.split(".")]
    except Exception:
        return False
    return (major, minor) >= (4, 56)


class OpenVLAWrapper:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        allow_token_fallback: bool = False,
        force_token_fallback: bool = False,
        enable_hidden_state_hooks: bool = False,
        model: Optional[AutoModelForVision2Seq] = None,
        processor: Optional[AutoProcessor] = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.device_map = device_map
        self.allow_token_fallback = allow_token_fallback
        self.force_token_fallback = force_token_fallback
        self.enable_hidden_state_hooks = enable_hidden_state_hooks
        self.processor = processor or AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if model is None:
            load_kwargs = dict(
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            if _force_eager_attention():
                load_kwargs["attn_implementation"] = "eager"
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

        self.collector = None
        if self.enable_hidden_state_hooks:
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
            # For sharded HF models, feed inputs to cuda:0 (root shard) for stable generation.
            vals = list(self.model.hf_device_map.values())
            if any(isinstance(v, str) and v.startswith("cuda") for v in vals):
                return torch.device("cuda:0")
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
        if instruction.startswith("RAW_PROMPT::"):
            prompt = instruction[len("RAW_PROMPT::") :]
        # Allow fully raw prompts when the caller provides a complete prompt.
        elif "In:" in instruction and "Out:" in instruction:
            prompt = instruction
        else:
            # Default OpenVLA-style prompt template.
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
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
        if (not self.force_token_fallback) and hasattr(self.model, "predict_action"):
            unnorm_key = self._resolve_unnorm_key()
            def _to_np_action(act_obj):
                if isinstance(act_obj, (list, tuple)) and len(act_obj) > 0:
                    act_obj = act_obj[0]
                if torch.is_tensor(act_obj):
                    return act_obj.detach().float().cpu().numpy().reshape(-1)
                return np.asarray(act_obj, dtype=np.float32).reshape(-1)
            try:
                if unnorm_key is not None:
                    try:
                        act = self.model.predict_action(**inputs, unnorm_key=unnorm_key)
                    except TypeError:
                        # Some forks accept no unnorm_key; retry minimal signature.
                        act = self.model.predict_action(**inputs)
                else:
                    act = self.model.predict_action(**inputs)
                action_np = _to_np_action(act)
                return torch.from_numpy(action_np[:7]).to(self._input_device())
            except Exception as e:
                if not self.allow_token_fallback:
                    raise RuntimeError(
                        f"OpenVLA predict_action failed for {self.model_name}; "
                        f"token fallback disabled to avoid degenerate actions. root_error={type(e).__name__}: {e}"
                    ) from e

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
        if self.collector is None:
            return torch.zeros((1, 1), device=self._input_device())
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
        if self.collector is not None:
            self.collector.remove_hooks()
        del self.model


def create_vla_wrapper(
    model_type: str,
    model_name: str,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    allow_token_fallback: bool = False,
    force_token_fallback: bool = False,
    enable_hidden_state_hooks: bool = False,
) -> OpenVLAWrapper:
    """
    Factory for VLA wrappers. Currently supports OpenVLA.
    """
    if model_type == "smolvla":
        from src.models.xvla_wrapper import XVLAWrapper

        return XVLAWrapper(model_name=model_name, device=device or "cuda")
    if model_type == "openvla_oft":
        from src.models.openvla_oft_wrapper import OpenVLAOFTWrapper
        try:
            return OpenVLAOFTWrapper(model_name, device=device, device_map=device_map)
        except Exception as e:
            # Keep the control stack usable when OFT env/runtime is brittle on low-VRAM hosts.
            print(f"[vla_wrapper] OpenVLA-OFT init failed; falling back to OpenVLA path: {e}")
            return OpenVLAWrapper(
                model_name,
                device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=torch_dtype or (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
                device_map=device_map or "auto",
                allow_token_fallback=allow_token_fallback,
                force_token_fallback=force_token_fallback,
                enable_hidden_state_hooks=enable_hidden_state_hooks,
            )
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
        allow_token_fallback=allow_token_fallback,
        force_token_fallback=force_token_fallback,
        enable_hidden_state_hooks=enable_hidden_state_hooks,
    )

