"""
Site customizations loaded automatically by Python if on PYTHONPATH.
Ensures torch.xpu exists for diffusers compatibility.
"""
import types

try:
    import torch

    if not hasattr(torch, "xpu"):
        class _XPU:
            def empty_cache(self):  # noqa: D401
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

            def _is_in_bad_fork(self):
                return False

        torch.xpu = _XPU()
except Exception:
    pass
