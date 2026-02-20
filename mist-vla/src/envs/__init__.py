"""
Multi-embodiment environment wrappers.

Provides UniversalEnvWrapper for standardized interaction with:
  - LIBERO (Franka Panda)
  - SimplerEnv (Google Robot, WidowX)
"""

from src.envs.universal_wrapper import UniversalEnvWrapper, make_env

__all__ = ["UniversalEnvWrapper", "make_env"]
