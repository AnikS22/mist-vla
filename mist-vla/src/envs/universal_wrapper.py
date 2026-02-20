"""
Module 2 — Universal Environment Wrapper
==========================================

Standardizes the observation/action interface across different robot
environments so the same SafetyMLP and data-collection pipeline can run
on any platform:

  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │   LIBERO     │   │ Google Robot│   │  WidowX     │
  │ (Franka 7D)  │   │ (7D + base) │   │ (6D)        │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │
         └───────────┬───────┴───────────────────┘
                     │
             ┌───────▼────────┐
             │ UniversalEnv   │
             │ Wrapper        │
             │                │
             │ obs = {        │
             │  "image": ..., │
             │  "proprio": ...│
             │  "eef_pos":... │
             │ }              │
             │                │
             │ action = (7,)  │
             │ dx,dy,dz,      │
             │ droll,dpitch,  │
             │ dyaw, grip     │
             └────────────────┘

Supported platforms:
  - "libero"      : LIBERO benchmark (Franka Panda, 7-DoF)
  - "google_robot": SimplerEnv Google Robot (7-DoF arm + mobile base)
  - "widowx"      : SimplerEnv WidowX (6-DoF)
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Standard observation keys
OBS_IMAGE = "image"           # (H, W, 3) uint8 RGB
OBS_PROPRIO = "proprio"       # (D,) float32 proprioception
OBS_EEF_POS = "eef_pos"       # (3,) float32 Cartesian EEF position
OBS_EEF_QUAT = "eef_quat"     # (4,) float32 EEF quaternion
OBS_GRIPPER = "gripper"       # (1,) float32 gripper state
OBS_INSTRUCTION = "instruction"  # str — language task description

# Standard action space: 7-DoF task-space
#   [dx, dy, dz, droll, dpitch, dyaw, gripper]
ACTION_DIM = 7


class UniversalEnvWrapper(abc.ABC):
    """Abstract base class for all environment wrappers.

    Every wrapper must produce standardized observations and accept
    standardized 7-DoF task-space actions.
    """

    def __init__(self, env_name: str, task_id: int = 0,
                 instruction: str = ""):
        self.env_name = env_name
        self.task_id = task_id
        self.instruction = instruction
        self._step_count = 0

    @abc.abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return a standardized observation."""
        ...

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Execute an action and return (obs, reward, done, info).

        Args:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, grip]

        Returns:
            obs:   standardized observation dict
            reward: scalar reward
            done:   episode complete flag
            info:   extra info (e.g. {"success": True/False})
        """
        ...

    @abc.abstractmethod
    def get_eef_pos(self) -> np.ndarray:
        """Return the current EEF Cartesian position (3,)."""
        ...

    @abc.abstractmethod
    def close(self):
        """Clean up the environment."""
        ...

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def _standardize_obs(self, image: np.ndarray,
                         proprio: Optional[np.ndarray] = None,
                         eef_pos: Optional[np.ndarray] = None,
                         eef_quat: Optional[np.ndarray] = None,
                         gripper: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Build a standardized observation dict."""
        obs = {
            OBS_IMAGE: np.asarray(image, dtype=np.uint8),
            OBS_INSTRUCTION: self.instruction,
        }
        if proprio is not None:
            obs[OBS_PROPRIO] = np.asarray(proprio, dtype=np.float32)
        if eef_pos is not None:
            obs[OBS_EEF_POS] = np.asarray(eef_pos, dtype=np.float32).reshape(3)
        if eef_quat is not None:
            obs[OBS_EEF_QUAT] = np.asarray(eef_quat, dtype=np.float32).reshape(4)
        if gripper is not None:
            obs[OBS_GRIPPER] = np.asarray(gripper, dtype=np.float32).reshape(-1)
        return obs


# ──────────────────────────────────────────────────────────────────────────
# LIBERO Wrapper
# ──────────────────────────────────────────────────────────────────────────

class LIBEROWrapper(UniversalEnvWrapper):
    """Wraps a LIBERO environment (Franka Panda 7-DoF).

    Action space is natively 7-DoF task-space, so no conversion needed.
    """

    def __init__(self, task_suite: str = "libero_spatial", task_id: int = 0,
                 resolution: int = 256):
        super().__init__(f"libero/{task_suite}", task_id=task_id)
        self.task_suite = task_suite
        self.resolution = resolution
        self._env = None
        self._task = None
        self._init_libero(task_id)

    def _init_libero(self, task_id: int):
        """Initialize the LIBERO environment for a specific task."""
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        bm = benchmark_dict[self.task_suite]()
        task = bm.get_task(task_id)
        self._task = task
        self.instruction = bm.get_task_demonstration(task_id)

        task_bddl = task.problem_folder
        from libero.libero.envs import OffScreenRenderEnv

        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": self.resolution,
            "camera_widths": self.resolution,
        }
        self._env = OffScreenRenderEnv(**env_args)
        self._env.seed(0)

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        raw_obs = self._env.reset()
        return self._convert_obs(raw_obs)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        assert action.shape == (7,), f"Expected 7-DoF action, got {action.shape}"
        raw_obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        obs = self._convert_obs(raw_obs)
        return obs, reward, done, info

    def get_eef_pos(self) -> np.ndarray:
        return np.array(self._env.sim.data.site_xpos[
            self._env.sim.model.site_name2id("grip_site")
        ], dtype=np.float32)

    def _convert_obs(self, raw_obs) -> Dict[str, Any]:
        """Convert LIBERO obs to standardized format."""
        # Image
        image = raw_obs.get("agentview_image",
                            raw_obs.get("image", np.zeros((self.resolution,
                                                           self.resolution, 3),
                                                          dtype=np.uint8)))
        if image.shape[0] != self.resolution:
            image = image[::-1]  # LIBERO returns flipped images

        # Proprio
        eef_pos = raw_obs.get("robot0_eef_pos", np.zeros(3))
        eef_quat = raw_obs.get("robot0_eef_quat", np.zeros(4))
        gripper = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
        gripper_scalar = float(gripper.mean()) if gripper.size else 0.0

        proprio = np.concatenate([
            np.asarray(eef_pos, dtype=np.float32).reshape(3),
            np.asarray(eef_quat, dtype=np.float32).reshape(4),
            np.array([gripper_scalar], dtype=np.float32),
        ])

        return self._standardize_obs(
            image=image,
            proprio=proprio,
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            gripper=np.array([gripper_scalar]),
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ──────────────────────────────────────────────────────────────────────────
# SimplerEnv Wrappers (Google Robot + WidowX)
# ──────────────────────────────────────────────────────────────────────────

class SimplerEnvWrapper(UniversalEnvWrapper):
    """Wraps a SimplerEnv environment (Google Robot or WidowX).

    Google Robot: 7-DoF arm + 2-DoF base
      Action dim = 8 (7 arm + 1 base movement) or 7 arm-only
      We only apply our XYZ correction to the ARM component.

    WidowX: 6-DoF arm + gripper
      Action dim = 7 (6 joints + gripper) in task-space
    """

    # Map of supported environments
    ENV_CONFIGS = {
        # Google Robot environments
        "google_robot_pick_coke_can": {
            "env_name": "google_robot_pick_coke_can",
            "robot": "google_robot",
            "action_dim": 7,  # dx, dy, dz, droll, dpitch, dyaw, grip
            "has_base": True,
            "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
            "base_action_indices": [],
        },
        "google_robot_move_near": {
            "env_name": "google_robot_move_near",
            "robot": "google_robot",
            "action_dim": 7,
            "has_base": False,
            "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
            "base_action_indices": [],
        },
        "google_robot_open_top_drawer": {
            "env_name": "google_robot_open_top_drawer",
            "robot": "google_robot",
            "action_dim": 7,
            "has_base": True,
            "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
            "base_action_indices": [],
        },
        # WidowX environments
        "widowx_spoon_on_towel": {
            "env_name": "widowx_spoon_on_towel",
            "robot": "widowx",
            "action_dim": 7,
            "has_base": False,
            "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
            "base_action_indices": [],
        },
        "widowx_carrot_on_plate": {
            "env_name": "widowx_carrot_on_plate",
            "robot": "widowx",
            "action_dim": 7,
            "has_base": False,
            "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
            "base_action_indices": [],
        },
        "widowx_stack_cube": {
            "env_name": "widowx_stack_cube",
            "robot": "widowx",
            "action_dim": 7,
            "has_base": False,
            "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
            "base_action_indices": [],
        },
    }

    def __init__(self, env_name: str, task_id: int = 0,
                 instruction: str = "", resolution: int = 256):
        super().__init__(f"simpler/{env_name}", task_id=task_id,
                         instruction=instruction)
        self.robot_env_name = env_name
        self.resolution = resolution
        self._env = None
        self._raw_obs = None

        # Get config
        if env_name in self.ENV_CONFIGS:
            self._cfg = self.ENV_CONFIGS[env_name]
        else:
            # Default config for unknown SimplerEnv tasks
            self._cfg = {
                "env_name": env_name,
                "robot": "unknown",
                "action_dim": 7,
                "has_base": False,
                "arm_action_indices": [0, 1, 2, 3, 4, 5, 6],
                "base_action_indices": [],
            }

        self._init_env()

    def _init_env(self):
        """Initialize the SimplerEnv environment."""
        try:
            import simpler_env
            self._env = simpler_env.make(self.robot_env_name)
        except ImportError:
            raise ImportError(
                "simpler_env not installed. Install with:\n"
                "  pip install simpler-env\n"
                "or: git clone https://github.com/simpler-env/SimplerEnv"
            )

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._raw_obs, info = self._env.reset()
        return self._convert_obs(self._raw_obs)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        assert action.shape == (7,), f"Expected 7-DoF action, got {action.shape}"

        # Map standard 7-DoF action to the environment's native action space
        native_action = self._to_native_action(action)

        self._raw_obs, reward, terminated, truncated, info = self._env.step(
            native_action)
        self._step_count += 1
        done = terminated or truncated
        obs = self._convert_obs(self._raw_obs)

        return obs, reward, done, info

    def _to_native_action(self, action: np.ndarray) -> np.ndarray:
        """Convert standard 7-DoF action to the environment's native format.

        For Google Robot with base:
            Standard (7,): [dx, dy, dz, droll, dpitch, dyaw, grip]
            Native (8,):   [dx, dy, dz, droll, dpitch, dyaw, grip, base_movement]
            → We set base_movement = 0 (safety: don't move the base)

        For WidowX:
            Standard (7,): [dx, dy, dz, droll, dpitch, dyaw, grip]
            Native (7,):   same mapping
        """
        if self._cfg.get("has_base"):
            # Append zero base movement
            native = np.zeros(8, dtype=np.float32)
            native[:7] = action
            return native
        return action.copy()

    def apply_steering_correction(self, action: np.ndarray,
                                  correction_xyz: np.ndarray,
                                  alpha: float = 1.0) -> np.ndarray:
        """Apply an XYZ steering correction ONLY to the arm components.

        For Google Robot: corrects arm XYZ, leaves base untouched.
        For WidowX: corrects XYZ directly.

        Args:
            action:          (7,) standard action
            correction_xyz:  (3,) Cartesian correction [dx, dy, dz]
            alpha:           steering gain

        Returns:
            (7,) corrected action
        """
        corrected = action.copy()
        # XYZ are always the first 3 components in task-space
        corrected[0] += alpha * correction_xyz[0]
        corrected[1] += alpha * correction_xyz[1]
        corrected[2] += alpha * correction_xyz[2]
        return corrected

    def get_eef_pos(self) -> np.ndarray:
        """Return current EEF Cartesian position (3,)."""
        try:
            # SimplerEnv typically stores EEF position in the physics engine
            if hasattr(self._env, 'agent'):
                tcp_pose = self._env.agent.tcp.pose
                return np.array(tcp_pose.p, dtype=np.float32)
            elif hasattr(self._env, 'unwrapped'):
                env = self._env.unwrapped
                if hasattr(env, 'agent'):
                    tcp_pose = env.agent.tcp.pose
                    return np.array(tcp_pose.p, dtype=np.float32)
        except Exception:
            pass

        # Fallback: try from observation
        if self._raw_obs is not None:
            if isinstance(self._raw_obs, dict):
                for key in ["tcp_pose", "eef_pos", "end_effector_pos"]:
                    if key in self._raw_obs:
                        pos = np.asarray(self._raw_obs[key], dtype=np.float32)
                        return pos[:3]
        return np.zeros(3, dtype=np.float32)

    def _convert_obs(self, raw_obs) -> Dict[str, Any]:
        """Convert SimplerEnv obs to standardized format."""
        # SimplerEnv can return either a dict or an image array
        if isinstance(raw_obs, dict):
            image = raw_obs.get("image",
                                raw_obs.get("rgb",
                                            raw_obs.get("pixels",
                                                        np.zeros((self.resolution,
                                                                  self.resolution, 3),
                                                                 dtype=np.uint8))))
            eef_pos = self.get_eef_pos()
            proprio = raw_obs.get("state", raw_obs.get("proprio", None))
        elif isinstance(raw_obs, np.ndarray) and raw_obs.ndim == 3:
            image = raw_obs
            eef_pos = self.get_eef_pos()
            proprio = None
        else:
            image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            eef_pos = np.zeros(3, dtype=np.float32)
            proprio = None

        return self._standardize_obs(
            image=image,
            eef_pos=eef_pos,
            proprio=proprio,
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

def make_env(platform: str, env_name: str = "",
             task_id: int = 0, instruction: str = "",
             **kwargs) -> UniversalEnvWrapper:
    """Create a universal environment wrapper.

    Args:
        platform:    "libero", "google_robot", "widowx", or "simpler"
        env_name:    specific environment name (e.g., "libero_spatial",
                     "google_robot_pick_coke_can")
        task_id:     task index within the environment
        instruction: language instruction for the task

    Returns:
        UniversalEnvWrapper instance

    Examples:
        >>> env = make_env("libero", "libero_spatial", task_id=0)
        >>> env = make_env("google_robot", "google_robot_pick_coke_can")
        >>> env = make_env("widowx", "widowx_spoon_on_towel")
    """
    platform = platform.lower()

    if platform == "libero":
        task_suite = env_name if env_name else "libero_spatial"
        return LIBEROWrapper(task_suite=task_suite, task_id=task_id,
                             **kwargs)
    elif platform in ("google_robot", "widowx", "simpler"):
        if not env_name:
            if platform == "google_robot":
                env_name = "google_robot_pick_coke_can"
            elif platform == "widowx":
                env_name = "widowx_spoon_on_towel"
            else:
                raise ValueError("Must specify env_name for SimplerEnv")
        return SimplerEnvWrapper(env_name=env_name, task_id=task_id,
                                 instruction=instruction, **kwargs)
    else:
        raise ValueError(
            f"Unknown platform: {platform}. "
            f"Supported: libero, google_robot, widowx, simpler"
        )


# ──────────────────────────────────────────────────────────────────────────
# Steering Integrator
# ──────────────────────────────────────────────────────────────────────────

class SteeringIntegrator:
    """Integrates the SafetyMLP with any UniversalEnvWrapper.

    Usage:
        env = make_env("google_robot", "google_robot_pick_coke_can")
        policy = PolicyAdapter.from_openvla("model_name")
        mlp = SafetyMLP.load("checkpoint.pt")
        integrator = SteeringIntegrator(env, policy, mlp, alpha=1.0)

        obs = env.reset()
        for step in range(max_steps):
            action, embedding = policy.predict(obs, return_embedding=True)
            safe_action = integrator.steer(action, embedding)
            obs, reward, done, info = env.step(safe_action)
    """

    def __init__(self, env: UniversalEnvWrapper,
                 safety_mlp: nn.Module,
                 scaler=None,
                 alpha: float = 1.0,
                 fail_threshold: float = 0.5):
        self.env = env
        self.safety_mlp = safety_mlp
        self.scaler = scaler
        self.alpha = alpha
        self.fail_threshold = fail_threshold
        self.safety_mlp.eval()

    def predict_correction(self, embedding: np.ndarray):
        """Predict a Cartesian correction from a VLA embedding.

        Returns:
            correction_xyz: (3,) correction in meters
            fail_prob:       float, probability of failure
        """
        import torch

        if self.scaler is not None:
            embedding = self.scaler.transform(embedding.reshape(1, -1))
        else:
            embedding = embedding.reshape(1, -1)

        x = torch.FloatTensor(embedding)
        with torch.no_grad():
            out = self.safety_mlp(x)

        correction = out["correction"].cpu().numpy().squeeze()
        fail_prob = float(torch.sigmoid(out["will_fail"]).cpu().item())

        return correction, fail_prob

    def steer(self, action: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """Apply safety steering to a proposed action.

        Only applies correction when the MLP predicts high failure probability.
        """
        correction, fail_prob = self.predict_correction(embedding)

        if fail_prob > self.fail_threshold:
            # Apply correction only to XYZ components
            steered = action.copy()
            steered[:3] += self.alpha * correction
            return steered
        else:
            return action
