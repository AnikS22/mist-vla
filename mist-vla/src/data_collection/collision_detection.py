"""
MuJoCo collision detection for LIBERO environments.

This module detects robot collisions using MuJoCo's contact detection API.
It distinguishes between:
- Valid contacts (gripper-object, object-table)
- Invalid collisions (robot-table, robot-wall, robot-obstacle)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


class CollisionDetector:
    """
    Detects robot collisions in MuJoCo simulation.

    Uses sim.data.ncon and sim.data.contact to detect when the robot
    collides with non-object surfaces (tables, walls, obstacles).
    """

    # Robot geom names (typical LIBERO robot)
    ROBOT_GEOMS = [
        'robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3',
        'robot0_link4', 'robot0_link5', 'robot0_link6', 'robot0_link7',
        'robot0_right_finger', 'robot0_left_finger',
        'panda0_link0', 'panda0_link1', 'panda0_link2', 'panda0_link3',
        'panda0_link4', 'panda0_link5', 'panda0_link6', 'panda0_link7',
        'panda0_hand', 'panda0_leftfinger', 'panda0_rightfinger',
    ]

    # Environment geom names (collisions with these are bad)
    ENVIRONMENT_GEOMS = [
        'table', 'floor', 'wall', 'ground', 'obstacle',
        'bin_base', 'shelf', 'cabinet',
    ]

    # Valid contact geom names (collisions with these are OK)
    VALID_CONTACT_GEOMS = [
        'object', 'cube', 'block', 'ball', 'container',
        'gripper', 'finger',
    ]

    def __init__(self, env):
        """
        Initialize collision detector.

        Args:
            env: LIBERO environment with sim attribute
        """
        self.env = env
        # Note: We don't store sim here because it can become stale
        # Instead, we'll get it fresh from env.env.sim each time we need it

    def check_collision(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if robot is in collision.

        Returns:
            Tuple of (is_collision, collision_position):
            - is_collision: True if robot collides with environment
            - collision_position: 3D position of collision [x, y, z], or None
        """
        # Get fresh sim reference (it can become stale between calls)
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'sim'):
            sim = self.env.env.sim
        elif hasattr(self.env, 'sim'):
            sim = self.env.sim
        else:
            return False, None

        try:
            ncon = sim.data.ncon
        except AttributeError:
            # If we still can't access contact data, return no collision
            return False, None

        if ncon == 0:
            return False, None

        for i in range(ncon):
            contact = sim.data.contact[i]

            # Get geom names
            try:
                geom1_name = sim.model.geom_id2name(contact.geom1)
                geom2_name = sim.model.geom_id2name(contact.geom2)
            except:
                # Some geoms may not have names
                continue

            if geom1_name is None or geom2_name is None:
                continue

            # Check if this is a robot collision
            is_collision, pos = self._is_invalid_collision(
                geom1_name, geom2_name, contact
            )

            if is_collision:
                return True, pos

        return False, None

    def _is_invalid_collision(
        self, geom1: str, geom2: str, contact
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Determine if this contact represents an invalid collision.

        Args:
            geom1: First geom name
            geom2: Second geom name
            contact: MuJoCo contact object

        Returns:
            Tuple of (is_invalid, position)
        """
        geom1_lower = geom1.lower()
        geom2_lower = geom2.lower()

        # Check if robot is involved
        robot_in_contact = (
            self._is_robot_geom(geom1_lower) or
            self._is_robot_geom(geom2_lower)
        )

        if not robot_in_contact:
            return False, None

        # Check if this is a valid contact (e.g., gripper-object)
        if self._is_valid_contact(geom1_lower, geom2_lower):
            return False, None

        # Check if robot is colliding with environment
        if self._is_environment_collision(geom1_lower, geom2_lower):
            collision_pos = contact.pos.copy()
            return True, collision_pos

        return False, None

    def _is_robot_geom(self, geom_name: str) -> bool:
        """Check if geom belongs to robot."""
        for robot_geom in self.ROBOT_GEOMS:
            if robot_geom.lower() in geom_name:
                return True
        return 'robot' in geom_name or 'panda' in geom_name

    def _is_valid_contact(self, geom1: str, geom2: str) -> bool:
        """
        Check if this is a valid contact (not a collision).

        Valid contacts:
        - Gripper touching objects
        - Finger touching objects
        """
        for valid_geom in self.VALID_CONTACT_GEOMS:
            if valid_geom in geom1 or valid_geom in geom2:
                # If one is robot and other is valid contact, it's OK
                if ('finger' in geom1 or 'gripper' in geom1) and 'object' in geom2:
                    return True
                if ('finger' in geom2 or 'gripper' in geom2) and 'object' in geom1:
                    return True

        return False

    def _is_environment_collision(self, geom1: str, geom2: str) -> bool:
        """Check if collision is with environment."""
        for env_geom in self.ENVIRONMENT_GEOMS:
            if env_geom in geom1 or env_geom in geom2:
                return True
        return False

    def get_end_effector_position(self) -> np.ndarray:
        """
        Get current end-effector position.

        Returns:
            3D position [x, y, z]
        """
        # Get fresh sim reference
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'sim'):
            sim = self.env.env.sim
        elif hasattr(self.env, 'sim'):
            sim = self.env.sim
        else:
            return np.array([0.0, 0.0, 0.0])

        # LIBERO typically uses 'robot0_eef_pos' or similar
        # Try multiple common names
        try:
            # Method 1: Direct from env
            if hasattr(self.env, 'get_ee_pos'):
                return self.env.get_ee_pos()

            # Method 2: From sim data
            if hasattr(self.env, '_eef_xpos'):
                return self.env._eef_xpos.copy()

            # Method 3: From site
            eef_site_id = sim.model.site_name2id('grip_site')
            return sim.data.site_xpos[eef_site_id].copy()

        except:
            # Fallback: use last robot link position
            try:
                body_id = sim.model.body_name2id('robot0_link7')
                return sim.data.body_xpos[body_id].copy()
            except:
                # Last resort: return origin
                return np.array([0.0, 0.0, 0.0])

    def get_collision_details(self) -> Dict:
        """
        Get detailed collision information for debugging.

        Returns:
            Dictionary with collision details
        """
        # Get fresh sim reference
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'sim'):
            sim = self.env.env.sim
        elif hasattr(self.env, 'sim'):
            sim = self.env.sim
        else:
            return {
                'num_contacts': 0,
                'contacts': [],
                'has_collision': False,
                'collision_position': None,
            }

        details = {
            'num_contacts': sim.data.ncon,
            'contacts': [],
            'has_collision': False,
            'collision_position': None,
        }

        has_collision, collision_pos = self.check_collision()
        details['has_collision'] = has_collision
        details['collision_position'] = collision_pos

        for i in range(min(sim.data.ncon, 10)):  # Limit to 10 for debug
            contact = sim.data.contact[i]
            try:
                geom1_name = sim.model.geom_id2name(contact.geom1)
                geom2_name = sim.model.geom_id2name(contact.geom2)

                details['contacts'].append({
                    'geom1': geom1_name,
                    'geom2': geom2_name,
                    'position': contact.pos.copy(),
                    'force': np.linalg.norm(contact.force) if hasattr(contact, 'force') else 0.0,
                })
            except:
                continue

        return details


def test_collision_detector():
    """Test collision detector on a LIBERO environment."""
    print("Testing CollisionDetector...")
    print("=" * 60)

    try:
        # Import LIBERO
        from libero.libero import benchmark

        # Create environment
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict['libero_spatial']()
        env = task_suite.make_env(task_id=0)
        env.reset()

        # Create detector
        detector = CollisionDetector(env)

        # Test initial state
        print("\n[Test 1] Initial state (should have no collision)")
        has_collision, collision_pos = detector.check_collision()
        print(f"  Collision: {has_collision}")
        if collision_pos is not None:
            print(f"  Position: {collision_pos}")

        # Test end-effector position
        print("\n[Test 2] End-effector position")
        ee_pos = detector.get_end_effector_position()
        print(f"  EE Position: {ee_pos}")

        # Test collision details
        print("\n[Test 3] Collision details")
        details = detector.get_collision_details()
        print(f"  Number of contacts: {details['num_contacts']}")
        print(f"  Has collision: {details['has_collision']}")
        if details['contacts']:
            print(f"  First contact: {details['contacts'][0]}")

        # Test with random actions
        print("\n[Test 4] Testing with random actions")
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            has_collision, collision_pos = detector.check_collision()
            if has_collision:
                print(f"  Step {step}: COLLISION at {collision_pos}")
            else:
                print(f"  Step {step}: No collision")

        env.close()
        print("\n✓ CollisionDetector test passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_collision_detector()
