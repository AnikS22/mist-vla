#!/usr/bin/env python3
"""Test MuJoCo contact API"""
import sys
sys.path.insert(0, '/mnt/beegfs/home/asahai2024/LIBERO')

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import os

print("=== Loading LIBERO environment ===")
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict['libero_spatial']()
task = task_suite.get_task(0)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file
)
env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=128,
    camera_widths=128,
)

init_states = task_suite.get_task_init_states(0)
env.reset()
env.set_init_state(init_states[0])

print("\n=== Environment Structure ===")
print(f"env type: {type(env)}")
print(f"env.env type: {type(env.env)}")

# Get sim
if hasattr(env, 'env') and hasattr(env.env, 'sim'):
    sim = env.env.sim
    print(f"\nsim type: {type(sim)}")
    print(f"sim module: {type(sim).__module__}")

    # List all attributes
    attrs = [a for a in dir(sim) if not a.startswith('_')]
    print(f"\nAll sim attributes ({len(attrs)} total):")
    for attr in attrs:
        print(f"  - {attr}")

    # Check for data-like attributes
    print(f"\n=== Checking data access ===")
    if hasattr(sim, 'data'):
        print(f"sim.data exists: {type(sim.data)}")
        if hasattr(sim.data, 'ncon'):
            print(f"  sim.data.ncon = {sim.data.ncon}")
    else:
        print("sim.data does NOT exist")

    # Check for MuJoCo 3.x style access
    if hasattr(sim, 'model'):
        print(f"\nsim.model exists: {type(sim.model)}")

    # Try accessing contact data directly
    print(f"\n=== Looking for contact info ===")
    contact_attrs = ['contact', 'contacts', 'ncon', 'get_contacts']
    for attr in contact_attrs:
        if hasattr(sim, attr):
            val = getattr(sim, attr)
            print(f"sim.{attr}: {type(val)}")
            if callable(val):
                try:
                    result = val()
                    print(f"  -> {attr}() returns: {type(result)}")
                    if hasattr(result, '__len__'):
                        print(f"  -> length: {len(result)}")
                except Exception as e:
                    print(f"  -> error calling: {e}")

    # Check if it's a wrapped MuJoCo object
    if hasattr(sim, '_model'):
        print(f"\nsim._model exists: {type(sim._model)}")
    if hasattr(sim, '_data'):
        print(f"sim._data exists: {type(sim._data)}")
        if hasattr(sim._data, 'ncon'):
            print(f"  sim._data.ncon = {sim._data.ncon}")

    # Try to step and create collisions
    print(f"\n=== Testing with simulation steps ===")
    for i in range(5):
        obs, reward, done, info = env.step([0] * 7)  # Zero action

        # Try different ways to get contact count
        methods = []
        if hasattr(sim, 'data') and hasattr(sim.data, 'ncon'):
            methods.append(('sim.data.ncon', sim.data.ncon))
        if hasattr(sim, '_data') and hasattr(sim._data, 'ncon'):
            methods.append(('sim._data.ncon', sim._data.ncon))

        print(f"Step {i}: ", end="")
        for name, val in methods:
            print(f"{name}={val} ", end="")
        if not methods:
            print("No contact method found!")
        print()

env.close()
print("\n=== Test Complete ===")
