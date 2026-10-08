"""Gymnasium-API wrapper for PyBullet's bundled Minitaur quadruped (MinitaurBulletEnv-v0).

`pybullet_envs_gymnasium` ports the Roboschool-style locomotors (Ant, HalfCheetah, ...)
but not the Minitaur, whose code still lives in `pybullet_envs.bullet.minitaur_gym_env`
and targets the pre-2021 `gym` API. This module loads that file without importing the
old `gym` package (a thin alias to gymnasium is injected) and exposes the environment
with the modern `reset(seed=...) -> (obs, info)` / 5-tuple `step` contract.

Importing this module registers `MinitaurBulletEnv-v0` (1000-step time limit, as in
the original PyBullet registration).
"""
from __future__ import annotations

import importlib
import os
import random
import sys
import types

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- shims so the legacy module imports cleanly -------------------------------------
if "gym" not in sys.modules:
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = gym.spaces
    sys.modules["gym.utils"] = gym.utils
    sys.modules["gym.utils.seeding"] = gym.utils.seeding
if "pkg_resources" not in sys.modules:
    try:
        from packaging.version import Version as _parse_version
    except ImportError:  # pragma: no cover
        def _parse_version(v):
            return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())
    _pr = types.ModuleType("pkg_resources")
    _pr.parse_version = _parse_version
    sys.modules["pkg_resources"] = _pr

# Load `pybullet_envs.bullet.minitaur_gym_env` without running `pybullet_envs/__init__.py`
# (which imports the old gym package and registers old-API environments).
if "pybullet_envs" not in sys.modules:
    import pybullet_data  # noqa: F401  (ensures pybullet is importable)
    _root = os.path.join(os.path.dirname(os.path.dirname(pybullet_data.__file__)), "pybullet_envs")
    _pkg = types.ModuleType("pybullet_envs")
    _pkg.__path__ = [_root]
    sys.modules["pybullet_envs"] = _pkg
_legacy = importlib.import_module("pybullet_envs.bullet.minitaur_gym_env")


class MinitaurEnv(gym.Env):
    """Minitaur quadruped: 28-d proprioceptive observation (motor angles, velocities,
    torques, base orientation quaternion), 8-d action in [-1, 1] (leg-space pose
    targets), reward = forward distance - energy penalty. Terminates on falling."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, **kwargs):
        self.render_mode = render_mode
        self._env = _legacy.MinitaurBulletEnv(render=False, **kwargs)
        self.observation_space = spaces.Box(
            self._env.observation_space.low.astype(np.float32),
            self._env.observation_space.high.astype(np.float32), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
            random.seed(seed)  # the env randomizer uses the stdlib RNG
        obs = self._env.reset()
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, done, info = self._env.step(action)
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, dict(info)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.render("rgb_array")
        return None

    def close(self):
        try:
            self._env._pybullet_client.disconnect()
        except Exception:
            pass


if "MinitaurBulletEnv-v0" not in gym.registry:
    gym.register(id="MinitaurBulletEnv-v0", entry_point="minitaur_env:MinitaurEnv",
                 max_episode_steps=1000, reward_threshold=5.0)
