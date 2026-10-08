"""Evaluate a trained PPO checkpoint with a deterministic (mean-action) policy.

    python ppo/evaluate.py runs/Ant-v4__ppo__1__.../checkpoint.pt --episodes 10 --video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agent import Agent

try:  # registers HalfCheetahBulletEnv-v0, AntBulletEnv-v0, ... on the Gymnasium API
    import pybullet_envs_gymnasium  # noqa: F401
    import minitaur_env  # noqa: F401  registers MinitaurBulletEnv-v0 (see ppo/minitaur_env.py)
except ImportError:
    pass


def make_env(env_id: str, render_mode: str | None, env_kwargs: dict):
    """gym.make that follows the robot with the model's 'track' camera when rendering (MuJoCo envs)."""
    if render_mode is not None:
        try:
            return gym.make(env_id, render_mode=render_mode, camera_name="track", **env_kwargs)
        except Exception:
            pass
    return gym.make(env_id, render_mode=render_mode, **env_kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--video", action="store_true", help="record episodes to <run_dir>/eval_videos")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    env = make_env(cfg["env_id"], "rgb_array" if a.video else None, cfg.get("env_kwargs") or {})
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if a.video:
        env = gym.wrappers.RecordVideo(env, str(a.checkpoint.parent / "eval_videos"), episode_trigger=lambda e: True)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    obs_rms = ckpt.get("obs_rms")

    agent = Agent(int(np.prod(env.observation_space.shape)), env.action_space, tuple(cfg["hidden"]), cfg["activation"])
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    returns, lengths = [], []
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        done = False
        while not done:
            if obs_rms is not None:  # apply the frozen training-time normalisation
                obs = np.clip((obs - obs_rms["mean"]) / np.sqrt(obs_rms["var"] + 1e-8), -cfg["obs_clip"], cfg["obs_clip"])
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = agent.action_value(x)[0] if a.stochastic else agent.act_deterministic(x)
            obs, _, term, trunc, info = env.step(action.squeeze(0).numpy())
            done = term or trunc
        returns.append(float(info["episode"]["r"]))
        lengths.append(int(info["episode"]["l"]))
        print(f"episode {ep}: return={returns[-1]:.2f} length={lengths[-1]}")
    env.close()
    out = {"env_id": cfg["env_id"], "episodes": a.episodes, "mean_return": float(np.mean(returns)),
           "std_return": float(np.std(returns)), "mean_length": float(np.mean(lengths))}
    print(json.dumps(out, indent=2))
    (a.checkpoint.parent / "eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
