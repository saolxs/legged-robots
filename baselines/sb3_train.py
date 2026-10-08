"""Train Stable-Baselines3 agents (PPO, SAC, TD3) on the same environments as ppo/train.py,
so on-policy and off-policy algorithms can be compared on equal footing.

    python baselines/sb3_train.py --algo sac --env-id Ant-v4 --seed 1 --total-timesteps 1000000
    python baselines/sb3_train.py --algo td3 --env-id MinitaurBulletEnv-v0 --seed 1

Runs are written to runs/<env>__sb3-<algo>__<seed>__<time>/ with the same TensorBoard
tags as ppo/train.py (charts/episodic_return), so ppo/plot.py groups them automatically:

    python ppo/plot.py runs --env-id Ant-v4     # curves for tanh64 (our PPO), sb3-sac, sb3-td3, ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ppo"))
try:
    import pybullet_envs_gymnasium  # noqa: F401
    import minitaur_env  # noqa: F401
except ImportError:
    pass

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}

# Hyperparameters follow the SB3 / rl-baselines3-zoo defaults for MuJoCo-style control.
ALGO_KWARGS = {
    "ppo": dict(n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh)),
    "sac": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256, tau=0.005,
                gamma=0.99, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
    "td3": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256, tau=0.005,
                gamma=0.99, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[256, 256])),
}


class EpisodeLogger(BaseCallback):
    """Write charts/episodic_return exactly like ppo/train.py so plot.py can read both."""

    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self.writer = writer
        self.returns: list[float] = []
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.returns.append(float(ep["r"]))
                self.writer.add_scalar("charts/episodic_return", ep["r"], self.num_timesteps)
                self.writer.add_scalar("charts/episodic_length", ep["l"], self.num_timesteps)
                if len(self.returns) % 10 == 0:
                    print(f"step={self.num_timesteps:>9d}  return={ep['r']:9.2f}  len={int(ep['l']):5d}  "
                          f"avg10={np.mean(self.returns[-10:]):9.2f}  sps={int(self.num_timesteps / (time.time() - self.t0))}")
        return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, default="sac")
    p.add_argument("--env-id", default="Ant-v4")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--exp-name", default=None, help="defaults to sb3-<algo>")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--env-kwargs", type=json.loads, default=None)
    a = p.parse_args()

    exp = a.exp_name or f"sb3-{a.algo}"
    run_dir = Path(a.out_dir) / f"{a.env_id}__{exp}__{a.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a) | {"algo_kwargs": str(ALGO_KWARGS[a.algo])}, indent=2))
    writer = SummaryWriter(str(run_dir))

    def make():
        env = gym.make(a.env_id, **(a.env_kwargs or {}))
        env = Monitor(env)  # SB3's episode-statistics wrapper
        # No ClipAction here: it widens the action box to (-inf, inf) and SB3 needs finite
        # bounds. SB3 clips sampled actions to the env's own bounds itself.
        return env

    venv = DummyVecEnv([make])
    venv.seed(a.seed)
    if a.algo == "ppo":  # on-policy: same obs/reward normalisation as ppo/train.py
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    kwargs = dict(ALGO_KWARGS[a.algo])
    if a.algo == "td3":
        n_act = venv.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1 * np.ones(n_act))
    device = "cuda" if a.cuda and torch.cuda.is_available() else "cpu"
    model = ALGOS[a.algo]("MlpPolicy", venv, seed=a.seed, device=device, verbose=0, **kwargs)
    print(f"algo={a.algo} env={a.env_id} seed={a.seed} device={device} params={sum(p.numel() for p in model.policy.parameters())}")

    cb = EpisodeLogger(writer)
    t0 = time.time()
    model.learn(total_timesteps=a.total_timesteps, callback=cb)
    model.save(str(run_dir / "model.zip"))
    if a.algo == "ppo":
        venv.save(str(run_dir / "vecnormalize.pkl"))
    summary = {"env_id": a.env_id, "algo": a.algo, "seed": a.seed, "global_step": model.num_timesteps,
               "final_avg_return_last20": float(np.mean(cb.returns[-20:])) if cb.returns else None,
               "episodes": len(cb.returns), "wall_time_s": time.time() - t0}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", json.dumps(summary))
    writer.close()
    venv.close()


if __name__ == "__main__":
    main()
