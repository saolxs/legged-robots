"""Train a locomotion policy in MuJoCo Playground (MJX, GPU) with Brax PPO and render a rollout.

    .venv/bin/python train_playground.py --env Go1JoystickFlatTerrain --timesteps 20000000
    .venv/bin/python train_playground.py --env G1JoystickFlatTerrain          # Unitree G1 humanoid
    .venv/bin/python train_playground.py --list

Writes <out>/<env>__<time>/{params.msgpack, metrics.json, progress.png, rollout.mp4}.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")  # headless rendering; EGL fails on this box
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import numpy as np

# Ampere (RTX 30xx) defaults to TF32 matmuls, which destabilises MJX training; force full precision.
jax.config.update("jax_default_matmul_precision", "highest")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Go1JoystickFlatTerrain")
    p.add_argument("--timesteps", type=int, default=None, help="override the preset num_timesteps")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="runs")
    p.add_argument("--list", action="store_true")
    p.add_argument("--command", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="x-vel y-vel yaw-rate for the rollout")
    p.add_argument("--rollout-steps", type=int, default=500)
    p.add_argument("--load", type=Path, default=None, help="skip training; roll out + render these saved params")
    a = p.parse_args()

    from mujoco_playground import registry, wrapper
    if a.list:
        print("locomotion:", *registry.locomotion.ALL_ENVS, sep="\n  ")
        return
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground.config import locomotion_params

    run_dir = a.load.parent if a.load else Path(a.out) / f"{a.env}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = registry.get_default_config(a.env)
    env = registry.load(a.env, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(a.env)
    if a.timesteps:
        ppo_params.num_timesteps = a.timesteps
    training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in training_params:
        del training_params["network_factory"]
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)
    from brax.io import model
    from brax.training.acme import running_statistics
    if a.load:
        nets = network_factory(env.observation_size, env.action_size,
                               preprocess_observations_fn=running_statistics.normalize)
        make_inference_fn = ppo_networks.make_inference_fn(nets)
        params = model.load_params(str(a.load))
        print(f"loaded {a.load}")
    else:
        print(f"env={a.env} device={jax.devices()[0]} timesteps={ppo_params.num_timesteps} num_envs={ppo_params.num_envs}")
        (run_dir / "config.json").write_text(json.dumps({"env": a.env, "seed": a.seed, "ppo": {k: str(v) for k, v in ppo_params.items()},
                                                         "env_cfg": str(env_cfg)}, indent=2))

        history: list[dict] = []
        t0 = time.time()

        def progress(num_steps, metrics):
            row = {"step": int(num_steps), "wall_s": time.time() - t0,
                   "eval/episode_reward": float(metrics.get("eval/episode_reward", np.nan))}
            history.append(row)
            print(f"step={row['step']:>11d}  reward={row['eval/episode_reward']:9.3f}  t={row['wall_s']:6.0f}s", flush=True)

        train_fn = functools.partial(ppo.train, **training_params, network_factory=network_factory,
                                     randomization_fn=registry.get_domain_randomizer(a.env),
                                     progress_fn=progress, seed=a.seed)
        make_inference_fn, params, _ = train_fn(environment=env, eval_env=registry.load(a.env, config=env_cfg),
                                                wrap_env_fn=wrapper.wrap_for_brax_training)
        print(f"training finished in {time.time() - t0:.0f}s")
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        model.save_params(str(run_dir / "params.msgpack"), params)

        # ---- plot --------------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot([h["step"] for h in history], [h["eval/episode_reward"] for h in history])
        ax.set_xlabel("environment steps"); ax.set_ylabel("eval episode reward"); ax.set_title(a.env)
        fig.tight_layout(); fig.savefig(run_dir / "progress.png", dpi=150)

    # ---- rollout + video ---------------------------------------------------------
    eval_env = registry.load(a.env, config=env_cfg)
    jit_reset, jit_step = jax.jit(eval_env.reset), jax.jit(eval_env.step)
    jit_policy = jax.jit(make_inference_fn(params, deterministic=True))
    rng = jax.random.PRNGKey(a.seed)
    state = jit_reset(rng)
    cmd = jp.array(a.command)
    rollout, rewards = [], []
    for _ in range(a.rollout_steps):
        if "command" in state.info:
            state.info["command"] = cmd
        rng, act_rng = jax.random.split(rng)
        ctrl, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state); rewards.append(float(state.reward))
    print(f"rollout: {len(rollout)} steps, mean reward/step={np.mean(rewards):.3f}, final done={bool(state.done)}")
    try:
        import mediapy as media
        try:
            import imageio_ffmpeg
            media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            pass
        fps = 1.0 / eval_env.dt
        every = max(1, int(round(fps / 30)))
        frames = eval_env.render(rollout[::every], camera="track", width=640, height=480)
        media.write_video(str(run_dir / "rollout.mp4"), frames, fps=fps / every)
        print("video:", run_dir / "rollout.mp4")
    except Exception as e:  # rendering is optional
        print("video skipped:", repr(e))
    print("outputs in", run_dir)


if __name__ == "__main__":
    main()
