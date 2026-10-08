"""PPO for legged-robot locomotion (Ant, HalfCheetah, Minitaur) and CartPole.

Single-file trainer that replaces the four near-identical scripts in
ant_ppo/, halfcheetah_ppo/, minitaur_ppo/ and cartpole_ppo/. It follows the
CleanRL ppo_continuous_action recipe (Huang et al., 2022) and runs on
Gymnasium >= 1.1 with MuJoCo 3 and PyBullet via pybullet_envs_gymnasium.

Examples
--------
    python ppo/train.py --env-id Ant-v4 --seed 1
    python ppo/train.py --env-id MinitaurBulletEnv-v0 --hidden 128 128 --activation relu
    python ppo/train.py --env-id CartPole-v1 --total-timesteps 100000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agent import Agent

try:  # registers HalfCheetahBulletEnv-v0, AntBulletEnv-v0, ... on the Gymnasium API
    import pybullet_envs_gymnasium  # noqa: F401
    import minitaur_env  # noqa: F401  registers MinitaurBulletEnv-v0 (see ppo/minitaur_env.py)
except ImportError:
    pass

# Per-environment defaults reproduce the settings used in the original report.
ENV_PRESETS: dict[str, dict] = {
    "Ant-v4": dict(total_timesteps=2_000_000, learning_rate=3e-4, num_minibatches=32, reward_clip=10.0),
    "Ant-v5": dict(total_timesteps=2_000_000, learning_rate=3e-4, num_minibatches=32, reward_clip=10.0),
    "HalfCheetah-v4": dict(total_timesteps=1_000_000, learning_rate=3e-4, num_minibatches=32, reward_clip=10.0),
    "HalfCheetahBulletEnv-v0": dict(total_timesteps=1_000_000, learning_rate=3e-4, num_minibatches=32, reward_clip=10.0),
    "MinitaurBulletEnv-v0": dict(total_timesteps=500_000, learning_rate=1e-4, num_minibatches=64, reward_clip=5.0),
    # additional legged robots (MuJoCo)
    "Hopper-v4": dict(total_timesteps=1_000_000),
    "Walker2d-v4": dict(total_timesteps=1_000_000),
    "Humanoid-v4": dict(total_timesteps=10_000_000, learning_rate=3e-4, hidden=(256, 256)),
    "Hopper-v5": dict(total_timesteps=1_000_000),
    "Walker2d-v5": dict(total_timesteps=1_000_000),
    "Humanoid-v5": dict(total_timesteps=10_000_000, learning_rate=3e-4, hidden=(256, 256)),
    # additional legged robots (PyBullet, via pybullet_envs_gymnasium)
    "AntBulletEnv-v0": dict(total_timesteps=2_000_000),
    "HopperBulletEnv-v0": dict(total_timesteps=1_000_000),
    "Walker2DBulletEnv-v0": dict(total_timesteps=1_000_000),
    "HumanoidBulletEnv-v0": dict(total_timesteps=10_000_000, hidden=(256, 256)),
    # additional legged robots (MuJoCo)
    "Hopper-v4": dict(total_timesteps=1_000_000),
    "Walker2d-v4": dict(total_timesteps=1_000_000),
    "Humanoid-v4": dict(total_timesteps=10_000_000, learning_rate=3e-4, hidden=(256, 256)),
    "Hopper-v5": dict(total_timesteps=1_000_000),
    "Walker2d-v5": dict(total_timesteps=1_000_000),
    "Humanoid-v5": dict(total_timesteps=10_000_000, learning_rate=3e-4, hidden=(256, 256)),
    # additional legged robots (PyBullet, via pybullet_envs_gymnasium)
    "AntBulletEnv-v0": dict(total_timesteps=2_000_000),
    "HopperBulletEnv-v0": dict(total_timesteps=1_000_000),
    "Walker2DBulletEnv-v0": dict(total_timesteps=1_000_000),
    "HumanoidBulletEnv-v0": dict(total_timesteps=10_000_000, hidden=(256, 256)),
    "CartPole-v1": dict(total_timesteps=500_000, learning_rate=2.5e-4, num_steps=128, num_envs=4,
                        num_minibatches=4, update_epochs=4, ent_coef=0.01, normalize=False),
}


@dataclass
class Args:
    env_id: str = "Ant-v4"
    exp_name: str = "ppo"
    seed: int = 1
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    normalize: bool = True            # observation + reward running normalisation
    obs_clip: float = 10.0
    reward_clip: float = 10.0
    cuda: bool = False                # small MLPs train faster on CPU
    torch_deterministic: bool = True
    capture_video: bool = False
    track: bool = False
    wandb_project_name: str = "legged-robots-ppo"
    wandb_entity: str | None = None
    out_dir: str = "runs"
    env_kwargs: dict | None = None    # e.g. '{"healthy_reward": 5, "healthy_z_range": [0.5, 2]}' to reproduce the 2022 Ant runs
    # derived
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Args()
    p.add_argument("--env-id", default=d.env_id)
    p.add_argument("--exp-name", default=d.exp_name)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--total-timesteps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-steps", type=int)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false", default=None)
    p.add_argument("--gamma", type=float)
    p.add_argument("--gae-lambda", type=float)
    p.add_argument("--num-minibatches", type=int)
    p.add_argument("--update-epochs", type=int)
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false", default=None)
    p.add_argument("--clip-coef", type=float)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false", default=None)
    p.add_argument("--ent-coef", type=float)
    p.add_argument("--vf-coef", type=float)
    p.add_argument("--max-grad-norm", type=float)
    p.add_argument("--target-kl", type=float)
    p.add_argument("--hidden", type=int, nargs="+", help="hidden layer sizes, e.g. --hidden 128 64 32")
    p.add_argument("--activation", choices=["tanh", "relu", "elu"])
    p.add_argument("--no-normalize", dest="normalize", action="store_false", default=None)
    p.add_argument("--obs-clip", type=float)
    p.add_argument("--reward-clip", type=float)
    p.add_argument("--cuda", action="store_true", default=None)
    p.add_argument("--capture-video", action="store_true", default=None)
    p.add_argument("--track", action="store_true", default=None)
    p.add_argument("--wandb-project-name")
    p.add_argument("--wandb-entity")
    p.add_argument("--out-dir")
    p.add_argument("--env-kwargs", type=json.loads, help="JSON dict forwarded to gym.make")
    ns = p.parse_args()

    cfg = asdict(d)
    cfg.update(ENV_PRESETS.get(ns.env_id, {}))                  # env preset overrides defaults
    cfg.update({k: v for k, v in vars(ns).items() if v is not None})  # CLI overrides preset
    cfg["hidden"] = tuple(cfg["hidden"])
    for k in ("batch_size", "minibatch_size", "num_iterations"):
        cfg.pop(k, None)
    args = Args(**cfg)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def make_env(env_id: str, idx: int, capture_video: bool, run_dir: Path, args: Args):
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        kw = dict(args.env_kwargs or {})
        if render_mode is not None:
            try:  # MuJoCo models ship a 'track' camera that follows the robot
                env = gym.make(env_id, render_mode=render_mode, camera_name="track", **kw)
            except Exception:
                env = gym.make(env_id, render_mode=render_mode, **kw)
        else:
            env = gym.make(env_id, **kw)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if render_mode:
            env = gym.wrappers.RecordVideo(env, str(run_dir / "videos"), episode_trigger=lambda e: e % 50 == 0)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if args.normalize:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda o: np.clip(o, -args.obs_clip, args.obs_clip), env.observation_space)
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -args.reward_clip, args.reward_clip)))
        return env
    return thunk


def save_checkpoint(path: Path, agent: Agent, envs, args: Args, global_step: int) -> None:
    ckpt = {"model": agent.state_dict(), "args": asdict(args), "global_step": global_step}
    if args.normalize:
        rms = envs.envs[0].get_wrapper_attr("obs_rms")
        ckpt["obs_rms"] = {"mean": rms.mean, "var": rms.var, "count": rms.count}
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(args), indent=2))

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=asdict(args), name=run_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(str(run_dir))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in asdict(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # SAME_STEP autoreset gives the classic gym semantics the PPO bookkeeping below assumes.
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_dir, args) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    agent = Agent(obs_dim, envs.single_action_space, args.hidden, args.activation).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    print(f"env={args.env_id} obs_dim={obs_dim} action_space={envs.single_action_space} device={device}")
    print(f"hidden={args.hidden} activation={args.activation} params={sum(p.numel() for p in agent.parameters())}")

    act_shape = envs.single_action_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + act_shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    episode_returns: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / args.num_iterations) * args.learning_rate

        # ---- rollout -----------------------------------------------------------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.action_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminated, truncated)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            # Gymnasium >= 1.0 vector envs report finished episodes under infos["final_info"];
            # older versions put them directly in infos["episode"]. Handle both.
            ep_info = infos.get("final_info", infos) if isinstance(infos, dict) else {}
            if "episode" in ep_info:
                for i in np.flatnonzero(ep_info["_episode"]):
                    r, l = float(ep_info["episode"]["r"][i]), int(ep_info["episode"]["l"][i])
                    episode_returns.append(r)
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)
                    if len(episode_returns) % 10 == 0:
                        print(f"step={global_step:>9d}  return={r:9.2f}  len={l:5d}  "
                              f"avg10={np.mean(episode_returns[-10:]):9.2f}  sps={int(global_step / (time.time() - start_time))}")

        # ---- GAE ---------------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal, nextvalues = 1.0 - next_done, next_value
                else:
                    nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---- optimisation ------------------------------------------------------------
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start:start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.action_value(b_obs[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = (b_values[mb] + torch.clamp(newvalue - b_values[mb], -args.clip_coef, args.clip_coef) - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if iteration % 10 == 0 or iteration == args.num_iterations:
            save_checkpoint(run_dir / "checkpoint.pt", agent, envs, args, global_step)

    save_checkpoint(run_dir / "checkpoint.pt", agent, envs, args, global_step)
    summary = {"env_id": args.env_id, "seed": args.seed, "global_step": global_step,
               "final_avg_return_last20": float(np.mean(episode_returns[-20:])) if episode_returns else None,
               "episodes": len(episode_returns), "wall_time_s": time.time() - start_time}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done:", json.dumps(summary))
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
