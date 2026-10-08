# Learning Locomotion Skills on Legged Robots with PPO

Code, logs and videos for *Learning Locomotion Behaviour on Legged Robots Using
Proximal Policy Optimization* (Samuel Oladejo, supervised by Prof. Benjamin
Rosman, University of the Witwatersrand). The report is in
`Learning_Locomotion_Skills_using_PPO.pdf`.

The trainer is a single-file PPO (CleanRL recipe, Huang et al. 2022) on the
Gymnasium 1.x API with MuJoCo 3 (`Ant-v4`, `HalfCheetah-v4`) and PyBullet
(`MinitaurBulletEnv-v0`, `HalfCheetahBulletEnv-v0`) plus `CartPole-v1` as a
sanity check.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

(or `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`)

## Train

```bash
.venv/bin/python ppo/train.py --env-id Ant-v4 --seed 1
.venv/bin/python ppo/train.py --env-id MinitaurBulletEnv-v0 --hidden 128 128 --activation relu
.venv/bin/python ppo/train.py --env-id CartPole-v1 --total-timesteps 100000
```

Per-environment defaults (timesteps, learning rate, minibatches, reward clip)
live in `ENV_PRESETS` inside `ppo/train.py`; any flag on the command line
overrides them. Run `ppo/train.py --help` for the full list.

Each run writes to `runs/<env>__<exp-name>__<seed>__<time>/`:
`config.json`, TensorBoard events, `checkpoint.pt` (policy weights **and** the
observation-normalisation statistics, so the policy can be replayed), and
`summary.json`.

Several seeds at once (one CPU core each):

```bash
ppo/run_seeds.sh Ant-v4 tanh64 "--hidden 64 64 --activation tanh" 1 2 3 4 5
```

Videos: pass `--capture-video` (training) or `--video` (evaluation). MuJoCo renders
off-screen; on a headless Linux box set `MUJOCO_GL=osmesa` (EGL needs a working GPU display stack).

## Evaluate and plot

```bash
.venv/bin/python ppo/evaluate.py runs/<run>/checkpoint.pt --episodes 10 --video
.venv/bin/python ppo/plot.py runs --env-id Ant-v4            # mean +/- std over seeds per exp-name
tensorboard --logdir runs
```

## Layout

```
ppo/train.py       PPO trainer (rollout, GAE, clipped objective, logging, checkpoints)
ppo/agent.py       actor-critic MLP (Gaussian head for Box, categorical head for Discrete)
ppo/evaluate.py    deterministic evaluation of a checkpoint, optional video
ppo/plot.py        aggregate TensorBoard logs across seeds
ppo/minitaur_env.py Gymnasium wrapper for PyBullet's Minitaur (registers MinitaurBulletEnv-v0)
ppo/run_seeds.sh   launch N seeds in parallel
ant_ppo/ halfcheetah_ppo/ minitaur_ppo/ cartpole_ppo/
                   original per-environment scripts from the report (gym 0.23 API,
                   kept for provenance; bugs fixed but not maintained)
*_graphs/ *_videos/ runs/ wandb/ inside the legacy dirs
                   artefacts from the November 2022 experiments
```

## More robots

Any Gymnasium MuJoCo or PyBullet locomotion task works; presets exist for
`Hopper`, `Walker2d`, `Humanoid` (`-v4`/`-v5`) and `AntBulletEnv-v0`,
`HopperBulletEnv-v0`, `Walker2DBulletEnv-v0`, `HumanoidBulletEnv-v0`:

```bash
.venv/bin/python ppo/train.py --env-id Humanoid-v4 --seed 1
```

## Other algorithms (SAC, TD3, SB3-PPO)

`baselines/sb3_train.py` trains Stable-Baselines3 agents on the same environments and
logs to the same TensorBoard tags, so `ppo/plot.py` compares them directly:

```bash
.venv/bin/python baselines/sb3_train.py --algo sac --env-id Ant-v4 --seed 1 --cuda
.venv/bin/python baselines/sb3_train.py --algo td3 --env-id MinitaurBulletEnv-v0 --seed 1
.venv/bin/python ppo/plot.py runs --env-id Ant-v4     # tanh64 (ours) vs sb3-sac vs sb3-td3
```

## Real robot models on the GPU: MuJoCo Playground

`playground/` is a separate uv project (GPU JAX + MJX) with thousands of parallel
environments for Unitree Go1, Boston Dynamics Spot, Google Barkour, Unitree H1 and G1,
Booster T1, Berkeley Humanoid and Robotis OP3:

```bash
cd playground && uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python train_playground.py --list
.venv/bin/python train_playground.py --env Go1JoystickFlatTerrain      # writes params, curve and rollout.mp4
.venv/bin/python train_playground.py --env G1JoystickFlatTerrain       # humanoid
```
