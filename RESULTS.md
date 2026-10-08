# Results (October 2026 rerun)

PPO, 64-64 tanh actor and critic, default presets in `ppo/train.py`, 3 seeds each, one CPU core per
run on a loaded 12-core machine. "Train" is the mean of the last 20 stochastic training episodes;
"Eval" is the deterministic (mean-action) policy over 10 episodes (Ant) or 5 episodes (Minitaur)
with a frozen observation normaliser. Curves, evaluation videos and the Go1 rollout are committed under `results/`.

## Ant-v4 (MuJoCo), 1M steps

| seed | train return | eval return (mean ± std) | eval length |
|-----:|-------------:|-------------------------:|------------:|
| 1 | 2416 | 3326 ± 257 | 981 |
| 2 | 1773 | 2662 ± 536 | 963 |
| 3 | 1723 | 2646 ± 57  | 1000 |
| **mean over seeds** | 1971 | **2878** | |

Reference: CleanRL's `ppo_continuous_action` reports roughly 2000 to 3000 on Ant-v4 at 1M steps
with the same recipe, so the rerun is in the expected range. The 2022 report trained Ant-v3 with
a modified reward (`healthy_reward=5`, `healthy_z_range=(0.5, 2)`), so its numbers are not comparable;
pass `--env-kwargs '{"healthy_reward": 5, "healthy_z_range": [0.5, 2]}'` to reproduce that setting.

## MinitaurBulletEnv-v0 (PyBullet), 500k steps

| seed | train return | eval return (mean ± std) | eval length |
|-----:|-------------:|-------------------------:|------------:|
| 1 | 11.1 | 18.3 ± 4.4 | 840 |
| 2 | 11.3 | 19.3 ± 2.1 | 959 |
| 3 | 13.7 | 27.1 ± 1.4 | 1000 |
| **mean over seeds** | 12.0 | **21.6** | |

The environment's registered "solved" threshold is 5.0; a random policy scores about -1.7.

## Videos

`results/*.mp4` (copied from `runs/<run>/eval_videos/`, produced by `ppo/evaluate.py --video`) (Ant uses the model's tracking
camera; Minitaur uses PyBullet's software renderer). Training-time videos: `ppo/train.py --capture-video`.

## Still to do for a paper

- 5 to 10 seeds per configuration and interquartile-mean confidence intervals (`rliable`).
- The architecture and activation grid from the report (`--hidden`, `--activation`), run through
  `ppo/run_seeds.sh`.
- Off-policy baselines at equal environment steps: `baselines/sb3_train.py --algo sac|td3`.
- Compute-matched runs on MuJoCo Playground's Go1 or Spot (`playground/train_playground.py`).

## MuJoCo Playground, Unitree Go1 joystick (MJX on an RTX 3090)

`playground/train_playground.py --env Go1JoystickFlatTerrain --timesteps 5000000` ran 14.7M
environment steps (8192 parallel envs, Brax PPO) in 297 s including JIT compilation, roughly
100k steps/s once compiled, against about 800 steps/s for the single-process CPU PPO above.
Eval episode reward peaked at 1.20 at 6.5M steps and ended at 0.32: at 7% of the preset's
budget the gait has not converged. A full-budget run writes `params.msgpack`, `progress.png`
and `rollout.mp4` to `playground/runs/<env>__<time>/`.
