"""Aggregate TensorBoard logs across seeds into mean +/- std learning curves.

    python ppo/plot.py runs --env-id Ant-v4 --tag charts/episodic_return --out ant_returns.png

Runs are grouped by experiment name (the part of the run directory between the
env id and the seed), so `--exp-name relu128` on several seeds becomes one curve.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(run_dir: Path, tag: str):
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", type=Path)
    p.add_argument("--env-id", required=True)
    p.add_argument("--tag", default="charts/episodic_return")
    p.add_argument("--bins", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    groups: dict[str, list] = defaultdict(list)
    for run in sorted(a.runs_dir.glob(f"{a.env_id}__*")):
        parts = run.name.split("__")
        if len(parts) < 4:
            continue
        data = load_scalar(run, a.tag)
        if data is not None and len(data[0]) > 1:
            groups[parts[1]].append(data)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, runs in sorted(groups.items()):
        max_step = min(r[0].max() for r in runs)
        grid = np.linspace(0, max_step, a.bins)
        curves = np.stack([np.interp(grid, s, v) for s, v in runs])
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(grid, mean, label=f"{name} (n={len(runs)})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        print(f"{name:20s} seeds={len(runs)} final={mean[-1]:.1f} +/- {std[-1]:.1f}")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(a.tag.split("/")[-1])
    ax.set_title(a.env_id)
    ax.legend()
    fig.tight_layout()
    out = a.out or Path(f"{a.env_id}_{a.tag.split('/')[-1]}.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
