"""Shared helpers for training one skill on a real Isaac Lab env (needs Isaac Lab; import inside the launch context)."""
from __future__ import annotations

import argparse
import contextlib

from ..catalog import Catalog
from ..fl.model import FusedPolicy


def base_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--robot", default="franka")
    ap.add_argument("--num_envs", type=int, default=256, help="keep small on a 10 GB GPU")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--out", default="runs")
    from isaaclab_tasks.utils import add_launcher_args
    add_launcher_args(ap)
    return ap


@contextlib.contextmanager
def open_skill_env(args, cat: Catalog):
    """Yields (env, policy) for args.skill. Training envs use state observations only (no cameras)."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation
    from ..sim import envs
    task, cfg = envs.train_cfg(cat, args.skill, args.robot, args.num_envs)
    cfg.sim.device = getattr(args, "device", None) or cfg.sim.device
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg)
        u = env.unwrapped
        obs_dim = u.single_observation_space["policy"].shape[0]
        act_dim = u.single_action_space.shape[0]
        policy = FusedPolicy({args.skill: dict(index=cat.skill_index(args.skill), obs_dim=obs_dim, act_dim=act_dim)},
                             cat.max_skills)
        try:
            yield env, policy
        finally:
            env.close()
