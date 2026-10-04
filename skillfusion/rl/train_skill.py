"""Train ONE skill locally (no federation) - baseline and smoke test.

    python -m skillfusion.rl.train_skill --skill pick_place --iterations 300 --viz none
"""
from __future__ import annotations

import os

import numpy as np

from ..catalog import Catalog
from .ppo import PPOCfg, SkillTrainer
from .local import base_parser, open_skill_env


def main():
    args = base_parser().parse_args()
    cat = Catalog.load()
    with open_skill_env(args, cat) as (env, policy):
        trainer = SkillTrainer(env, policy, args.skill, PPOCfg(), device=args.device or "cuda:0")
        stats = trainer.train(args.iterations)
        os.makedirs(args.out, exist_ok=True)
        np.savez(os.path.join(args.out, f"solo_{args.skill}.npz"), **policy.params_numpy())
        print("done", stats)


if __name__ == "__main__":
    main()
