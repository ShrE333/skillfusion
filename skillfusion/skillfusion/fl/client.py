"""NVFlare client: one process = one skill. Each round: receive global trunk -> local PPO -> send trunk.

Launched by ``fl/job.py`` (the recipe passes the arguments). Private adapter/head/value never leave
this process; they are saved to ``runs/fl/private_<skill>.npz`` and stitched in at fusion time.
"""
from __future__ import annotations

import os

import numpy as np

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

from ..catalog import Catalog
from ..rl.local import base_parser, open_skill_env
from ..rl.ppo import PPOCfg, SkillTrainer


def main():
    ap = base_parser()
    ap.add_argument("--local_iters", type=int, default=40, help="PPO iterations per federated round")
    ap.add_argument("--prox_mu", type=float, default=0.01)
    args = ap.parse_args()
    cat = Catalog.load()
    flare.init()
    with open_skill_env(args, cat) as (env, policy):
        trainer = SkillTrainer(env, policy, args.skill, PPOCfg(prox_mu=args.prox_mu), device=args.device or "cuda:0")
        os.makedirs(os.path.join(args.out, "fl"), exist_ok=True)
        while flare.is_running():
            model = flare.receive()
            policy.trunk.load_state_dict({k: _t(v) for k, v in model.params.items()})
            trainer.set_global_trunk()
            stats = trainer.train(args.local_iters, log_every=10)
            trunk = {k: v.detach().cpu() for k, v in policy.trunk.state_dict().items()}
            flare.send(FLModel(params=trunk, metrics={"mean_reward": stats["mean_reward"]},
                               meta={"NUM_STEPS_CURRENT_ROUND": int(stats["samples"])}))
            np.savez(os.path.join(args.out, "fl", f"trunk_{args.skill}.npz"), **{k: v.numpy() for k, v in trunk.items()})
            private = {k: v for k, v in policy.params_numpy().items() if not k.startswith("trunk.")}
            np.savez(os.path.join(args.out, "fl", f"private_{args.skill}.npz"), **private)


def _t(v):
    import torch
    return v if isinstance(v, torch.Tensor) else torch.as_tensor(v)


if __name__ == "__main__":
    main()
