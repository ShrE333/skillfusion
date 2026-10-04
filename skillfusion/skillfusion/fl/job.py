"""Server side: FedAvg over skill clients with NVFlare's recipe API.

    python -m skillfusion.fl.job --skills pick_place,lift,push --rounds 6 --num_envs 128

NVFlare's API moves between minor versions - if an import below fails, check
https://nvflare.readthedocs.io (FedAvgRecipe / SimEnv) and adjust the two marked lines.
"""
from __future__ import annotations

import argparse
import os
import sys

from ..catalog import Catalog
from .model import Trunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="pick_place,lift,push")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--local_iters", type=int, default=40)
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--workspace", default="runs/nvflare")
    args = ap.parse_args()
    skills = args.skills.split(",")
    cat = Catalog.load()
    for s in skills:
        cat.skill(s)

    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe          # <- verify path for your NVFlare version
    from nvflare.recipe import SimEnv                                    # <- verify path for your NVFlare version

    script = os.path.join(os.path.dirname(__file__), "client.py")
    # Heterogeneous clients: per-client args (skill differs). If your recipe version only supports one
    # train_args string, run one job per skill on Nebius (jobs/nebius_fl_client.sh) and use the POC server.
    recipe = FedAvgRecipe(
        name="skillfusion", min_clients=len(skills), num_rounds=args.rounds, initial_model=Trunk(),
        train_script=script,
        train_args=f"--skill {skills[0]} --num_envs {args.num_envs} --local_iters {args.local_iters} --viz none",
        per_site_config={f"site-{i + 1}": {"train_args": f"--skill {s} --num_envs {args.num_envs} "
                                                        f"--local_iters {args.local_iters} --viz none"}
                         for i, s in enumerate(skills)},
    )
    recipe.execute(SimEnv(num_clients=len(skills), workspace_root=args.workspace))
    print("federated training finished; run: python -m skillfusion.fl.fuse_cli")


if __name__ == "__main__":
    sys.exit(main())
