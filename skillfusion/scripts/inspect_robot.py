"""Print body/joint names of a robot so catalog.yaml (ee_body, wrist camera mount) can be verified.
    ./isaaclab.sh -p scripts/inspect_robot.py --robot franka --viz none"""
import argparse
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, parse_env_cfg
from skillfusion.catalog import Catalog

ap = argparse.ArgumentParser(); ap.add_argument("--robot", default="franka"); add_launcher_args(ap)
args = ap.parse_args()
cat = Catalog.load()
task = cat.skill_spec("lift")["base_task"][args.robot]
cfg = parse_env_cfg(task, num_envs=1)
with launch_simulation(cfg, args):
    import gymnasium as gym, isaaclab_tasks  # noqa
    env = gym.make(task, cfg=cfg)
    r = env.unwrapped.scene["robot"]
    print("bodies:", r.body_names); print("joints:", r.joint_names)
    print("catalog ee_body:", cat.raw["robots"][args.robot]["ee_body"])
    env.close()
