"""Confirm every base_task in configs/catalog.yaml is registered in YOUR Isaac Lab install.
    ./isaaclab.sh -p scripts/list_tasks.py"""
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from skillfusion.catalog import Catalog

cat = Catalog.load()
registered = set(gym.registry.keys())
bad = 0
for skill, spec in cat.raw["skills"].items():
    for robot, task in spec["base_task"].items():
        ok = task in registered
        bad += not ok
        print(f"{'OK ' if ok else 'MISSING'} {skill:12s} {robot:7s} {task}")
print("all good" if not bad else f"{bad} task id(s) missing - fix configs/catalog.yaml (wxai ids appear after the Trossen port)")
