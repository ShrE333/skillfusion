"""The tiny API the LLM-written programs see (see agent.codegen.API_DOC)."""
from __future__ import annotations

from ..catalog import Catalog
from .skills import SkillRunner


class RobotAPI:
    def __init__(self, cat: Catalog, runner: SkillRunner, log=print):
        self._cat, self._r, self._log = cat, runner, log
        self._slots: dict[str, int] = {}

    def _goal(self, g, spread: bool = False):
        if not isinstance(g, str):
            return list(g)
        c = self._cat.region_center(g)
        if spread:                       # several blocks to one region: fan them out so they don't collide
            r = self._cat.region(g)
            k = self._slots.get(r, 0)
            self._slots[r] = k + 1
            c[0] += ((k % 3) - 1) * 0.07
            c[1] += (k // 3) * 0.07
        return c

    def reach(self, goal, steps: int = 120):
        self._log(f"reach {goal}"); self._r.reach(self._goal(goal), steps)

    def pick_place(self, obj: str, goal, steps: int = 250):
        obj = self._cat.prop(obj)
        self._log(f"pick_place {obj} -> {goal}"); self._r.pick_place(obj, self._goal(goal, spread=True), steps)

    def lift(self, obj: str, steps: int = 150):
        self._r.lift(self._cat.prop(obj), steps)

    def push(self, obj: str, goal, steps: int = 200):
        self._r.push(self._cat.prop(obj), self._goal(goal), steps)

    def open_drawer(self, steps: int = 250):
        self._r.open_drawer(steps)

    def gripper(self, open: bool):
        self._r.gripper(bool(open))

    def camera_rgb(self):
        return self._r.camera_frame()


class SceneAPI:
    def __init__(self, runner: SkillRunner):
        self._r = runner

    def position(self, obj: str):
        return self._r.obj_pos_b(obj)[0].tolist()

    def props(self):
        return list(self._r.entities)
