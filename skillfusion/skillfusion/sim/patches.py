"""Per-skill patches applied on top of the stock Isaac Lab task cfgs (referenced from catalog.yaml).

Every patch takes ``(env_cfg, catalog)`` and edits the cfg in place. Goals are commanded in the
robot base frame, which is why a policy trained here works in any scene the workcell is moved into.
"""
from __future__ import annotations

from ..catalog import Catalog


def _region_ranges(cat: Catalog, region: str, z: tuple[float, float]) -> dict:
    r = cat.workcell["regions"][region]
    (cx, cy, _), (hx, hy, _) = r["center"], r["half_extent"]
    return dict(pos_x=(cx - hx, cx + hx), pos_y=(cy - hy, cy + hy), pos_z=z)


def _set_goal(env_cfg, ranges: dict) -> None:
    rg = env_cfg.commands.object_pose.ranges
    for k, v in ranges.items():
        setattr(rg, k, v)


def patch_pick_place(env_cfg, cat: Catalog) -> None:
    """Goal = anywhere on the table in regions A/B/C (random), so the same policy serves any A->B.
    The stock lift reward already pays for reaching, lifting and goal tracking. Carry height is
    kept low so the policy learns to *set the block down* near the goal."""
    boxes = [_region_ranges(cat, r, (0.05, 0.12)) for r in ("A", "B", "C")]
    env_cfg.commands.object_pose.ranges.pos_x = (min(b["pos_x"][0] for b in boxes), max(b["pos_x"][1] for b in boxes))
    env_cfg.commands.object_pose.ranges.pos_y = (min(b["pos_y"][0] for b in boxes), max(b["pos_y"][1] for b in boxes))
    env_cfg.commands.object_pose.ranges.pos_z = (0.05, 0.12)


def patch_lift(env_cfg, cat: Catalog) -> None:
    pass    # stock task: hover goal 0.25-0.5 m


def patch_push(env_cfg, cat: Catalog) -> None:
    """Goal on the table surface; drop the 'must be lifted' gate so pushing is rewarded."""
    env_cfg.commands.object_pose.ranges.pos_z = (0.03, 0.03)
    for name in ("object_goal_tracking", "object_goal_tracking_fine_grained"):
        term = getattr(env_cfg.rewards, name)
        term.params["minimal_height"] = 0.0


def patch_reach(env_cfg, cat: Catalog) -> None:
    pass    # stock reach task


def patch_open_drawer(env_cfg, cat: Catalog) -> None:
    pass    # stock cabinet task


def apply(name: str, env_cfg, cat: Catalog) -> None:
    """Resolve 'module:function' from the catalog and apply it."""
    import importlib
    mod, fn = name.split(":")
    getattr(importlib.import_module(mod), fn)(env_cfg, cat)
