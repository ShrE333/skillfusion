"""Placement math: put the (fixed) robot workcell beside / in front of / behind an anchor object.

Everything the simulator needs is expressed in the *robot base frame*, so trained policies never
see a different world: we move the scene around the workcell, not the other way round.
Pure numpy, no Isaac dependency -> unit-testable anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalog import Catalog

RELATIONS = {
    # relation -> (direction in anchor frame (unit), robot faces: "same" | "toward" | "away")
    "beside": ((0.0, -1.0), "same"),
    "right_of": ((0.0, -1.0), "same"),
    "left_of": ((0.0, 1.0), "same"),
    "in_front_of": ((1.0, 0.0), "toward"),
    "behind": ((-1.0, 0.0), "toward"),
}
RELATION_ALIASES = {
    "next to": "beside", "next_to": "beside", "by": "beside", "near": "beside", "beside": "beside",
    "besides": "beside", "right": "right_of", "left": "left_of", "in front": "in_front_of",
    "in front of": "in_front_of", "front": "in_front_of", "behind": "behind", "back": "behind",
}


def normalize_relation(text: str) -> str:
    t = str(text).lower().strip().replace("-", " ")
    if t in RELATIONS:
        return t
    if t in RELATION_ALIASES:
        return RELATION_ALIASES[t]
    t2 = t.replace(" ", "_")
    if t2 in RELATIONS:
        return t2
    raise ValueError(f"unknown relation {text!r}; options: {sorted(RELATIONS)}")


def rot(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def rect_corners(center, half, yaw_deg) -> np.ndarray:
    hx, hy = half
    local = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    return np.asarray(center) + local @ rot(yaw_deg).T


def rects_overlap(c1, h1, y1, c2, h2, y2, margin: float = 0.0) -> bool:
    """Separating-axis test for two oriented rectangles (2D)."""
    a = rect_corners(c1, (h1[0] + margin, h1[1] + margin), y1)
    b = rect_corners(c2, h2, y2)
    axes = [rot(y1)[:, 0], rot(y1)[:, 1], rot(y2)[:, 0], rot(y2)[:, 1]]
    for ax in axes:
        pa, pb = a @ ax, b @ ax
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


@dataclass
class Placement:
    env: str
    anchor: str
    relation: str
    base_xy: tuple[float, float]          # robot base in world
    yaw_deg: float                        # robot forward direction in world
    ok: bool
    conflicts: list[str] = field(default_factory=list)
    objects_in_base_frame: dict[str, dict] = field(default_factory=dict)


def _footprint(cat: Catalog):
    fp = cat.workcell["footprint"]
    center = np.array([(fp["x_min"] + fp["x_max"]) / 2, (fp["y_min"] + fp["y_max"]) / 2])
    half = np.array([(fp["x_max"] - fp["x_min"]) / 2, (fp["y_max"] - fp["y_min"]) / 2])
    return center, half


def place(cat: Catalog, env: str, anchor: str, relation: str, gap: float = 0.15,
          lateral: float = 0.0) -> Placement:
    env = cat.env(env)
    anchor = cat.anchor(env, anchor)
    relation = normalize_relation(relation)
    objs = cat.raw["environments"][env]["objects"]
    a = objs[anchor]
    ax, ay, ayaw = a["pose"]
    a_half = np.array([a["size"][0] / 2, a["size"][1] / 2])
    a_center = np.array([ax, ay])

    dir_local, facing = RELATIONS[relation]
    u = rot(ayaw) @ np.array(dir_local)                    # world direction away from the anchor
    side = rot(ayaw) @ np.array([-dir_local[1], dir_local[0]])
    if facing == "same":
        yaw = ayaw
    elif facing == "toward":
        yaw = np.degrees(np.arctan2(-u[1], -u[0]))
    else:
        yaw = np.degrees(np.arctan2(u[1], u[0]))

    fp_center, fp_half = _footprint(cat)
    R = rot(yaw)

    def footprint_at(t: float):
        base = a_center + u * t + side * lateral
        return base, base + R @ fp_center

    t = 0.0
    while True:                                           # march outward until the footprint clears the anchor
        base, fc = footprint_at(t)
        if not rects_overlap(fc, fp_half, yaw, a_center, a_half, ayaw, margin=gap):
            break
        t += 0.02
        if t > 10:
            raise RuntimeError("could not place workcell")

    conflicts = []
    for name, o in objs.items():
        if name == anchor:
            continue
        oh = np.array([o["size"][0] / 2, o["size"][1] / 2])
        if rects_overlap(fc, fp_half, yaw, np.array(o["pose"][:2]), oh, o["pose"][2]):
            conflicts.append(name)

    frame = {}
    Rinv = rot(-yaw)
    for name, o in objs.items():
        p = Rinv @ (np.array(o["pose"][:2]) - base)
        frame[name] = {"xy": p.tolist(), "yaw_deg": float(o["pose"][2] - yaw), "size": list(o["size"]),
                       "color": list(o["color"]), "z_base": cat.workcell["floor_z"]}
    return Placement(env, anchor, relation, (float(base[0]), float(base[1])), float(yaw),
                     ok=not conflicts, conflicts=conflicts, objects_in_base_frame=frame)


def suggest_relations(cat: Catalog, env: str, anchor: str) -> list[str]:
    """Relations that place the workcell without collisions (used when the user's choice conflicts)."""
    good = []
    for rel in RELATIONS:
        try:
            if place(cat, env, anchor, rel).ok:
                good.append(rel)
        except RuntimeError:
            pass
    return good
