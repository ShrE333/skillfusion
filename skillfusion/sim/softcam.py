"""Software wrist camera: analytic ray-casting of the (all-cuboid) scene. Pure numpy.

Why: on an A100 (no RT cores) Isaac RTX cameras are unavailable, and the Newton-Warp camera renderer
needs the Newton physics backend, while Isaac Lab 3.0's Franka *lift* task is PhysX-only. Every object in
our scenes is a primitive box, so we can render exactly what the wrist camera would see straight from
the simulator's poses, independent of physics backend. (The gripper's own fingers are not drawn.)
All poses in the robot base frame; quaternions are Isaac Lab 3.0 order (x, y, z, w).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_R(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class Box:
    center: np.ndarray      # (3,)
    R: np.ndarray           # (3,3) local->base
    half: np.ndarray        # (3,)
    color: np.ndarray       # (3,) in 0..1


LIGHT = np.array([0.35, -0.25, 0.9]); LIGHT = LIGHT / np.linalg.norm(LIGHT)


def render(cam_pos, cam_R, boxes: list[Box], floor_z: float, size=(320, 240), fov_deg=75.0,
           sky=(0.62, 0.72, 0.85)) -> np.ndarray:
    """cam_R columns are the camera's ROS axes in the base frame: right, down, forward."""
    w, h = size
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(w) - (w - 1) / 2, np.arange(h) - (h - 1) / 2)
    d_cam = np.stack([u / f, v / f, np.ones_like(u)], -1).reshape(-1, 3)
    d = d_cam @ np.asarray(cam_R).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.asarray(cam_pos, dtype=np.float64)
    n = d.shape[0]
    best_t = np.full(n, np.inf)
    rgb = np.tile(np.array(sky), (n, 1))

    # floor with a 0.25 m checker
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (floor_z - o[2]) / d[:, 2]
    hit = (t > 0) & np.isfinite(t)
    p = o + d * t[:, None]
    chk = ((np.floor(p[:, 0] / 0.25) + np.floor(p[:, 1] / 0.25)) % 2).astype(bool)
    base = np.where(chk[:, None], 0.42, 0.34) * np.ones(3)
    rgb[hit] = base[hit] * (0.55 + 0.45 * LIGHT[2])
    best_t[hit] = t[hit]

    for b in boxes:
        lo = (d @ b.R)                         # ray dir in box frame
        lo_o = (o - b.center) @ b.R
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-b.half - lo_o) / lo
            t2 = (b.half - lo_o) / lo
        tmin = np.minimum(t1, t2); tmax = np.maximum(t1, t2)
        tmin = np.where(np.isnan(tmin), -np.inf, tmin); tmax = np.where(np.isnan(tmax), np.inf, tmax)
        t_near = tmin.max(1); t_far = tmax.min(1)
        ok = (t_far >= np.maximum(t_near, 1e-4)) & (t_near > 1e-4) & (t_near < best_t)
        if not ok.any():
            continue
        axis = tmin.argmax(1)
        sign = -np.sign(lo[np.arange(n), axis])
        nrm_local = np.zeros((n, 3)); nrm_local[np.arange(n), axis] = sign
        nrm = nrm_local @ b.R.T
        shade = 0.45 + 0.55 * np.clip(nrm @ LIGHT, 0, 1)
        rgb[ok] = (b.color[None, :] * shade[:, None])[ok]
        best_t[ok] = t_near[ok]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(h, w, 3)
