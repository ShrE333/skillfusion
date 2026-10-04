"""Environment factories (need Isaac Lab; import lazily inside the launch context).

Training envs: stock Isaac Lab tasks + a skill patch, state observations only (10 GB VRAM friendly).
Demo env:      Franka lift task with 1 env, the scene from ``Placement`` as static cuboids, extra
               free blocks, and a wrist camera (Newton-Warp renderer -> works without RT cores).
"""
from __future__ import annotations

import math

from ..catalog import Catalog
from ..scene import Placement
from . import patches


def train_cfg(cat: Catalog, skill: str, robot: str, num_envs: int, physics: str | None = None):
    from isaaclab_tasks.utils import parse_env_cfg
    spec = cat.skill_spec(skill)
    task = spec["base_task"].get(robot)
    if task is None:
        raise ValueError(f"skill {skill} has no task for robot {robot}")
    cfg = parse_env_cfg(task, num_envs=num_envs)
    if spec.get("patch"):
        patches.apply(spec["patch"], cfg, cat)
    return task, cfg


def _yaw_quat_xyzw(deg: float):
    h = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(h), math.cos(h))          # Isaac Lab 3.0: x, y, z, w


def demo_cfg(cat: Catalog, placement: Placement, blocks: dict[str, tuple[float, float]], robot: str = "franka",
             camera: str | None = "soft", cam_size: tuple[int, int] = (320, 240)):
    """Build the demo env cfg. ``blocks``: {prop_name: (x, y)} on the table in the robot base frame."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.utils import parse_env_cfg

    task = cat.skill_spec("pick_place")["base_task"][robot]
    cfg = parse_env_cfg(task, num_envs=1)
    cfg.scene.env_spacing = 6.0

    # self-contained table (the stock one needs Nucleus assets): top surface at z = 0
    cfg.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, -0.5), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(size=(0.9, 1.2, 1.0), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.2))))

    # scene objects from the catalog, already expressed in the robot base frame
    floor = cat.workcell["floor_z"]
    for name, o in placement.objects_in_base_frame.items():
        sx, sy, sz = o["size"] if len(o["size"]) == 3 else (*o["size"][:2], 1.0)
        spec = cat.raw["environments"][placement.env]["objects"][name]
        height = spec["size"][2] if len(spec["size"]) > 2 else 1.0
        setattr(cfg.scene, f"obj_{name}", AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{name}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(o["xy"][0], o["xy"][1], floor + height / 2),
                                                    rot=_yaw_quat_xyzw(o["yaw_deg"])),
            spawn=sim_utils.CuboidCfg(size=(spec["size"][0], spec["size"][1], height),
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(o["color"])))))

    # free blocks
    # The first block takes the stock ``object`` slot (rewards/terminations reference it); the rest are extra.
    entity: dict[str, str] = {}
    for i, (name, (x, y)) in enumerate(blocks.items()):
        p = cat.raw["props"][name]
        size = p["size"] if isinstance(p["size"], list) else [p["size"]] * 3
        entity[name] = "object" if i == 0 else name
        setattr(cfg.scene, entity[name], RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object" if i == 0 else f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, size[2] / 2 + 0.002), rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=sim_utils.CuboidCfg(
                size=tuple(size), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=p["mass"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(p["color"])))))

    # camera="soft": analytic ray-cast (sim/softcam.py) - works on any GPU/physics backend (default).
    # camera="sensor": Isaac Lab CameraCfg with the kit-less Newton-Warp renderer (needs the Newton physics backend).
    if camera == "sensor":
        cam = cat.raw["robots"][robot]["wrist_camera"]
        kw = {}
        try:      # kit-less Warp renderer: no RT cores needed (A100). rgb / depth only.
            from isaaclab_newton.renderers import NewtonWarpRendererCfg
            kw["renderer_cfg"] = NewtonWarpRendererCfg()
        except ImportError:
            pass
        cfg.scene.wrist_cam = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam['mount_body']}/wrist_cam",
            update_period=0.0, width=cam_size[0], height=cam_size[1], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.02, 5.0)),
            offset=CameraCfg.OffsetCfg(pos=tuple(cam["offset_pos"]), rot=tuple(cam["offset_rot_xyzw"]),
                                       convention=cam["convention"]), **kw)
    cfg.skillfusion_scene = {         # read by sim.skills.SkillRunner for the software camera
        "floor_z": floor,
        "static": [{"center": [0.55, 0.0, -0.5], "half": [0.45, 0.6, 0.5], "yaw_deg": 0.0, "color": [0.45, 0.32, 0.2]}] + [
            {"center": [o["xy"][0], o["xy"][1], floor + (cat.raw["environments"][placement.env]["objects"][n]["size"][2:] or [1.0])[0] / 2],
             "half": [o["size"][0] / 2, o["size"][1] / 2, (cat.raw["environments"][placement.env]["objects"][n]["size"][2:] or [1.0])[0] / 2],
             "yaw_deg": o["yaw_deg"], "color": o["color"]} for n, o in placement.objects_in_base_frame.items()],
        "blocks": {n: {"half": [(cat.raw["props"][n]["size"] if isinstance(cat.raw["props"][n]["size"], list) else [cat.raw["props"][n]["size"]] * 3)[i] / 2 for i in range(3)],
                       "color": cat.raw["props"][n]["color"]} for n in blocks},
        "camera": cat.raw["robots"][robot]["wrist_camera"], "size": list(cam_size)}
    cfg.skillfusion_entities = entity      # prop name -> scene entity name (read by sim.api.Scene)
    return task, cfg
