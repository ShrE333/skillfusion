"""Run skills on the demo env: a learned policy (FusedPolicy) if a checkpoint exists, otherwise a
scripted IK controller so the demo never depends on training having converged.

Franka IK-Rel action = [dx, dy, dz, droll, dpitch, dyaw, gripper], scale 0.5 (see catalog note);
gripper > 0 opens, < 0 closes (BinaryJointPositionAction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.manager_based.manipulation.lift.mdp import object_position_in_robot_root_frame
from isaaclab.envs.mdp import joint_pos_rel, joint_vel_rel, last_action

OPEN, CLOSE = 1.0, -1.0
ACTION_SCALE = 0.5


@dataclass
class Frame:
    step: int
    ee: list[float]
    obj: list[float] | None
    note: str


class SkillRunner:
    def __init__(self, env, entities: dict[str, str], scene_meta: dict | None = None, ee_body: str = "panda_hand", ee_offset=(0.0, 0.0, 0.107),
                 policy=None, on_step: Callable[[Frame], None] | None = None):
        self.env = env
        self.u = env.unwrapped
        self.entities = entities
        self.robot = self.u.scene["robot"]
        self.ee_idx = self.robot.find_bodies(ee_body)[0][0]
        self.ee_offset = torch.tensor(ee_offset, device=self.u.device)
        self.policy = policy                      # optional FusedPolicy
        self.on_step = on_step
        self.scene_meta = scene_meta
        self.t = 0
        self.grip = OPEN
        self.env.reset()

    # ---- state -------------------------------------------------------------------------------
    def ee_pos_b(self) -> torch.Tensor:
        d = self.robot.data
        pos_w = d.body_pos_w.torch[:, self.ee_idx]
        quat_w = d.body_quat_w.torch[:, self.ee_idx]
        from isaaclab.utils.math import quat_apply
        tip_w = pos_w + quat_apply(quat_w, self.ee_offset.expand(pos_w.shape[0], -1))
        tip_b, _ = subtract_frame_transforms(d.root_pos_w.torch, d.root_quat_w.torch, tip_w)
        return tip_b

    def obj_pos_b(self, prop: str) -> torch.Tensor:
        return object_position_in_robot_root_frame(self.u, object_cfg=SceneEntityCfg(self.entities[prop]))

    # ---- software wrist camera ----------------------------------------------------------------
    def camera_frame(self):
        """RGB (H,W,3 uint8) of what the wrist camera sees, ray-cast from live poses (sim/softcam.py)."""
        import numpy as np
        from .softcam import Box, render, quat_to_R
        m = self.scene_meta
        if not m:
            return None
        d = self.robot.data
        pos_w = d.body_pos_w.torch[:, self.ee_idx]
        quat_w = d.body_quat_w.torch[:, self.ee_idx]
        p_b, q_b = subtract_frame_transforms(d.root_pos_w.torch, d.root_quat_w.torch, pos_w, quat_w)
        cam = m["camera"]
        R_hand = quat_to_R(q_b[0].cpu().numpy())
        R_off = quat_to_R(cam["offset_rot_xyzw"])
        cam_pos = p_b[0].cpu().numpy() + R_hand @ np.array(cam["offset_pos"])
        boxes = []
        for s in m["static"]:
            a = np.radians(s["yaw_deg"])
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            boxes.append(Box(np.array(s["center"]), R, np.array(s["half"]), np.array(s["color"])))
        for name, b in m["blocks"].items():
            ent = self.u.scene[self.entities[name]]
            pw, qw = ent.data.root_pos_w.torch[:, :3], ent.data.root_quat_w.torch
            pb, qb = subtract_frame_transforms(d.root_pos_w.torch, d.root_quat_w.torch, pw, qw)
            boxes.append(Box(pb[0].cpu().numpy(), quat_to_R(qb[0].cpu().numpy()), np.array(b["half"]), np.array(b["color"])))
        return render(cam_pos, R_hand @ R_off, boxes, m["floor_z"], size=tuple(m["size"]))

    # ---- one control step --------------------------------------------------------------------
    def _step(self, target_b: torch.Tensor, grip: float, prop: str | None, note: str) -> float:
        err = target_b - self.ee_pos_b()
        act = torch.zeros(self.u.num_envs, 7, device=self.u.device)
        act[:, :3] = (err / ACTION_SCALE * 3.0).clamp(-1.0, 1.0)
        act[:, 6] = grip
        self.env.step(act)
        self.t += 1
        if self.on_step:
            self.on_step(Frame(self.t, self.ee_pos_b()[0].tolist(),
                               self.obj_pos_b(prop)[0].tolist() if prop else None, note))
        return float(err.norm())

    def _goto(self, target, grip, prop, note, tol=0.012, max_steps=150):
        tgt = torch.as_tensor(target, device=self.u.device, dtype=torch.float32).expand(self.u.num_envs, 3)
        for _ in range(max_steps):
            if self._step(tgt, grip, prop, note) < tol:
                return True
        return False

    def _hold(self, grip, n, prop, note):
        cur = self.ee_pos_b()
        for _ in range(n):
            self._step(cur, grip, prop, note)

    # ---- skills ------------------------------------------------------------------------------
    def reach(self, goal, steps=150):
        self._goto(goal, self.grip, None, f"reach {goal}", max_steps=steps)

    def gripper(self, open_: bool, n=25):
        self.grip = OPEN if open_ else CLOSE
        self._hold(self.grip, n, None, "open gripper" if open_ else "close gripper")

    def pick_place(self, prop: str, goal, steps=250, hover=0.16, grasp_z=0.02):
        """Scripted pick-and-place: above -> descend -> grasp -> lift -> above goal -> descend -> release."""
        if self.policy is not None:
            return self._run_policy("pick_place", prop, goal, steps)
        p = self.obj_pos_b(prop)[0].tolist()
        g = list(goal)
        self.gripper(True, 5)
        self._goto([p[0], p[1], hover], OPEN, prop, f"above {prop}")
        self._goto([p[0], p[1], grasp_z], OPEN, prop, f"descend to {prop}")
        self.gripper(False, 30)
        self._goto([p[0], p[1], hover], CLOSE, prop, f"lift {prop}")
        self._goto([g[0], g[1], hover], CLOSE, prop, "carry")
        self._goto([g[0], g[1], grasp_z + 0.01], CLOSE, prop, "lower")
        self.gripper(True, 25)
        self._goto([g[0], g[1], hover], OPEN, prop, "retreat")

    def lift(self, prop: str, steps=150):
        p = self.obj_pos_b(prop)[0].tolist()
        self._goto([p[0], p[1], 0.16], OPEN, prop, "above")
        self._goto([p[0], p[1], 0.02], OPEN, prop, "descend")
        self.gripper(False, 30)
        self._goto([p[0], p[1], 0.30], CLOSE, prop, "lift")

    def push(self, prop: str, goal, steps=200):
        p = self.obj_pos_b(prop)[0].tolist()
        g = list(goal)
        d = torch.tensor([g[0] - p[0], g[1] - p[1]])
        d = (d / d.norm().clamp_min(1e-6)).tolist()
        self.gripper(False, 10)
        self._goto([p[0] - 0.09 * d[0], p[1] - 0.09 * d[1], 0.06], CLOSE, prop, "behind block")
        self._goto([g[0] - 0.05 * d[0], g[1] - 0.05 * d[1], 0.06], CLOSE, prop, "push")

    def open_drawer(self, steps=250):
        raise NotImplementedError("open_drawer runs only in the cabinet env (train/eval); not in the table demo")

    # ---- learned policy ----------------------------------------------------------------------
    def _obs(self, prop, goal_b):
        cmd = self.u.command_manager.get_term("object_pose")
        cmd.pose_command_b[:, :3] = torch.as_tensor(goal_b, device=self.u.device, dtype=torch.float32)
        cmd.pose_command_b[:, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.u.device)   # xyzw identity
        return torch.cat([joint_pos_rel(self.u), joint_vel_rel(self.u), self.obj_pos_b(prop),
                          cmd.pose_command_b, last_action(self.u)], dim=-1)

    @torch.no_grad()
    def _run_policy(self, skill, prop, goal, steps):
        for _ in range(steps):
            a, _ = self.policy.act(self._obs(prop, goal), skill, deterministic=True)
            self.env.step(a)
            self.t += 1
            if self.on_step:
                self.on_step(Frame(self.t, self.ee_pos_b()[0].tolist(), self.obj_pos_b(prop)[0].tolist(), f"{skill} (policy)"))
