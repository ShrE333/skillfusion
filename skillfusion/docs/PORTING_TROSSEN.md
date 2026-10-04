# Adding the Trossen WidowX AI (Isaac Lab 3.0 port)

`TrossenRobotics/trossen_ai_isaac` targets Isaac Lab 2.3. The catalog entry `wxai` is marked
`needs_port` and the demo refuses to spawn it until this is done. What changes in 3.0:

| 2.x | 3.0 |
|---|---|
| quaternions `(w, x, y, z)` | `(x, y, z, w)` - grep the asset cfg for `rot=` |
| `robot.data.joint_pos` (tensor) | `robot.data.joint_pos.torch` (ProxyArray) |
| `write_root_pose_to_sim(...)` | `write_root_pose_to_sim_index(...)` / `_mask` |
| `--headless` | `--viz none` / `--viz viser` |

Steps:
1. `pip install` the package into the Isaac Lab env; run `scripts/inspect_robot.py`-style inspection on `WXAI_CFG`
   to get real body names, then fix `ee_body` / `wrist_camera.mount_body` in `configs/catalog.yaml`.
2. Fix quaternion order in the asset cfg (init_state rot, camera offsets).
3. Register lift / reach / drawer tasks for it (copy `isaaclab_tasks/.../lift/config/franka/` and swap the
   robot, `body_name`, gripper joint names/open-close values, IK `body_offset`). Register ids
   `Isaac-Lift-Cube-WXAI-v0` etc. (the ids already listed in the catalog).
4. `python scripts/list_tasks.py` -> the wxai rows turn `OK`; set `status: ready` for `wxai`.
The 6-DoF arm has no `panda_joint.*` naming: change `joint_names` regexes in the copied cfgs.
