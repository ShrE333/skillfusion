# Setup (Isaac Sim 6.0.1 + Isaac Lab 3.0 on the shared A100 container)

Assumptions from the brief: A100 (no RT cores), 10 GB VRAM, only viser can be viewed.

1. `pip install -e .[fl]` inside the Isaac Lab python environment (`./isaaclab.sh -p -m pip install -e .`).
2. `export NEBIUS_API_KEY=...` (Nebius Token Factory; models: Nemotron 3 Super for agent + code-gen, Nano as fast fallback,
   override with `NEMOTRON_MODEL`).
3. `python scripts/list_tasks.py` - verifies the task ids in `configs/catalog.yaml` exist in your install.
4. `python scripts/inspect_robot.py --robot franka --viz none` - confirm `panda_hand` and tune the wrist-camera
   offset (`configs/catalog.yaml`) while watching the panel.
5. `scripts/fetch_docs.sh docs/isaac` - doc corpus for code-gen RAG.
6. Ports to forward: **8080** viser 3D scene, **8081** command/camera panel, **8082** push-to-talk page
   (browsers only allow the microphone on `localhost`/HTTPS, so use `ssh -L`).
7. `scripts/run_demo.sh`.

## Why these choices on this hardware
* **PhysX + software wrist camera.** Isaac Lab 3.0's Franka lift task is PhysX-only; the RTX camera renderer needs RT cores
  (A100 has none) and the kit-less Newton-Warp renderer needs the Newton backend. All our scene objects are boxes, so
  `sim/softcam.py` ray-casts the wrist view directly. If you later train on an RTX GPU or a Newton-enabled task,
  `envs.demo_cfg(..., camera="sensor")` uses a real `CameraCfg`.
* **State-based RL, 128-256 envs**, no cameras in training -> a few GB of VRAM. The LLM is on Token Factory, not the GPU.
* **One skill = one federated client**, run sequentially or as separate Nebius jobs; only the trunk is exchanged.

## Verify on the real install (not testable in the dev sandbox)
* `--viz viser` combined with the Kit/PhysX launch on a headless A100.
* Sign/scale of the IK-Rel gripper command (positive = open) and `ACTION_SCALE` in `sim/skills.py`.
* Camera offset; NVFlare recipe import paths (`fl/job.py` marks the two lines).
* Model ids on Token Factory (`agent/llm.py`), ASR model ids (`voice/asr.py`).
