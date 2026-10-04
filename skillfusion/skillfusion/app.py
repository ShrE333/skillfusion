"""SkillFusion demo: voice/text command -> Nemotron agent -> Isaac Lab scene + skills, viewed in viser.

    python -m skillfusion.app --viz viser --device cuda:0 [--ckpt runs/fused.npz] [--asr]

Viser ports: 8080 = 3D scene (Isaac Lab visualizer), 8081 = command/camera panel, 8082 = push-to-talk page.
"""
from __future__ import annotations

import argparse
import time
import traceback

from .agent import codegen
from .agent.llm import NemotronClient
from .agent.planner import Planner
from .catalog import Catalog, CatalogError
from .scene import place, suggest_relations

BLOCKS_AT_A = {"red_block": (0.50, 0.20), "blue_block": (0.42, 0.26), "green_block": (0.58, 0.14)}


def default_program(action: dict) -> str:
    objs = action.get("objects") or list(BLOCKS_AT_A)
    goal = action.get("to") or "B"
    lines = [f"for o in {objs!r}:", f"    robot.pick_place(o, {goal!r})", "robot.reach('home')"]
    if action["skill"] == "push":
        lines = [f"for o in {objs!r}:", f"    robot.push(o, {goal!r})", "robot.reach('home')"]
    elif action["skill"] == "lift":
        lines = [f"robot.lift({objs[0]!r})"]
    return "\n".join(lines) + "\n"


class Demo:
    def __init__(self, args, panel, planner: Planner, cat: Catalog, docs=None):
        self.args, self.panel, self.planner, self.cat, self.docs = args, panel, planner, cat, docs
        self.env = self.runner = self.cfg = None
        self.state: dict = {"spawned": False}
        self.last_task: dict | None = None
        self.policy = None

    # -------------------------------------------------------------------------- helpers ----------
    def say(self, msg: str) -> None:
        print(msg, flush=True)
        self.panel.log(msg)

    def _push_camera(self, frame=None) -> None:
        try:
            img = self.runner.camera_frame()
            if img is not None:
                self.panel.camera(img)
        except Exception:      # noqa: BLE001 - camera is best effort
            pass

    def _load_policy(self):
        if not self.args.ckpt:
            return None
        import numpy as np
        from .fl.model import FusedPolicy
        z = np.load(self.args.ckpt)
        skills = {k.split(".")[1]: None for k in z.files if k.startswith("head.") and k.endswith("mu.weight")}
        spec = {}
        for s in skills:
            spec[s] = dict(index=self.cat.skill_index(s), obs_dim=z[f"adapter.{s}.0.weight"].shape[1],
                           act_dim=z[f"head.{s}.mu.weight"].shape[0])
        pol = FusedPolicy(spec, self.cat.max_skills).to(self.args.device)
        pol.load_numpy({k: z[k] for k in z.files}, strict=False)
        self.say(f"loaded fused policy with skills: {sorted(spec)}")
        return pol.eval()

    # -------------------------------------------------------------------------- handlers ---------
    def handle(self, text: str) -> None:
        self.say(f"**you:** {text}")
        try:
            act = self.planner.parse(text, self.state)
        except CatalogError as e:
            return self.say(f"I didn't get that: {e}")
        self.say(f"agent ({act['source']}): `{ {k: v for k, v in act.items() if k != 'source'} }`")
        getattr(self, f"do_{act['action']}", self.do_unsupported)(act)

    def do_unsupported(self, act):
        self.say(f"'{act['action']}' is not available in the live demo; use scripts/train_local.py or the Nebius jobs.")

    def do_status(self, act):
        self.say(f"state: {self.state}")

    def do_spawn(self, act):
        if act["robot_status"] != "ready":
            return self.say(f"{act['robot']} needs the Isaac Lab 3.0 port first (docs/PORTING_TROSSEN.md); "
                            "say 'spawn the franka ...' for now.")
        p = place(self.cat, act["env"], act["anchor"], act["relation"])
        if not p.ok:
            alts = suggest_relations(self.cat, act["env"], act["anchor"])
            return self.say(f"'{act['relation']}' {act['anchor']} collides with {p.conflicts}. Try: {alts}")
        if self.env is not None:
            self.env.close()
        import gymnasium as gym
        from .sim import envs
        from .sim.skills import SkillRunner
        task, cfg = envs.demo_cfg(self.cat, p, BLOCKS_AT_A)
        self.cfg = cfg
        cfg.sim.device = self.args.device
        self.env = gym.make(task, cfg=cfg)
        self.policy = self._load_policy()
        self.runner = SkillRunner(self.env, cfg.skillfusion_entities, cfg.skillfusion_scene, policy=self.policy,
                                  on_step=lambda f: (f.step % 6 == 0) and self._push_camera())
        self.state = {"spawned": True, "robot": act["robot"], "env": act["env"], "anchor": act["anchor"],
                      "relation": act["relation"]}
        self.say(f"spawned {act['robot']} in {act['env']} {act['relation'].replace('_', ' ')} the {act['anchor']}")
        self.panel.status(f"{act['robot']} @ {act['env']}")

    def do_task(self, act):
        if not self.state["spawned"]:
            return self.say("spawn a robot first, e.g. 'spawn the franka in the warehouse beside the shelf'.")
        from .sim.api import RobotAPI, SceneAPI
        self.last_task = act
        prog = None
        if self.planner.llm is not None and self.planner.llm.available:
            try:
                prog = codegen.generate(self.planner.llm, f"{act}", self.docs)
                self.say("Nemotron wrote the program:\n```python\n" + prog + "```")
            except Exception as e:     # noqa: BLE001
                self.say(f"code generation failed ({e}); using the built-in program")
        prog = prog or default_program(act)
        api = RobotAPI(self.cat, self.runner, log=self.say)
        self.panel.status("running task")
        codegen.run_code(prog, {"robot": api, "scene": SceneAPI(self.runner), "log": self.say})
        self.panel.status("task done")
        self.say("done. Say 'divide it into steps' to decompose.")

    def do_decompose(self, act):
        if not self.last_task:
            return self.say("nothing to decompose yet - run a task first.")
        steps = self.planner.decompose(self.last_task)
        for i, s in enumerate(steps, 1):
            idx = self.cat.skill_index(s["skill"])
            self.say(f"step {i}: **{s['skill']}** (skill #{idx}) {s['object'] or ''} -> {s['goal'] or ''} - {s['why']}")
        self.state["steps"] = steps

    def do_fuse(self, act):
        import subprocess, sys
        self.say("federated fusion: aggregating skill policies (see runs/fl)...")
        r = subprocess.run([sys.executable, "-m", "skillfusion.fl.fuse_cli", "--out", "runs/fused.npz"],
                           capture_output=True, text=True)
        self.say(r.stdout.strip() or r.stderr.strip()[-400:])
        if r.returncode == 0 and self.runner is not None:
            self.args.ckpt = "runs/fused.npz"
            self.runner.policy = self.policy = self._load_policy()

    def do_reset(self, act):
        if self.env is not None:
            self.env.reset()
            self.say("scene reset")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="fused policy (.npz) to drive skills instead of the scripted controller")
    ap.add_argument("--docs", default=None, help="directory of Isaac Sim 6.0.1 / Isaac Lab 3.0 docs for code-gen RAG")
    ap.add_argument("--asr", action="store_true", help="enable the NeMo ASR push-to-talk page")
    ap.add_argument("--panel-port", type=int, default=8081)
    ap.add_argument("--voice-port", type=int, default=8082)
    return ap


def main():
    ap = build_parser()
    from isaaclab_tasks.utils import add_launcher_args
    add_launcher_args(ap)
    args = ap.parse_args()
    args.device = getattr(args, "device", None) or "cuda:0"

    cat = Catalog.load()
    llm = NemotronClient()
    planner = Planner(cat, llm)
    docs = None
    if args.docs:
        docs = codegen.DocIndex()
        print(f"indexed {docs.add_dir(args.docs)} doc files")

    from .sim.panel import Panel
    panel = Panel(port=args.panel_port)
    print(f"[skillfusion] Nemotron via Token Factory: {'ON' if llm.available else 'OFF (rules fallback; set NEBIUS_API_KEY)'}")

    if args.asr:
        from .voice.asr import ASR
        from .voice import web
        asr = ASR(device=args.device)
        web.serve_in_thread(web.make_app(asr.transcribe_pcm16, panel.commands.put), port=args.voice_port)
        print(f"[skillfusion] push-to-talk page on :{args.voice_port} (port-forward it, mic needs localhost)")

    demo = Demo(args, panel, planner, cat, docs)
    panel.log("Say or type: *spawn the franka in the warehouse beside the shelf*")

    # Nothing needs the simulator until the first spawn; then the launcher is entered lazily.
    from isaaclab_tasks.utils import launch_simulation
    import isaaclab_tasks  # noqa: F401  (registers task ids)
    from .sim import envs
    first = None
    while first is None:
        cmd = panel.next_command(0.2)
        if cmd:
            try:
                a = planner.parse(cmd, demo.state)
            except CatalogError as e:
                panel.log(f"didn't get that: {e}"); continue
            if a["action"] == "spawn" and a["robot_status"] == "ready":
                first = cmd
            else:
                panel.log("start with a spawn command, e.g. *spawn the franka in the kitchen beside the counter*")
    a = planner.parse(first)
    p = place(cat, a["env"], a["anchor"], a["relation"])
    _, cfg0 = envs.demo_cfg(cat, p, BLOCKS_AT_A)
    with launch_simulation(cfg0, args):
        demo.handle(first)
        try:
            while True:
                cmd = panel.next_command(0.1)
                if cmd:
                    try:
                        demo.handle(cmd)
                    except Exception:      # noqa: BLE001 - keep the demo alive
                        demo.say("error:\n```\n" + traceback.format_exc()[-600:] + "\n```")
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
