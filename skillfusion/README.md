# SkillFusion

**Talk to a robot, watch an AI agent write and run its skills, then fuse the skills with federated learning.**
Built for the Nebius x NVIDIA Global AI Hackathon (Physical AI).

```
 voice ─► NeMo ASR ─┐                        ┌─► Isaac Lab 3.0 scene (viser :8080)
 (browser mic)      ├─► Nemotron agent ──────┤   Franka + warehouse/kitchen/lab, wrist camera panel (:8081)
 typed command ─────┘   (Token Factory)      └─► skills: reach · pick_place · lift · push · open_drawer
                         │  ▲                              ▲
        catalog-checked  │  │ Isaac docs (RAG)             │ per-skill PPO (state obs)
        actions + code   ▼  │                              │
                    plan / code-gen / decompose      NVFlare: FedAvg over a shared trunk,
                                                    private adapter+head per skill ─► ONE fused policy
```

## What it does
1. **"Spawn the franka in the warehouse and place it beside the shelf."** Nemotron turns speech into a JSON action;
   every name is validated against `configs/catalog.yaml` (it can *select*, never invent). `scene.py` computes a
   collision-free placement (beside / in front of / behind / left / right) and the scene is built around the fixed workcell,
   so a policy trained once works in every environment.
2. **"Pick and place the blocks from A to B."** Nemotron, grounded on Isaac Sim/Lab docs, writes a short program against a
   tiny skill API; an AST gate (imports, names, dunders) runs before anything executes. The wrist-camera view streams to the panel.
3. **"Divide it into steps."** The agent decomposes the task into catalog skills; each step is routed to a backend RL skill policy.
4. **"Fuse the skills."** Each skill is a federated client (PPO locally). NVFlare FedAvg fuses the shared trunk; skill-specific
   adapters/heads stay with their owner; `fl/fuse_cli.py` stitches everything into a single checkpoint that runs every skill.

## Hackathon requirements
| Requirement | Where |
|---|---|
| NVIDIA open model | Nemotron 3 (agent, code-gen) via `agent/llm.py`; NeMo ASR (`voice/asr.py`) |
| Nebius | Token Factory for the LLM; Serverless AI Jobs for federated clients (`jobs/nebius_fl_client.sh`) |
| Physical AI | Isaac Sim 6.0.1 / Isaac Lab 3.0 manipulation, RL skills, federated fusion |

## Quick start
See [docs/SETUP.md](docs/SETUP.md). Short version: `scripts/run_demo.sh`, open viser on :8080 and the panel on :8081, type
or speak *"spawn the franka in the warehouse beside the shelf"*. Without `NEBIUS_API_KEY` a keyword parser keeps the demo alive
(the panel shows whether Nemotron or the fallback produced each action).
Train + fuse: `scripts/train_all.sh`.

## Status (honest)
| Part | State |
|---|---|
| catalog, placement, planner, code gate, RAG, FL math, fuse CLI, software camera, voice web app | unit-tested (31 tests: `python tests/run_all.py`) |
| PPO, FusedPolicy, NVFlare client/job, Isaac env builders, skill runner, viser panel | written against the Isaac Lab 3.0 source, **not run** (no GPU/Isaac in the dev sandbox) - see "Verify" in docs/SETUP.md |
| Trossen WidowX AI | needs Isaac Lab 3.0 port - [docs/PORTING_TROSSEN.md](docs/PORTING_TROSSEN.md) |
| throw / catch skills | stretch: catalog stubs, code-gen path only |
| open_drawer | trains in the cabinet env; not part of the table demo scene |

## Layout
`configs/catalog.yaml` single source of truth · `skillfusion/agent` LLM, planner, code-gen · `skillfusion/scene.py` placement ·
`skillfusion/sim` envs, skill runner, camera, panel · `skillfusion/rl` PPO · `skillfusion/fl` model, aggregation, NVFlare · `tests/`

## License
MIT - see [LICENSE](LICENSE).
