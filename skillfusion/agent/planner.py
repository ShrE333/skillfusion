"""Voice text -> validated action, and task -> skill steps.

The LLM proposes; ``validate_action`` disposes. Nothing reaches the simulator unless every name
resolves through the catalog. A keyword parser gives the same schema when the LLM is unreachable,
so the demo never dies on a network blip.
"""
from __future__ import annotations

import re
from typing import Any

from ..catalog import Catalog, CatalogError
from ..scene import normalize_relation
from .llm import NemotronClient, LLMError

ACTIONS = ("spawn", "task", "decompose", "train", "fuse", "reset", "status")

SYSTEM = """You are the command parser of a robot demo. Convert the user's spoken command into ONE JSON object.
Allowed actions and fields:
 {"action":"spawn","robot":R,"env":E,"anchor":O,"relation":"beside|in_front_of|behind|left_of|right_of"}
 {"action":"task","skill":S,"objects":[props],"from":"A|B|C","to":"A|B|C"}
 {"action":"decompose"}            # split the last finished task into skill steps
 {"action":"train","skill":S}
 {"action":"fuse"}                 # federated fusion of all trained skills
 {"action":"reset"} | {"action":"status"}
Use ONLY names from the catalog below. If something is missing or unclear use null. JSON only.

CATALOG
"""


def build_messages(cat: Catalog, utterance: str, state: dict | None) -> list[dict]:
    ctx = f"\nCURRENT STATE: {state}" if state else ""
    return [{"role": "system", "content": SYSTEM + cat.prompt_summary() + ctx},
            {"role": "user", "content": utterance}]


def validate_action(cat: Catalog, raw: dict) -> dict:
    """Normalise names through the catalog; raise CatalogError/ValueError on anything unknown."""
    if not isinstance(raw, dict) or raw.get("action") not in ACTIONS:
        raise ValueError(f"bad action: {raw!r}")
    a = raw["action"]
    out: dict[str, Any] = {"action": a}
    if a == "spawn":
        out["robot"] = cat.robot(raw.get("robot") or "")
        out["env"] = cat.env(raw.get("env") or "")
        anchor = raw.get("anchor")
        out["anchor"] = cat.anchor(out["env"], anchor) if anchor else _default_anchor(cat, out["env"])
        out["relation"] = normalize_relation(raw.get("relation") or "beside")
        out["robot_status"] = cat.raw["robots"][out["robot"]]["status"]
    elif a == "task":
        out["skill"] = cat.skill(raw.get("skill") or "pick_place")
        objs = raw.get("objects") or []
        out["objects"] = [cat.prop(o) for o in objs] if objs else []
        out["from"] = cat.region(raw["from"]) if raw.get("from") else None
        out["to"] = cat.region(raw["to"]) if raw.get("to") else None
    elif a == "train":
        out["skill"] = cat.skill(raw.get("skill") or "")
    return out


def _default_anchor(cat: Catalog, env: str) -> str:
    return next(k for k, v in cat.raw["environments"][env]["objects"].items() if v.get("anchor"))


# ------------------------------------------------------------------ keyword fallback -----------
def rule_based(cat: Catalog, text: str) -> dict:
    t = text.lower()
    if re.search(r"\b(fuse|federat|merge)\b", t):
        return {"action": "fuse"}
    if re.search(r"\b(divide|split|break|decompose|steps?)\b", t):
        return {"action": "decompose"}
    if re.search(r"\b(train|learn)\b", t):
        for s in cat.raw["skills"]:
            if s.replace("_", " ") in t or s in t:
                return {"action": "train", "skill": s}
    robot = _find(cat.raw["robots"], t)
    env = _find(cat.raw["environments"], t)
    if robot and (env or re.search(r"\b(spawn|create|load)\b", t)):
        if env is None:
            raise CatalogError("which environment? " + ", ".join(cat.raw["environments"]))
        rel = next((r for r in ("in front of", "behind", "left of", "right of", "next to", "beside", "by")
                    if r in t), "beside")
        anchor = None
        for name, o in cat.raw["environments"][env]["objects"].items():
            if o.get("anchor") and _find({name: o}, t.split(rel, 1)[-1]):
                anchor = name
                break
        return {"action": "spawn", "robot": robot, "env": env, "anchor": anchor, "relation": rel}
    if re.search(r"\b(pick|place|move|carry|stack|lift|push|open|throw|catch|reach)\b", t):
        skill = "pick_place"
        for kw, s in (("open", "open_drawer"), ("drawer", "open_drawer"), ("lift", "lift"), ("push", "push"),
                      ("throw", "throw"), ("catch", "catch"), ("reach", "reach")):
            if kw in t and "pick" not in t:
                skill = s
                break
        m = re.search(r"from\s+(?:point\s+)?([abc])\b.*?\bto\s+(?:point\s+)?([abc])\b", t)
        objs = [p for p, spec in cat.raw["props"].items() if _find({p: spec}, t)]
        if not objs and re.search(r"\bblocks?\b|\bcubes?\b", t):
            objs = [p for p in cat.raw["props"] if p.endswith("_block")]
        return {"action": "task", "skill": skill, "objects": objs,
                "from": m.group(1) if m else "A", "to": m.group(2) if m else "B"}
    if re.search(r"\bstatus\b", t):
        return {"action": "status"}
    if re.search(r"\breset\b", t):
        return {"action": "reset"}
    raise CatalogError(f"could not understand: {text!r}")


def _find(table: dict, text: str) -> str | None:
    try:
        return Catalog._resolve(table, text, "x")
    except CatalogError:
        return None


class Planner:
    def __init__(self, cat: Catalog, llm: NemotronClient | None = None):
        self.cat, self.llm = cat, llm

    def parse(self, utterance: str, state: dict | None = None) -> dict:
        """Returns a validated action; ``source`` tells the demo UI whether Nemotron or rules produced it."""
        if self.llm is not None and self.llm.available:
            try:
                raw = self.llm.chat_json(build_messages(self.cat, utterance, state), max_tokens=512)
                act = validate_action(self.cat, raw)
                act["source"] = "nemotron"
                return act
            except (LLMError, CatalogError, ValueError, KeyError):
                pass                                           # fall through to rules
        act = validate_action(self.cat, rule_based(self.cat, utterance))
        act["source"] = "rules"
        return act

    # ------------------------------------------------------------------ decomposition -------
    def decompose(self, task: dict) -> list[dict]:
        """Split a finished task into skill steps, each a catalog skill the RL backend can run."""
        if self.llm is not None and self.llm.available:
            try:
                msgs = [{"role": "system", "content":
                         "Split the robot task into 2-8 ordered steps. Each step must use a skill from the catalog and be "
                         "executable by one RL policy. Reply JSON: {\"steps\":[{\"skill\":S,\"object\":P|null,\"goal\":A|B|C|home|null,\"why\":str}]}\n"
                         + self.cat.prompt_summary()},
                        {"role": "user", "content": f"Task: {task}"}]
                data = self.llm.chat_json(msgs, max_tokens=800)
                steps = [self._step(s) for s in data["steps"]]
                if steps:
                    return steps
            except (LLMError, CatalogError, ValueError, KeyError, TypeError):
                pass
        return self.decompose_rules(task)

    def _step(self, s: dict) -> dict:
        skill = self.cat.skill(s["skill"])
        if self.cat.raw["skills"][skill]["status"] != "ready":
            raise CatalogError(f"skill {skill} not ready")
        return {"skill": skill, "object": self.cat.prop(s["object"]) if s.get("object") else None,
                "goal": self.cat.region(s["goal"]) if s.get("goal") else None, "why": str(s.get("why", ""))}

    def decompose_rules(self, task: dict) -> list[dict]:
        goal = task.get("to") or "B"
        steps: list[dict] = []
        for obj in task.get("objects") or [None]:
            steps.append({"skill": "reach", "object": obj, "goal": task.get("from") or "A", "why": "pre-grasp above object"})
            steps.append({"skill": "pick_place", "object": obj, "goal": goal, "why": "grasp, carry, release"})
        steps.append({"skill": "reach", "object": None, "goal": "home", "why": "retreat to home pose"})
        return steps
