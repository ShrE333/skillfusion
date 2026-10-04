"""Catalog loader: the only place robots / environments / skills are defined.

The LLM may only *select* names from here. Every name coming out of the model is resolved
through the alias tables below and rejected if unknown.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "catalog.yaml"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", _norm(text))


class CatalogError(ValueError):
    pass


@dataclass
class Catalog:
    raw: dict[str, Any]

    # ---- construction / validation -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        with open(path or DEFAULT_PATH) as f:
            cat = cls(yaml.safe_load(f))
        cat.validate()
        return cat

    def validate(self) -> None:
        skills = self.raw["skills"]
        idx = [s["index"] for s in skills.values()]
        if len(set(idx)) != len(idx):
            raise CatalogError(f"duplicate skill indices: {idx}")
        if max(idx) >= self.raw["max_skills"]:
            raise CatalogError("skill index >= max_skills")
        for name, env in self.raw["environments"].items():
            if not any(o.get("anchor") for o in env["objects"].values()):
                raise CatalogError(f"environment {name} has no anchor objects")
        for r in ("A", "B", "C", "home"):
            if r not in self.raw["workcell"]["regions"]:
                raise CatalogError(f"missing region {r}")

    # ---- alias resolution ----------------------------------------------------------------
    @staticmethod
    def _resolve(table: dict[str, dict], text: str, what: str) -> str:
        q = _squash(text)
        if not q:
            raise CatalogError(f"empty {what}")
        best, best_len = None, 0
        for name, spec in table.items():
            names = [name.replace("_", " ")] + list(spec.get("aliases", []))
            for a in names:
                a = _squash(a)
                if a and (q == a or re.search(rf"\b{re.escape(a)}\b", q)) and len(a) > best_len:
                    best, best_len = name, len(a)
        if best is None:
            raise CatalogError(f"unknown {what} {text!r}; options: {sorted(table)}")
        return best

    def robot(self, text: str) -> str:
        return self._resolve(self.raw["robots"], text, "robot")

    def env(self, text: str) -> str:
        return self._resolve(self.raw["environments"], text, "environment")

    def anchor(self, env: str, text: str) -> str:
        objs = {k: v for k, v in self.raw["environments"][env]["objects"].items() if v.get("anchor")}
        return self._resolve(objs, text, f"object in {env}")

    def prop(self, text: str) -> str:
        return self._resolve(self.raw["props"], text, "prop")

    def skill(self, text: str) -> str:
        q = _squash(text).replace(" ", "_")
        if q in self.raw["skills"]:
            return q
        raise CatalogError(f"unknown skill {text!r}; options: {sorted(self.raw['skills'])}")

    def region(self, text: str) -> str:
        t = str(text).strip()
        m = re.fullmatch(r"(?:point|region|spot|zone)?\s*([abc])", t, flags=re.I)
        if m:
            return m.group(1).upper()
        if _squash(t) in ("home", "start"):
            return "home"
        raise CatalogError(f"unknown region {text!r}; options: A, B, C, home")

    # ---- accessors -----------------------------------------------------------------------
    @property
    def workcell(self) -> dict:
        return self.raw["workcell"]

    @property
    def max_skills(self) -> int:
        return int(self.raw["max_skills"])

    def skill_spec(self, name: str) -> dict:
        return self.raw["skills"][self.skill(name)]

    def skill_index(self, name: str) -> int:
        return int(self.skill_spec(name)["index"])

    def ready_skills(self) -> list[str]:
        return [k for k, v in self.raw["skills"].items() if v.get("status") == "ready"]

    def region_center(self, name: str) -> list[float]:
        return list(self.workcell["regions"][self.region(name)]["center"])

    def prompt_summary(self) -> str:
        """Compact catalog text for LLM prompts."""
        lines = ["ROBOTS: " + ", ".join(f"{k} ({v['status']})" for k, v in self.raw["robots"].items())]
        for name, env in self.raw["environments"].items():
            anchors = [k for k, v in env["objects"].items() if v.get("anchor")]
            lines.append(f"ENV {name}: anchors = {', '.join(anchors)}")
        lines.append("PROPS: " + ", ".join(self.raw["props"]))
        lines.append("REGIONS: A, B, C, home")
        for name, s in self.raw["skills"].items():
            lines.append(f"SKILL {name} [{s['status']}] params={s['params']}: {s['description']}")
        return "\n".join(lines)
