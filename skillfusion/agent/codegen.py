"""Doc-grounded code generation with a hard safety gate.

Nemotron is given (a) the retrieved Isaac Sim 6.0.1 / Isaac Lab 3.0 doc chunks and (b) a *tiny
skill API* (``skillfusion.sim.api``). It writes a short Python program that calls that API. The
program is AST-checked (whitelisted imports/calls, no dunder access) before it is run.
"""
from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path

from .llm import NemotronClient

API_DOC = '''
Available in the program namespace (do NOT import anything except math, numpy):
  robot.reach(goal: str|list[float], steps:int=120)          # goal: "A","B","C","home" or [x,y,z] in robot base frame
  robot.pick_place(obj: str, goal: str, steps:int=250)       # grasp prop `obj` and put it on region `goal`
  robot.open_drawer(steps:int=250)
  robot.push(obj: str, goal: str, steps:int=200)
  robot.lift(obj: str, steps:int=150)
  robot.gripper(open: bool)
  robot.camera_rgb() -> ndarray[H,W,3]                       # wrist camera frame
  scene.position(obj: str) -> list[float]                    # [x,y,z] in robot base frame
  scene.props() -> list[str]
  log(msg: str)
'''

ALLOWED_IMPORTS = {"math", "numpy"}
ALLOWED_NAMES = {"robot", "scene", "log", "range", "len", "enumerate", "zip", "min", "max", "abs", "round",
                 "float", "int", "str", "list", "dict", "sorted", "print", "True", "False", "None", "math", "np",
                 "numpy", "isinstance", "sum", "tuple"}
BANNED_ATTR_PREFIX = "__"


class UnsafeCode(ValueError):
    pass


def check_code(src: str) -> ast.AST:
    tree = ast.parse(src)
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    raise UnsafeCode(f"import of {m!r} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith(BANNED_ATTR_PREFIX):
            raise UnsafeCode(f"dunder attribute {node.attr!r}")
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.Try, ast.With)):
            raise UnsafeCode(f"{type(node).__name__} not allowed")
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            defined.update(a.arg for a in node.args.args)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.alias):
            defined.add(node.asname or node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in ALLOWED_NAMES and node.id not in defined:
                raise UnsafeCode(f"name {node.id!r} not allowed")
    return tree


def run_code(src: str, namespace: dict) -> None:
    tree = check_code(src)
    import numpy as np  # noqa
    ns = {"math": math, "np": np, "numpy": np, **namespace}
    exec(compile(tree, "<agent-code>", "exec"), {"__builtins__": {
        "range": range, "len": len, "enumerate": enumerate, "zip": zip, "min": min, "max": max, "abs": abs,
        "round": round, "float": float, "int": int, "str": str, "list": list, "dict": dict, "sorted": sorted,
        "print": print, "isinstance": isinstance, "sum": sum, "tuple": tuple, "__import__": _safe_import}}, ns)


def _safe_import(name, *a, **k):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(name)
    return __import__(name, *a, **k)


# ------------------------------------------------------------------ retrieval -------------------
def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]{2,}", s.lower())


class DocIndex:
    """Small BM25 over markdown/rst chunks. Point it at Isaac Sim 6.0.1 + Isaac Lab 3.0 docs."""

    def __init__(self, chunk_chars: int = 1400):
        self.chunks: list[tuple[str, str]] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.chunk_chars = chunk_chars

    def add_dir(self, root: str | Path, exts=(".md", ".rst", ".txt")) -> int:
        n = 0
        for p in Path(root).rglob("*"):
            if p.suffix in exts and p.is_file():
                self.add_text(str(p), p.read_text(errors="ignore"))
                n += 1
        return n

    def add_text(self, source: str, text: str) -> None:
        for i in range(0, len(text), self.chunk_chars):
            chunk = text[i:i + self.chunk_chars]
            tf = Counter(_tok(chunk))
            self.chunks.append((source, chunk))
            self.tf.append(tf)
            self.df.update(tf.keys())

    def search(self, query: str, k: int = 4) -> list[tuple[str, str]]:
        N = max(len(self.chunks), 1)
        avg = sum(sum(t.values()) for t in self.tf) / N or 1
        q = _tok(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = sum(tf.values())
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * dl / avg))
            if s:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


CODEGEN_SYSTEM = """You write short Python programs that drive a simulated robot through a fixed skill API.
Rules: use only the API below; no imports except math/numpy; no file/network/system access; no classes.
Output ONLY one ```python block.
""" + API_DOC


def generate(llm: NemotronClient, task: str, docs: DocIndex | None = None, steps: list[dict] | None = None) -> str:
    ctx = ""
    if docs is not None:
        hits = docs.search(task, k=3)
        ctx = "\nRELEVANT ISAAC DOCS (for background, the API above is authoritative):\n" + "\n---\n".join(
            f"[{s}]\n{c}" for s, c in hits)
    user = f"Task: {task}\nSkill steps (if any): {steps}"
    text = llm.chat([{"role": "system", "content": CODEGEN_SYSTEM + ctx}, {"role": "user", "content": user}],
                    max_tokens=1500)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    code = (m.group(1) if m else text).strip()
    check_code(code)                                          # raises UnsafeCode -> caller may retry
    return code
