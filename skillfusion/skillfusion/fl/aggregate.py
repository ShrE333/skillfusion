"""Framework-agnostic federated fusion math (numpy). Used by the NVFlare client/job for the real
run and by the unit tests / offline simulator.

Parameter naming convention (must match ``fl.model.FusedPolicy.state_dict``):
    trunk.*            shared across skills  -> federated (FedAvg / FedProx)
    adapter.<skill>.*  private input adapter -> stays with the skill's owner
    head.<skill>.*     private actor head    -> stays with the skill's owner
    value.<skill>.*    private critic head   -> stays with the skill's owner
After the last round ``fuse`` stitches the global trunk with every owner's private parts into ONE
checkpoint that can run every skill, selected by the fixed skill code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Params = dict[str, np.ndarray]


def is_shared(name: str) -> bool:
    return name.startswith("trunk.")


def split_params(params: Params) -> tuple[Params, Params]:
    shared = {k: v for k, v in params.items() if is_shared(k)}
    private = {k: v for k, v in params.items() if not is_shared(k)}
    return shared, private


def skill_code(index: int, dim: int = 32, max_skills: int = 16, seed: int = 0) -> np.ndarray:
    """Fixed (non-learned) orthonormal code per catalog skill index. Non-learned on purpose: plain
    FedAvg is then exact, no per-row masking or dilution of a shared embedding table."""
    if not 0 <= index < max_skills or dim < max_skills:
        raise ValueError("bad skill index / dim")
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return q[index].astype(np.float32)


@dataclass
class ClientUpdate:
    skill: str
    n_samples: float
    shared: Params
    private: Params = field(default_factory=dict)


def fedavg(updates: list[ClientUpdate]) -> Params:
    """Sample-weighted average of the shared parameters."""
    if not updates:
        raise ValueError("no updates")
    keys = set(updates[0].shared)
    for u in updates:
        if set(u.shared) != keys:
            raise ValueError(f"client {u.skill} sent different shared keys")
    w = np.array([u.n_samples for u in updates], dtype=np.float64)
    if (w <= 0).any():
        raise ValueError("n_samples must be > 0")
    w = w / w.sum()
    return {k: sum(wi * u.shared[k].astype(np.float64) for wi, u in zip(w, updates)).astype(np.float32)
            for k in keys}


def fuse(global_shared: Params, updates: list[ClientUpdate]) -> Params:
    """One checkpoint: global trunk + every skill's private adapter/head/value parameters."""
    out = dict(global_shared)
    for u in updates:
        for k, v in u.private.items():
            if k in out:
                raise ValueError(f"duplicate parameter {k} from {u.skill}")
            out[k] = v
    return out


def l2_drift(a: Params, b: Params) -> float:
    """Client drift monitor: how far the local trunk moved from the global one."""
    return float(np.sqrt(sum(((a[k] - b[k]) ** 2).sum() for k in a if k in b)))
