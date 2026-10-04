"""Final fusion: global trunk (federated) + every skill's private adapter/head/value -> ONE checkpoint.

Pure numpy - runs anywhere. Trunk source: ``--trunk`` (.npz with keys 'net.0.weight'... or 'trunk.net...').
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .aggregate import ClientUpdate, fuse


def load_trunk(path: str) -> dict:
    if path.endswith(".npz"):
        z = np.load(path)
        return {(k if k.startswith("trunk.") else f"trunk.{k}"): z[k] for k in z.files}
    import torch                                            # .pt from NVFlare's global model
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    return {(k if k.startswith("trunk.") else f"trunk.{k}"): v.numpy() for k, v in sd.items()}


def find_trunk(d: str, explicit: str | None) -> dict:
    """Explicit path > <dir>/global_trunk.npz > FedAvg of the clients' last local trunks (== last round)."""
    from .aggregate import fedavg
    if explicit:
        return load_trunk(explicit)
    g = os.path.join(d, "global_trunk.npz")
    if os.path.exists(g):
        return load_trunk(g)
    files = sorted(glob.glob(os.path.join(d, "trunk_*.npz")))
    if not files:
        raise SystemExit(f"no trunk found in {d}: pass --trunk (NVFlare global model) or run the clients first")
    ups = [ClientUpdate(os.path.basename(f), 1.0, load_trunk(f)) for f in files]
    return fedavg(ups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/fl")
    ap.add_argument("--trunk", default=None, help="global trunk; default <dir>/global_trunk.npz")
    ap.add_argument("--out", default="runs/fused.npz")
    a = ap.parse_args()
    trunk = find_trunk(a.dir, a.trunk)
    ups = []
    for f in sorted(glob.glob(os.path.join(a.dir, "private_*.npz"))):
        skill = os.path.basename(f)[len("private_"):-len(".npz")]
        z = np.load(f)
        ups.append(ClientUpdate(skill, 1.0, {}, {k: z[k] for k in z.files}))
    if not ups:
        raise SystemExit(f"no private_*.npz in {a.dir}")
    fused = fuse(trunk, ups)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **fused)
    print(f"fused {len(ups)} skills ({', '.join(u.skill for u in ups)}) + shared trunk -> {a.out} ({len(fused)} tensors)")


if __name__ == "__main__":
    main()
