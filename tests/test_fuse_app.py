import os, tempfile, subprocess, sys
import numpy as np
from skillfusion.agent import codegen
from skillfusion.app import default_program

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_default_programs_pass_gate():
    for skill in ("pick_place", "push", "lift"):
        codegen.check_code(default_program({"skill": skill, "objects": ["red_block", "blue_block"], "to": "B"}))

def test_fuse_cli_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        fl = os.path.join(d, "fl"); os.makedirs(fl)
        for s, v in (("pick_place", 1.0), ("lift", 3.0)):
            np.savez(os.path.join(fl, f"trunk_{s}.npz"), **{"net.0.weight": np.full((2, 2), v, np.float32)})
            np.savez(os.path.join(fl, f"private_{s}.npz"), **{f"head.{s}.mu.weight": np.full((1,), v, np.float32)})
        out = os.path.join(d, "fused.npz")
        r = subprocess.run([sys.executable, "-m", "skillfusion.fl.fuse_cli", "--dir", fl, "--out", out],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        z = np.load(out)
        assert set(z.files) == {"trunk.net.0.weight", "head.pick_place.mu.weight", "head.lift.mu.weight"}
        assert np.allclose(z["trunk.net.0.weight"], 2.0)            # FedAvg of 1 and 3
