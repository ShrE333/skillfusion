"""Minimal pytest-free runner: python tests/run_all.py (pytest also works)."""
import importlib, pathlib, sys, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
fails = n = 0
for f in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
    mod = importlib.import_module(f.stem)
    for name in dir(mod):
        if name.startswith("test_"):
            n += 1
            try:
                getattr(mod, name)()
            except Exception:
                fails += 1
                print(f"FAIL {f.stem}.{name}"); traceback.print_exc()
print(f"{n - fails}/{n} passed")
sys.exit(1 if fails else 0)
