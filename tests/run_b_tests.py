#!/usr/bin/env python3
"""Avoidance test suite (dependency-free runner; the research image ships no pytest).

    python3 tests/run_b_tests.py                       # with requirements-l1.txt installed
    docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace dronelab:research-v1.17 python3 tests/run_b_tests.py

Unit tests of the avoidance mathematics and comparators from the research repository.
"""
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_mb")

MODULES = ["tests.test_coupled_safety", "tests.test_hocbf_comparator", "tests.test_rarf_ablation"]


def main() -> int:
    passed = failed = 0
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            print(f"FAIL {modname}: import error ({exc})")
            failed += 1
            continue
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if not name.startswith("test_") or not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
                print(f"PASS {modname}.{name}  ({time.time() - t0:.1f} s)", flush=True)
                passed += 1
            except Exception:
                print(f"FAIL {modname}.{name}", flush=True)
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
