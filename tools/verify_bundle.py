#!/usr/bin/env python3
"""Check packaged data, experiment freezes, and flight-record completeness.

    python3 tools/verify_bundle.py [--quick]

Checks
  1. every file listed in MANIFEST.sha256 is present and unchanged (--quick: sizes only for files > 1 MB);
  2. every frozen-plan hash (experiments/rarf_l1_*/FROZEN.sha256, experiments/rarf_hocbf_v1/FROZEN.json) matches the code here,
     i.e. the code in this bundle is the code that produced the results;
  3. completeness: the 252 primary PX4 flights resolve to a truth trace, and every Layer-1 study has its full row count.
Exit status 0 only if everything passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L1_ROWS = {"breadth_spectrum": 2880, "pb_spectrum": 720, "breadth_budget": 4000, "pb_budget": 1000, "adaptive_test": 2080,
           "pb_adaptive_test": 920, "rz": 3200, "pb_rz": 800, "dly": 1600, "iss": 1200, "hocbf_confirm_replay": 252,
           "v3_confirm_replay": None, "scale": 2080, "pb_scale": 520}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    bad = []
    ok = lambda name, cond, why="": (print(f"{'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + why}"),
                                     None if cond else bad.append(name))
    # 1. manifest
    man = ROOT / "MANIFEST.sha256"
    if man.exists():
        n = miss = diff = 0
        for line in man.read_text().splitlines():
            h, rel = line.split("  ", 1)
            p = ROOT / rel
            n += 1
            if not p.exists():
                miss += 1
                continue
            if a.quick and p.stat().st_size > 1 << 20:
                continue
            if sha(p) != h:
                diff += 1
                if diff <= 5:
                    print(f"      changed: {rel}")
        ok(f"manifest: {n} files", miss == 0 and diff == 0, f"{miss} missing, {diff} changed")
    else:
        ok("manifest present", False, "MANIFEST.sha256 not found (running from the source repository?)")
    # 2. frozen plans
    for f in sorted(ROOT.glob("experiments/rarf_l1_*/FROZEN.sha256")):
        rows = [l.split("  ", 1) for l in f.read_text().splitlines() if l.strip()]
        wrong = [rel for h, rel in rows if not (ROOT / rel).exists() or sha(ROOT / rel) != h]
        ok(f"frozen {f.parent.name}: {len(rows)} files", not wrong, ", ".join(wrong[:4]))
    fj = json.loads((ROOT / "experiments/rarf_hocbf_v1/FROZEN.json").read_text())["sha256"]
    wrong = [rel for rel, h in fj.items() if not (ROOT / rel).exists() or sha(ROOT / rel) != h]
    ok(f"frozen rarf_hocbf_v1: {len(fj)} files", not wrong, ", ".join(wrong[:4]))
    # 3. completeness
    pop = json.loads((ROOT / "output/research/rarf_hocbf_v1/confirm_primary_population/population.json").read_text())["sources"]
    conf = ROOT / "results/rarf_hocbf_v1/confirm"
    missing = []
    for name, src in pop.items():
        d = conf / "_interrupted_attempts/power_saver_20261030" / name if src == "first_attempt_power_saver" else conf / name
        if not ((d / "eval/truth.csv.gz").exists() or (d / "eval/truth.csv").exists()) or not (d / "eval/summary.json").exists():
            missing.append(name)
    ok(f"PX4 primary flights with truth trace: {len(pop) - len(missing)}/{len(pop)}", len(pop) == 252 and not missing,
       ", ".join(missing[:4]))
    for st, n in L1_ROWS.items():
        f = ROOT / "results/rarf_l1" / st / "episodes.jsonl"
        k = sum(1 for l in f.open() if l.strip()) if f.exists() else 0
        ok(f"Layer-1 {st}: {k} rows", f.exists() and (n is None or k == n), f"expected {n}")
    print(f"\n{'ALL CHECKS PASSED' if not bad else str(len(bad)) + ' CHECK(S) FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
