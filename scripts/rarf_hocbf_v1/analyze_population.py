#!/usr/bin/env python3
"""Primary-population wrapper for the rarf_hocbf_v1 confirmatory analysis (DEVIATIONS.md, correction of 2026-10-30 23:30 IST).

Frozen rule (rarf_plant_time_v3 FROZEN_PLAN §5/§8): a completed episode with failed instrument criteria is kept and flagged, never
re-flown.  For every scenario-arm cell whose FIRST completed attempt lies in confirm/_interrupted_attempts/power_saver_20261030/ (it
has eval/summary.json), that attempt is analysed in place of the later re-flight.  Everything else is the normal confirm directory.
The unmodified scripts/rarf_hocbf_v1/analyze.py is run on a symlink view, with its output redirected to
output/research/rarf_hocbf_v1/confirm_primary_population/.  The plain `analyze.py confirm` output (re-flights) is the sensitivity.

    python3 scripts/rarf_hocbf_v1/analyze_population.py            (inside the image)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "results/rarf_hocbf_v1/confirm"
FIRST = CONF / "_interrupted_attempts/power_saver_20261030"
OUT = ROOT / "output/research/rarf_hocbf_v1/confirm_primary_population"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="hocbf_pop_"))
    view = tmp / "results/rarf_hocbf_v1/confirm"
    view.mkdir(parents=True)
    first = {p.name: p for p in FIRST.iterdir() if p.is_dir() and (p / "eval" / "summary.json").exists()}
    used = {}
    for p in sorted(CONF.iterdir()):
        if p.is_dir() and not p.name.startswith("_"):
            src = first.get(p.name, p)
            os.symlink(src, view / p.name)
            used[p.name] = "first_attempt_power_saver" if p.name in first else "confirm"
    for nm, p in first.items():                       # a first attempt whose re-flight has not happened yet
        if nm not in used:
            os.symlink(p, view / nm)
            used[nm] = "first_attempt_power_saver"
    (tmp / "experiments/rarf_hocbf_v1/specs").mkdir(parents=True)
    os.symlink(ROOT / "experiments/rarf_hocbf_v1/specs/confirm.json", tmp / "experiments/rarf_hocbf_v1/specs/confirm.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (tmp / "output/research/rarf_hocbf_v1").mkdir(parents=True)
    os.symlink(OUT, tmp / "output/research/rarf_hocbf_v1/confirm")
    (OUT / "population.json").write_text(json.dumps({"rule": "DEVIATIONS.md 2026-10-30 23:30 correction", "sources": used}, indent=1))
    sys.path.insert(0, str(ROOT / "scripts/rarf_hocbf_v1"))
    import analyze
    analyze.ROOT = tmp
    # The frozen v3 binaries() hard-codes the pairs FF-NOM, FF-SPD, RARF-FF; this population has no SPD arm (PLAN.md amendment A2),
    # so the unchanged function raised KeyError('SPD') after the primary contrasts.  Same code, run with absent arms masked out
    # (deviation recorded in DEVIATIONS.md 2026-10-31).
    _b = analyze.A.binaries.__code__
    src = Path(analyze.A.__file__).read_text()
    src = src.replace('for a, b in (("FF", "NOM"), ("FF", "SPD"), ("RARF", "FF")):',
                      'for a, b in [p for p in (("FF", "NOM"), ("FF", "SPD"), ("RARF", "FF")) if p[0] in ai and p[1] in ai]:')
    ns = vars(analyze.A)
    exec(compile(src[src.index("def binaries(df):"):src.index("def main():")], analyze.A.__file__, "exec"), ns)
    analyze.run("confirm")


if __name__ == "__main__":
    main()
