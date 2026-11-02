#!/usr/bin/env python3
"""EXPLORATORY (post-hoc, not in the registered primary family): the E1 interaction I = L_COUPLED - L_PHYS - L_REPORT + L_A0 for EVERY method,
plus the filter benefit D24R - D0R within each attack cell, bootstrap 95% CI over complete scenario blocks (10,000 resamples, seed 20261025).
    python3 scripts/coupled_safety/interaction_by_method.py --episodes episodes_e1.csv --spec specs/e1.json --out interaction_by_method.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--episodes", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
meta = pd.DataFrame([{"name": e["name"], **e["meta"]} for e in json.loads(Path(a.spec).read_text())["episodes"]])
df = pd.read_csv(a.episodes).merge(meta, on="name")
rng = np.random.default_rng(20261025)


def ci(v):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), size=(10000, len(v)))
    b = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


rows = []
P = {m: df[df.method == m].pivot_table(index="scenario", columns="cell", values="min_sep_ha") for m in ("D0R", "D24R", "D26", "D24R_NOSQ")}
for m, p in P.items():
    I = -(p.COUPLED - p.PHYS - p.REPORT + p.A0)
    e, lo, hi = ci(I)
    rows.append({"quantity": "interaction_I", "method": m, "estimate_m": e, "lo95": lo, "hi95": hi})
for c in ("A0", "PHYS", "REPORT", "COUPLED"):
    for m in ("D24R", "D26", "D24R_NOSQ"):
        e, lo, hi = ci(P[m][c] - P["D0R"][c])
        rows.append({"quantity": f"minsep_{m}_minus_D0R_in_{c}", "method": m, "estimate_m": e, "lo95": lo, "hi95": hi})
pd.DataFrame(rows).to_csv(a.out, index=False)
print(pd.DataFrame(rows).round(3).to_string())
