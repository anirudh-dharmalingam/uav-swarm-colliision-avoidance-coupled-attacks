#!/usr/bin/env python3
"""B-F0 fidelity gate: Layer-1 harness vs PX4 SITL on the SAME scenario specs (inside the image: pandas).

    python3 scripts/rarf_l1/fidelity.py --l1 results/rarf_l1/v3_confirm_replay/episodes.jsonl \
        --px4 output/research/rarf_plant_time_v3/confirm/episodes.csv --contrasts FF:NOM,FF:SPD,RARF:FF,RARF:NOM --out output/research/rarf_l1/fidelity_v3

Gate rule (fixed 2026-10-30, before any Layer-1 / PX4 comparison was computed):
  For every listed paired contrast X - Y (geometry-averaged min_sep_ha per scenario, then the mean over scenarios; cluster bootstrap
  over scenarios, 10,000 resamples, numpy seed 20261104), per geometry and pooled:
    AGREE      PX4 95 % CI excludes 0 and the Layer-1 estimate has the same sign with its 95 % CI excluding 0
    NULL-AGREE PX4 CI includes 0 and the Layer-1 CI includes 0
    MAGNITUDE  same sign, but the Layer-1 point estimate lies outside the PX4 CI (reported; the ordering is still usable)
    DISAGREE   anything else
  A Layer-1 sweep may state an ordering of X vs Y only for contrasts that AGREE (or MAGNITUDE) at every geometry where PX4 resolves it;
  a DISAGREE contrast makes any Layer-1 statement about that pair exploratory.  Absolute levels are never transferred: per-cell mean
  absolute error and the episode-level correlation are reported.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261104


def method_of(meta_or_name):
    return meta_or_name


def load_l1(p):
    rows = [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    E = pd.DataFrame(rows)
    E = E[E.valid.astype(str).str.lower() == "true"].copy()
    E["method"] = E.meta.map(lambda m: m.get("method"))
    E["geometry"] = E.meta.map(lambda m: float(m.get("geometry", 0.0)))
    E["seed"] = E.meta.map(lambda m: int(m.get("seed")))
    return E


def load_px4(p):
    E = pd.read_csv(p)
    E = E[E.valid.astype(str).str.lower() == "true"].copy()
    return E


def boot(diff_by_seed, rng, B=10000):
    x = np.asarray(diff_by_seed, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def contrast(E, X, Y, geo=None):
    S = E if geo is None else E[E.geometry == geo]
    a = S[S.method == X].groupby("seed").min_sep_ha.mean()
    b = S[S.method == Y].groupby("seed").min_sep_ha.mean()
    common = a.index.intersection(b.index)
    return (a[common] - b[common]).to_numpy()


def verdict(px, l1):
    p_res = not (px[1] <= 0 <= px[2])
    l_res = not (l1[1] <= 0 <= l1[2])
    if not p_res:
        return "NULL-AGREE" if not l_res else "DISAGREE"
    if np.sign(px[0]) == np.sign(l1[0]) and l_res:
        return "AGREE" if px[1] <= l1[0] <= px[2] else "MAGNITUDE"
    return "DISAGREE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", required=True)
    ap.add_argument("--px4", required=True)
    ap.add_argument("--contrasts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L, P = load_l1(a.l1), load_px4(a.px4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = L.merge(P[["name", "min_sep_ha", "viol15", "route_complete", "crash"]], on="name", suffixes=("_l1", "_px4"))
    R = {"n_l1": len(L), "n_px4": len(P), "n_matched": len(m),
         "episode_corr_min_sep_ha": float(np.corrcoef(m.min_sep_ha_l1, m.min_sep_ha_px4)[0, 1]) if len(m) > 2 else None,
         "viol15_agreement": float((m.viol15_l1 == m.viol15_px4).mean()) if len(m) else None,
         "cells": {}, "contrasts": {}}
    for (meth, g), s in m.groupby(["method", "geometry"]) if "method" in m else []:
        R["cells"][f"{meth}|g{g:g}"] = {"n": len(s), "l1_mean": float(s.min_sep_ha_l1.mean()), "px4_mean": float(s.min_sep_ha_px4.mean()),
                                         "mae": float((s.min_sep_ha_l1 - s.min_sep_ha_px4).abs().mean()),
                                         "l1_viol15": int(s.viol15_l1.sum()), "px4_viol15": int(s.viol15_px4.sum())}
    geos = sorted(set(P.geometry.dropna().astype(float)))
    for c in a.contrasts.split(","):
        X, Y = c.split(":")
        R["contrasts"][c] = {}
        for g in [None] + geos:
            rng_p, rng_l = np.random.default_rng(SEED), np.random.default_rng(SEED)
            px = boot(contrast(P, X, Y, g), rng_p)
            l1 = boot(contrast(L, X, Y, g), rng_l)
            R["contrasts"][c]["pooled" if g is None else f"g{g:g}"] = {"px4": px, "l1": l1, "verdict": verdict(px, l1)}
    (out / "fidelity.json").write_text(json.dumps(R, indent=1, default=float))
    m.to_csv(out / "matched_episodes.csv", index=False)
    print(json.dumps({k: R[k] for k in ("n_l1", "n_px4", "n_matched", "episode_corr_min_sep_ha", "viol15_agreement")}, indent=1))
    for c, d in R["contrasts"].items():
        print(c, {k: (round(v["px4"][0], 3), round(v["l1"][0], 3), v["verdict"]) for k, v in d.items()})


if __name__ == "__main__":
    main()
