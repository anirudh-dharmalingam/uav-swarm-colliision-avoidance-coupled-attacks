"""Trigger functions for the attack-trigger ablation (T1..T7).

Every trigger takes the attacker's local observation (see base.local_observation)
plus bookkeeping and returns a boolean "attack now". The point of the ablation is
to show that the local collective trigger (T7) nearly matches the global-oracle
trigger (T6) while using only information a compromised endpoint can actually
observe.
"""

from __future__ import annotations

import numpy as np


def t1_fixed_time(obs, t, dt, ctx):
    """T1: fire during a fixed time window, ignoring swarm state."""
    start, end = ctx.get("window", (8.0, 1e9))
    return start <= t * dt <= end


def t2_random(obs, t, dt, ctx):
    """T2: fire at random with fixed probability."""
    return ctx["rng"].random() < ctx.get("prob", 0.3)


def t3_local_density(obs, t, dt, ctx):
    """T3: fire when local neighbour density is high."""
    return obs["density"] >= ctx.get("density_thresh", 0.275)


def t4_distance_closing(obs, t, dt, ctx):
    """T4: fire when a neighbour is close AND closing."""
    return obs["min_distance"] <= ctx.get("dist_thresh", 3.5) and obs["mean_closing_rate"] < 0.0


def t5_local_graph(obs, t, dt, ctx):
    """T5: fire on a locally observable graph feature (neighbourhood lambda_2)."""
    return obs["local_lambda2"] >= ctx.get("local_lambda_thresh", 2.0)


def t6_global_oracle(obs, t, dt, ctx):
    """T6: ORACLE. Fire on the true global lambda_2 (not realistically available;
    evaluation upper bound only)."""
    return ctx.get("global_lambda2", 0.0) >= ctx.get("global_thresh", 2.0)


def t7_collective_local(obs, t, dt, ctx):
    """T7 (proposed): locally observable collective-vulnerability trigger.

    Fire when the neighbourhood is dense, well-connected AND actively compressing
    (closing in, or tighter than the desired formation) -- the state in which
    eroding neighbour margins is cheapest. The compression clause makes it fire
    *selectively* (transients / bottlenecks), keeping the stealth cost low.
    """
    compressing = (obs["mean_closing_rate"] < ctx.get("closing_thresh", -0.15)
                   or obs["compression"] > ctx.get("compression_thresh", 0.5))
    return (
        obs["neighbor_count"] >= ctx.get("k0", 3)
        and obs["density"] >= ctx.get("density_thresh", 0.27)
        and obs["local_lambda2"] >= ctx.get("lambda_thresh", 1.0)
        and compressing
    )


def t_always(obs, t, dt, ctx):
    """Always-on (upper bound on attack intensity; not part of the ablation)."""
    return True


TRIGGERS = {
    "ALWAYS": t_always,
    "T1": t1_fixed_time,
    "T2": t2_random,
    "T3": t3_local_density,
    "T4": t4_distance_closing,
    "T5": t5_local_graph,
    "T6": t6_global_oracle,
    "T7": t7_collective_local,
}


def make_trigger(name: str):
    return TRIGGERS[name.upper()]
