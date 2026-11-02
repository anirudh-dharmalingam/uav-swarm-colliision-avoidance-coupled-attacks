"""Attacker models for the compromised-endpoint threat model.

Threat model (frozen): exactly one legitimate UAV is compromised *after*
deployment. It knows its own state, the states of neighbours it can hear, the
distributed control law, and runs a lightweight local policy. It may perturb its
own bounded actuation and falsify its own outgoing state messages. It cannot
touch other UAVs' firmware/weights, read their memory, decrypt traffic, inject
packets as another identity, or exceed its actuator limits.

Attack families (A0..A8) and the trigger ablation (T1..T7) live here.
"""

from .base import Attack, AttackOutput, NoAttack, local_observation
from .baselines import (
    RandomPerturbation,
    ConstantBias,
    PeriodicTrigger,
    FalseDataInjection,
    ReplayDelay,
    FixedStateTrigger,
)
from .collective import CollectiveMarginAttack, CompressionAttack, StealthRampAttack
from .learned import LearnedAttack
from .adaptive import WhiteBoxEvader, InnovationEvader
from .triggers import TRIGGERS, make_trigger
from .randomized import RandomizedAttack, sample_params, perturb_params

__all__ = [
    "Attack",
    "AttackOutput",
    "NoAttack",
    "local_observation",
    "RandomPerturbation",
    "ConstantBias",
    "PeriodicTrigger",
    "FalseDataInjection",
    "ReplayDelay",
    "FixedStateTrigger",
    "CollectiveMarginAttack",
    "CompressionAttack",
    "StealthRampAttack",
    "LearnedAttack",
    "WhiteBoxEvader",
    "InnovationEvader",
    "TRIGGERS",
    "make_trigger",
    "build_attack",
]


def build_attack(name: str, **kw):
    """Factory used by the experiment runner. `name` is an A-id or alias."""
    name = name.upper()
    table = {
        "A0": NoAttack,
        "A1": RandomPerturbation,
        "A2": ConstantBias,
        "A3": PeriodicTrigger,
        "A4": FalseDataInjection,
        "A5": ReplayDelay,
        "A6": FixedStateTrigger,
        "A8": CollectiveMarginAttack,
        "A7": LearnedAttack,
        "A9": StealthRampAttack,
        "A10": WhiteBoxEvader,
        "A12": InnovationEvader,
        "A13": RandomizedAttack,
    }
    if name in table:
        return table[name](**kw)
    if name == "ALWAYS" or (name.startswith("T") and name[1:].isdigit()):
        return CompressionAttack(trigger=name, **kw)
    raise ValueError(f"unknown attack id {name!r} (A7/RL built via rl_attacker)")
