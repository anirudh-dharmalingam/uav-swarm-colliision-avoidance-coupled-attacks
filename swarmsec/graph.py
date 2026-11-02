"""Proximity-graph utilities: neighbour sets, Laplacian, algebraic connectivity.

lambda_2(L) (the Fiedler value / algebraic connectivity) is used strictly as an
*evaluation* metric. The attacker never reads the global graph; it only sees the
induced subgraph over its own neighbourhood (see attacks.triggers).
"""

from __future__ import annotations

import numpy as np


def distance_matrix(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=2)


def adjacency(positions: np.ndarray, radius: float) -> np.ndarray:
    d = distance_matrix(positions)
    a = (d <= radius).astype(float)
    np.fill_diagonal(a, 0.0)
    return a


def laplacian(adj: np.ndarray) -> np.ndarray:
    deg = np.diag(adj.sum(axis=1))
    return deg - adj


def algebraic_connectivity(positions: np.ndarray, radius: float) -> float:
    """Second-smallest Laplacian eigenvalue. 0 means the graph is disconnected."""
    adj = adjacency(positions, radius)
    if adj.shape[0] < 2:
        return 0.0
    eig = np.linalg.eigvalsh(laplacian(adj))
    eig = np.sort(eig)
    return float(max(eig[1], 0.0))


def connected_components(positions: np.ndarray, radius: float) -> int:
    adj = adjacency(positions, radius)
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u] > 0)[0]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return comps


def neighbours_of(agent: int, positions: np.ndarray, radius: float) -> np.ndarray:
    """Indices within `radius` of `agent` (excluding itself)."""
    d = np.linalg.norm(positions - positions[agent], axis=1)
    idx = np.nonzero((d <= radius) & (np.arange(len(d)) != agent))[0]
    return idx


def local_connectivity(agent: int, positions: np.ndarray, radius: float) -> float:
    """lambda_2 of the induced subgraph on {agent} U neighbours(agent).

    This *is* locally observable: the agent knows the positions of the nodes it
    can hear, hence the edges among them. Used as a realistic attack trigger.
    """
    idx = neighbours_of(agent, positions, radius)
    if len(idx) == 0:
        return 0.0
    members = np.concatenate(([agent], idx))
    sub = positions[members]
    return algebraic_connectivity(sub, radius)


def neighbours_of_pt(point: np.ndarray, agent: int, positions: np.ndarray, radius: float) -> np.ndarray:
    """Like neighbours_of but measured from an explicit (fresh) own point."""
    d = np.linalg.norm(positions - point, axis=1)
    return np.nonzero((d <= radius) & (np.arange(len(d)) != agent))[0]
