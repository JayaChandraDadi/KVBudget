"""Choose per-layer K/V bit-widths under a KV-cache memory budget.

Budget = average effective bits per cached element (metadata included), e.g.
4.5 is what uniform 4-bit (group size 64) costs, 16 is the fp16 cache.

Three search spaces over the same sensitivity estimates:

* ``independent`` (KVBudget): every layer's K and V each pick from
  {2, 3, 4, 8, 16} bits.
* ``pairs`` (KVTuner-style): each layer picks one (K, V) pair from a fixed
  list such as K8V4, K4V2, ... -- the search space of layer-wise pair
  methods like KVTuner (ICML 2025). Note this is the *space*, not a
  reimplementation of KVTuner's search algorithm.
* ``uniform``: the same (K, V) for every layer (mlx-lm today, with K == V).

The solver is the exact multiple-choice-knapsack DP from BitBudget
(https://github.com/JayaChandraDadi/BitBudget), vendored here.
"""

from __future__ import annotations

import math

import numpy as np

from .cache import LayerKV, component_bytes_per_token
from .profile import Sensitivity

DEFAULT_PAIRS = [(None, None), (8, 8), (8, 4), (4, 8), (4, 4), (8, 2), (4, 2), (2, 2)]


def solve_dp(errors, costs, budget: float, max_units: int = 100_000):
    """Exact multiple-choice knapsack over discretized cost (rounded up, so the
    budget is never exceeded). Returns the chosen option index per item."""
    min_cost = sum(min(c) for c in costs)
    if min_cost > budget * (1 + 1e-12):
        raise ValueError(f"budget infeasible: minimum cost {min_cost:.1f} > {budget:.1f}")
    flat = [c for row in costs for c in row]
    if all(float(c).is_integer() for c in flat) and budget / math.gcd(*map(int, flat)) <= 5e6:
        # integer byte costs: solve exactly on the gcd grid
        unit = math.gcd(*map(int, flat))
    else:
        unit = budget / max_units
    ic = [[int(math.ceil(c / unit - 1e-9)) for c in row] for row in costs]
    cap = int(math.floor(budget / unit + 1e-9))
    dp = np.full(cap + 1, np.inf)
    dp[0] = 0.0
    choice = np.zeros((len(errors), cap + 1), dtype=np.int16)
    for i, (es, cs) in enumerate(zip(errors, ic)):
        new = np.full(cap + 1, np.inf)
        nc = np.zeros(cap + 1, dtype=np.int16)
        for j, (c, e) in enumerate(zip(cs, es)):
            if c > cap:
                continue
            cand = np.full(cap + 1, np.inf)
            cand[c:] = dp[: cap + 1 - c] + e
            better = cand < new
            new[better] = cand[better]
            nc[better] = j
        dp, choice[i] = new, nc
    if not np.isfinite(dp).any():
        return [min(range(len(c)), key=lambda j: c[j]) for c in costs]
    c = int(np.argmin(dp))
    picks = [0] * len(errors)
    for i in range(len(errors) - 1, -1, -1):
        picks[i] = int(choice[i, c])
        c -= ic[i][picks[i]]
    return picks


def _bytes(sens: Sensitivity, bits, dtype_bytes=2) -> float:
    g = sens.geometry
    return component_bytes_per_token(g["n_kv_heads"], g["head_dim"], bits,
                                     sens.group_size, dtype_bytes)


def elements_per_token(sens: Sensitivity) -> int:
    g = sens.geometry
    return 2 * g["n_layers"] * g["n_kv_heads"] * g["head_dim"]


def plan_bytes_per_token(sens: Sensitivity, plan: list[LayerKV]) -> float:
    return sum(_bytes(sens, p.k_bits) + _bytes(sens, p.v_bits) for p in plan)


def plan_avg_bits(sens: Sensitivity, plan: list[LayerKV]) -> float:
    return plan_bytes_per_token(sens, plan) * 8 / elements_per_token(sens)


def predicted_error(sens: Sensitivity, plan: list[LayerKV]) -> float:
    return sum(sens.error(i, "k", p.k_bits) + sens.error(i, "v", p.v_bits)
               for i, p in enumerate(plan))


def allocate(sens: Sensitivity, avg_bits: float, mode: str = "independent",
             bits_options=None, pairs=None) -> list[LayerKV]:
    n = sens.geometry["n_layers"]
    budget = avg_bits / 8 * elements_per_token(sens)
    if mode == "independent":
        opts = [None] + sorted(bits_options or sens.bits)
        errors, costs = [], []
        for layer in range(n):
            for comp in ("k", "v"):
                errors.append([sens.error(layer, comp, b) for b in opts])
                costs.append([_bytes(sens, b) for b in opts])
        picks = solve_dp(errors, costs, budget)
        return [LayerKV(opts[picks[2 * i]], opts[picks[2 * i + 1]]) for i in range(n)]
    if mode == "pairs":
        pairs = [p for p in (pairs or DEFAULT_PAIRS)
                 if all(b is None or b in sens.bits for b in p)]
        errors = [[sens.error(i, "k", k) + sens.error(i, "v", v) for k, v in pairs]
                  for i in range(n)]
        costs = [[_bytes(sens, k) + _bytes(sens, v) for k, v in pairs] for _ in range(n)]
        picks = solve_dp(errors, costs, budget)
        return [LayerKV(*pairs[j]) for j in picks]
    raise ValueError("mode must be 'independent' or 'pairs'")


def describe(plan: list[LayerKV]) -> dict:
    labels = [p.label() for p in plan]
    return {"layers": labels, "counts": {l: labels.count(l) for l in sorted(set(labels))}}


def plan_to_json(plan: list[LayerKV]) -> list[list]:
    return [[p.k_bits, p.v_bits] for p in plan]


def plan_from_json(rows) -> list[LayerKV]:
    return [LayerKV(k, v) for k, v in rows]
