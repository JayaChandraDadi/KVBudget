"""Multi-head -> grouped-query attention conversion (Track C).

Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints* (EMNLP 2023): convert an MHA checkpoint to GQA
by mean-pooling the key and value projection heads within each group, then
"uptrain" briefly. The paper groups *adjacent* heads.

KVBudget adds **similarity grouping**: replacing a group's K/V heads by their
mean costs, per token, the within-group spread of those heads' outputs. So
heads are grouped to minimise the summed squared distance between their key
and value activations on calibration data (a balanced clustering problem,
solved greedily and refined by pairwise swaps). Query heads are then
permuted so each group's query heads are contiguous (the layout GQA kernels
expect) -- a function-preserving reordering of ``q_proj`` rows and
``o_proj`` columns.

Conversion operates on plain ``{name: ndarray}`` weight dicts so the same
code serves the MLX models and the JAX toy trainer.
"""

from __future__ import annotations

import itertools

import numpy as np


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def group_adjacent(n_heads: int, n_groups: int) -> list[list[int]]:
    if n_heads % n_groups:
        raise ValueError("n_heads must be divisible by n_groups")
    m = n_heads // n_groups
    return [list(range(g * m, (g + 1) * m)) for g in range(n_groups)]


def pairwise_sq_dist(acts: np.ndarray) -> np.ndarray:
    """``acts``: (heads, N, d) per-head activations -> (heads, heads) mean
    squared distance."""
    h = acts.shape[0]
    flat = acts.reshape(h, -1).astype(np.float64)
    sq = (flat**2).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * flat @ flat.T
    return np.maximum(d, 0) / acts.shape[1]


def merge_cost(dist: np.ndarray, groups: list[list[int]]) -> float:
    """Sum over groups of the within-group squared spread around the mean,
    which equals (1/m) * sum of pairwise squared distances in the group."""
    return float(sum(dist[np.ix_(g, g)].sum() / (2 * len(g)) for g in groups))


def group_by_similarity(dist: np.ndarray, n_groups: int, refine: bool = True) -> list[list[int]]:
    """Balanced grouping minimising :func:`merge_cost`."""
    h = dist.shape[0]
    if h % n_groups:
        raise ValueError("n_heads must be divisible by n_groups")
    m = h // n_groups
    if n_groups == 1:
        return [list(range(h))]
    unassigned = set(range(h))
    groups = []
    while unassigned:
        # seed: the head that is hardest to place (farthest from the rest)
        seed = max(unassigned, key=lambda i: dist[i, list(unassigned)].sum())
        g = [seed]
        unassigned.remove(seed)
        while len(g) < m:
            nxt = min(unassigned, key=lambda j: dist[j, g].sum())
            g.append(nxt)
            unassigned.remove(nxt)
        groups.append(sorted(g))
    if refine:
        improved = True
        while improved:
            improved = False
            best = merge_cost(dist, groups)
            for a, b in itertools.combinations(range(len(groups)), 2):
                for i in list(groups[a]):
                    for j in list(groups[b]):
                        ga = [x for x in groups[a] if x != i] + [j]
                        gb = [x for x in groups[b] if x != j] + [i]
                        trial = list(groups)
                        trial[a], trial[b] = sorted(ga), sorted(gb)
                        c = merge_cost(dist, trial)
                        if c < best - 1e-12:
                            groups, best, improved = trial, c, True
                            break
                    if improved:
                        break
                if improved:
                    break
    return sorted(groups, key=lambda g: g[0])


# --------------------------------------------------------------------------- #
# weight conversion
# --------------------------------------------------------------------------- #
def convert_layer(wq, wk, wv, wo, n_heads: int, head_dim: int, groups: list[list[int]],
                  bq=None, bk=None, bv=None):
    """Convert one MHA layer to GQA.

    Shapes follow ``nn.Linear``: ``wq/wk/wv: (heads*head_dim, hidden)``,
    ``wo: (hidden, heads*head_dim)``. Returns new (wq, wk, wv, wo[, biases]).
    """
    perm = [h for g in groups for h in g]
    q = wq.reshape(n_heads, head_dim, -1)[perm].reshape(wq.shape)
    o = wo.reshape(wo.shape[0], n_heads, head_dim)[:, perm].reshape(wo.shape)
    k = wk.reshape(n_heads, head_dim, -1)
    v = wv.reshape(n_heads, head_dim, -1)
    k_new = np.stack([k[g].mean(0) for g in groups]).reshape(len(groups) * head_dim, -1)
    v_new = np.stack([v[g].mean(0) for g in groups]).reshape(len(groups) * head_dim, -1)
    out = [q, k_new, v_new, o]
    if bq is not None:
        out.append(bq.reshape(n_heads, head_dim)[perm].reshape(-1))
        out.append(np.stack([bk.reshape(n_heads, head_dim)[g].mean(0) for g in groups]).reshape(-1))
        out.append(np.stack([bv.reshape(n_heads, head_dim)[g].mean(0) for g in groups]).reshape(-1))
    return out


def convert_weights(weights: dict, n_layers: int, n_heads: int, head_dim: int,
                    groups_per_layer: list[list[list[int]]], prefix: str = "model.layers"):
    """MHA weight dict -> GQA weight dict (other tensors copied)."""
    new = dict(weights)
    for i in range(n_layers):
        p = f"{prefix}.{i}.self_attn."
        args = [weights[p + f"{n}_proj.weight"] for n in "qkvo"]
        bias = [weights.get(p + f"{n}_proj.bias") for n in "qkv"]
        res = convert_layer(*args, n_heads, head_dim, groups_per_layer[i],
                            *(bias if bias[0] is not None else [None] * 3))
        for n, w in zip("qkvo", res[:4]):
            new[p + f"{n}_proj.weight"] = w
        if len(res) > 4:
            for n, b in zip("qkv", res[4:]):
                new[p + f"{n}_proj.bias"] = b
    return new


# --------------------------------------------------------------------------- #
# MLX helpers
# --------------------------------------------------------------------------- #
def head_activations(model, tokens, max_tokens: int = 4096):
    """Per-layer (K, V) projection outputs per head, shape (heads, N, head_dim),
    captured from an mlx-lm Llama-family model."""
    import mlx.core as mx

    acts = []
    h = model.model.embed_tokens(tokens)
    attn0 = model.layers[0].self_attn
    n_kv = attn0.n_kv_heads
    for layer in model.layers:
        x = layer.input_layernorm(h)
        k = layer.self_attn.k_proj(x).reshape(-1, n_kv, attn0.head_dim)
        v = layer.self_attn.v_proj(x).reshape(-1, n_kv, attn0.head_dim)
        stride = max(1, k.shape[0] // max_tokens)
        acts.append((np.array(k[::stride].astype(mx.float32)).transpose(1, 0, 2),
                     np.array(v[::stride].astype(mx.float32)).transpose(1, 0, 2)))
        h = layer(h, "causal")
    return acts


def plan_groups(acts, n_groups: int, method: str) -> list[list[list[int]]]:
    out = []
    for k, v in acts:
        n = k.shape[0]
        if method == "adjacent":
            out.append(group_adjacent(n, n_groups))
        elif method == "similarity":
            out.append(group_by_similarity(pairwise_sq_dist(k) + pairwise_sq_dist(v), n_groups))
        else:
            raise ValueError("method must be 'adjacent' or 'similarity'")
    return out


def grouping_cost(acts, groups_per_layer) -> float:
    """Total within-group K+V activation spread (the quantity similarity
    grouping minimises); lower means less information lost by merging."""
    return sum(merge_cost(pairwise_sq_dist(k) + pairwise_sq_dist(v), g)
               for (k, v), g in zip(acts, groups_per_layer))
