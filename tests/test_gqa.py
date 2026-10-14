import itertools

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama

from kvbudget.gqa import (
    convert_weights,
    group_adjacent,
    group_by_similarity,
    grouping_cost,
    head_activations,
    merge_cost,
    pairwise_sq_dist,
    plan_groups,
)

from .conftest import tiny_llama


def _mha(n_heads=8, hidden=256, layers=2, seed=0):
    return tiny_llama(hidden=hidden, layers=layers, vocab=64, heads=n_heads, kv_heads=n_heads,
                      seed=seed, logit_scale=3.0)


def _to_gqa(model, groups_per_layer, n_groups):
    a = model.args
    hd = a.hidden_size // a.num_attention_heads
    w = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    new = convert_weights(w, a.num_hidden_layers, a.num_attention_heads, hd, groups_per_layer)
    args = llama.ModelArgs(**{**a.__dict__, "num_key_value_heads": n_groups})
    m = llama.Model(args)
    m.load_weights([(k, mx.array(v)) for k, v in new.items()])
    return m


def test_identity_grouping_preserves_function():
    m = _mha()
    x = mx.random.randint(0, 64, (2, 40))
    groups = [[[h] for h in range(8)]] * 2
    g = _to_gqa(m, groups, 8)
    assert mx.allclose(m(x), g(x), atol=1e-5).item()


@pytest.mark.parametrize("method", ["adjacent", "similarity"])
def test_conversion_exact_when_group_heads_are_identical(method):
    """If K/V heads within each (shuffled) group are equal, merging loses
    nothing; the q/o permutation must keep the function intact."""
    m = _mha()
    rng = np.random.default_rng(0)
    true_groups = [sorted(g) for g in np.split(rng.permutation(8), 4)]
    hd = 32
    for layer in m.model.layers:
        for name in ("k_proj", "v_proj"):
            w = np.array(getattr(layer.self_attn, name).weight).reshape(8, hd, -1)
            for g in true_groups:
                w[g] = w[g[0]]
            getattr(layer.self_attn, name).weight = mx.array(w.reshape(8 * hd, -1))
    x = mx.random.randint(0, 64, (2, 40))
    if method == "adjacent":
        groups = [true_groups] * 2  # adjacency is irrelevant here; use the planted groups
    else:
        groups = plan_groups(head_activations(m, x), 4, "similarity")
        assert all(sorted(map(sorted, g)) == sorted(true_groups) for g in groups)
    g = _to_gqa(m, groups, 4)
    assert mx.allclose(m(x), g(x), atol=1e-4).item()


def test_similarity_grouping_beats_adjacent_on_cost():
    m = _mha(seed=3)
    x = mx.random.randint(0, 64, (2, 64))
    acts = head_activations(m, x)
    for G in (4, 2):
        sim = grouping_cost(acts, plan_groups(acts, G, "similarity"))
        adj = grouping_cost(acts, plan_groups(acts, G, "adjacent"))
        assert sim <= adj + 1e-9


def test_similarity_grouping_is_balanced_and_finds_planted_clusters():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(3, 50, 8)) * 5
    labels = rng.permutation(np.repeat(np.arange(3), 4))
    acts = np.stack([centers[l] + 0.01 * rng.normal(size=(50, 8)) for l in labels])
    groups = group_by_similarity(pairwise_sq_dist(acts), 3)
    assert sorted(len(g) for g in groups) == [4, 4, 4]
    assert sorted(itertools.chain(*groups)) == list(range(12))
    for g in groups:
        assert len({labels[i] for i in g}) == 1


def test_similarity_grouping_optimal_on_small_instances():
    """Compare against brute force for 6 heads into 3 pairs."""
    rng = np.random.default_rng(2)
    for _ in range(5):
        d = pairwise_sq_dist(rng.normal(size=(6, 20, 4)))
        got = merge_cost(d, group_by_similarity(d, 3))
        best = min(
            merge_cost(d, [list(p[0:2]), list(p[2:4]), list(p[4:6])])
            for p in itertools.permutations(range(6))
        )
        assert got <= best * 1.10 + 1e-9  # greedy + swaps: within 10% of optimum


def test_merge_cost_definition():
    acts = np.array([[[0.0]], [[2.0]]])  # two heads, one token, scalar
    d = pairwise_sq_dist(acts)
    # spread around the mean (1): (0-1)^2 + (2-1)^2 = 2
    assert merge_cost(d, [[0, 1]]) == pytest.approx(2.0)


def test_group_adjacent():
    assert group_adjacent(8, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    with pytest.raises(ValueError):
        group_adjacent(8, 3)
