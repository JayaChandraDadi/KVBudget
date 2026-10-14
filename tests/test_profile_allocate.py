import math

import mlx.core as mx
import pytest

from kvbudget.allocate import (
    DEFAULT_PAIRS,
    allocate,
    plan_avg_bits,
    plan_from_json,
    plan_to_json,
    predicted_error,
    solve_dp,
)
from kvbudget.cache import FakeQuantKVCache, make_cache, uniform_plan
from kvbudget.profile import Sensitivity, profile
from kvbudget.scoring import build_reference, score

from .conftest import tiny_llama


@pytest.fixture(scope="module")
def setup():
    m = tiny_llama(hidden=256, layers=3, vocab=128, logit_scale=4.0, kv_heads=2, heads=4)
    mx.random.seed(3)
    toks = mx.random.randint(0, 128, (2, 192))
    ref = build_reference(m, toks, chunk=64, score_from=64, max_positions=64)
    sens = profile(m, toks, bits=(2, 3, 4, 8), chunk=64, score_from=64, max_positions=64,
                   reference=ref, verbose=False)
    return m, toks, ref, sens


def test_reference_scores_zero_against_itself(setup):
    m, _, ref, _ = setup
    r = score(m, ref, lambda: make_cache(m, uniform_plan(3, None, None)))
    assert r["kl"] == pytest.approx(0.0, abs=1e-6)
    assert r["delta_ppl"] == pytest.approx(0.0, abs=1e-5)


def test_chunking_does_not_change_fp_nll(setup):
    m, toks, ref, _ = setup
    ref_big = build_reference(m, toks, chunk=192, score_from=64, max_positions=64)
    assert ref_big.nll == pytest.approx(ref.nll, rel=1e-4)


def test_real_and_simulated_scores_agree(setup):
    m, _, ref, _ = setup
    plan = uniform_plan(3, 4, 2)
    a = score(m, ref, lambda: make_cache(m, plan, real=True))
    b = score(m, ref, lambda: make_cache(m, plan, real=False))
    assert a["kl"] == pytest.approx(b["kl"], rel=1e-3)


def test_sensitivity_shape_and_ordering(setup):
    _, _, _, sens = setup
    assert set(sens.errors) == {f"{i}.{c}" for i in range(3) for c in "kv"}
    for key, e in sens.errors.items():
        assert e[2] > e[8] >= 0, key
    assert sens.error(0, "k", None) == 0.0


def test_sensitivity_roundtrip(tmp_path, setup):
    _, _, _, sens = setup
    sens.save(tmp_path / "s.json")
    s2 = Sensitivity.load(tmp_path / "s.json")
    assert s2.errors == sens.errors and s2.geometry == sens.geometry


@pytest.mark.parametrize("avg", [2.6, 3.0, 4.5, 6.0, 9.0])
def test_allocation_respects_budget(setup, avg):
    _, _, _, sens = setup
    for mode in ("independent", "pairs"):
        plan = allocate(sens, avg, mode)
        assert plan_avg_bits(sens, plan) <= avg + 1e-9


def test_independent_never_worse_than_pairs(setup):
    """With the same candidate bits, the pair space is a subset of the
    independent space, so the optimum can only improve."""
    _, _, _, sens = setup
    for avg in (3.0, 4.0, 5.0, 7.0):
        ind = predicted_error(sens, allocate(sens, avg, "independent", bits_options=[2, 4, 8]))
        prs = predicted_error(sens, allocate(sens, avg, "pairs"))
        assert ind <= prs * (1 + 1e-6) + 1e-12


def test_more_budget_never_hurts(setup):
    _, _, _, sens = setup
    errs = [predicted_error(sens, allocate(sens, a)) for a in (2.6, 3.5, 4.5, 6, 9, 16)]
    assert all(x >= y - 1e-12 for x, y in zip(errs, errs[1:]))
    assert errs[-1] == 0.0  # 16 bits -> everything fp16


def test_infeasible_budget():
    with pytest.raises(ValueError):
        solve_dp([[1.0, 0.0]], [[10, 20]], 5)


def test_plan_json_roundtrip(setup):
    _, _, _, sens = setup
    plan = allocate(sens, 4.0)
    assert plan_from_json(plan_to_json(plan)) == plan


def test_predicted_error_tracks_measured(setup):
    """The additive estimate should rank configurations like the measurement."""
    m, _, ref, sens = setup
    plans = [uniform_plan(3, 2, 2), uniform_plan(3, 4, 4), uniform_plan(3, 8, 8),
             allocate(sens, 4.0)]
    pred = [predicted_error(sens, p) for p in plans]
    meas = [score(m, ref, lambda p=p: make_cache(m, p, real=False))["kl"] for p in plans]
    order = lambda v: sorted(range(len(v)), key=lambda i: v[i])
    assert order(pred)[0] == order(meas)[0] and order(pred)[-1] == order(meas)[-1]


def test_default_pairs_contains_fp16():
    assert (None, None) in DEFAULT_PAIRS
