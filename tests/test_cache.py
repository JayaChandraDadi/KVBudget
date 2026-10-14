import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from kvbudget.cache import (
    FakeQuantKVCache,
    LayerKV,
    MixedKVCache,
    component_bytes_per_token,
    effective_bits,
    install,
    logical_nbytes,
    make_cache,
    probe_geometry,
    uninstall,
    uniform_plan,
)

from .conftest import tiny_llama


def _run_chunks(model, x, caches, cuts=(100, 160)):
    outs, s = [], 0
    for c in list(cuts) + [x.shape[1]]:
        if c > s:
            outs.append(model(x[:, s:c], cache=caches))
        s = c
    for t in range(3):  # a few single-token decode steps
        outs.append(model(x[:, t : t + 1], cache=caches))
    return mx.concatenate(outs, axis=1)


PLANS = [
    [LayerKV(4, 2), LayerKV(None, 8), LayerKV(3, None)],
    [LayerKV(8, 8), LayerKV(2, 2), LayerKV(4, 4)],
    [LayerKV(None, None), LayerKV(2, 8), LayerKV(8, 2)],
]


@pytest.mark.parametrize("plan", PLANS)
def test_real_cache_matches_simulation(plan):
    """Quantized storage + quantized_matmul attention computes the same thing as
    attending over quantize->dequantize values."""
    m = tiny_llama(hidden=256, layers=3, vocab=128, kv_heads=2, heads=4)
    x = mx.random.randint(0, 128, (2, 240))
    real = _run_chunks(m, x, make_cache(m, plan, real=True))
    sim = _run_chunks(m, x, make_cache(m, plan, real=False))
    assert mx.allclose(real, sim, atol=1e-4).item()


@pytest.mark.parametrize("bits", [4, 8])
def test_uniform_matches_mlx_lm_quantized_cache(bits):
    """With K == V bits our attention path agrees with mlx-lm's own."""
    m = tiny_llama(hidden=256, layers=2, vocab=128, kv_heads=2, heads=4)
    x = mx.random.randint(0, 128, (1, 200))
    ours = _run_chunks(m, x, make_cache(m, uniform_plan(2, bits, bits)))
    theirs = _run_chunks(m, x, [QuantizedKVCache(64, bits) for _ in range(2)])
    assert mx.allclose(ours, theirs, atol=1e-4).item()


def test_fp16_plan_is_exact():
    m = tiny_llama(hidden=256, layers=2, vocab=128)
    x = mx.random.randint(0, 128, (1, 120))
    a = m(x, cache=make_cache(m, uniform_plan(2, None, None)))
    b = m(x)
    assert mx.allclose(a, b, atol=1e-5).item()


def test_quantization_error_ordering():
    m = tiny_llama(hidden=256, layers=2, vocab=128, logit_scale=4.0)
    x = mx.random.randint(0, 128, (1, 200))
    ref = m(x)
    errs = [mx.mean((m(x, cache=make_cache(m, uniform_plan(2, b, b))) - ref) ** 2).item()
            for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_trim_rollback_reproduces_decode():
    m = tiny_llama(hidden=256, layers=2, vocab=128)
    x = mx.random.randint(0, 128, (1, 80))
    for plan in (uniform_plan(2, 4, 2), uniform_plan(2, 3, None)):
        caches = make_cache(m, plan)
        m(x, cache=caches)
        a = m(mx.array([[5, 6, 7]]), cache=caches)
        for c in caches:
            assert c.trim(3) == 3
        b = m(mx.array([[5, 6, 7]]), cache=caches)
        assert mx.allclose(a, b, atol=1e-5).item()
        assert all(c.offset == 83 for c in caches)


def test_growth_across_allocation_steps():
    """Writes that straddle the 256-token allocation boundary."""
    m = tiny_llama(hidden=256, layers=2, vocab=128)
    x = mx.random.randint(0, 128, (1, 600))
    plan = uniform_plan(2, 4, 2)
    a = _run_chunks(m, x, make_cache(m, plan), cuts=(250, 300, 511, 513))
    b = _run_chunks(m, x, make_cache(m, plan, real=False), cuts=(600,))
    assert mx.allclose(a, b, atol=1e-4).item()


def test_byte_accounting():
    m = tiny_llama(hidden=256, layers=2, vocab=128, kv_heads=2, heads=4, dtype=mx.bfloat16)
    geo = probe_geometry(m)
    assert (geo.n_layers, geo.n_kv_heads, geo.head_dim) == (2, 2, 64)
    plan = [LayerKV(4, 2), LayerKV(None, 8)]
    caches = make_cache(m, plan)
    m(mx.random.randint(0, 128, (1, 256)), cache=caches)  # exactly one allocation step
    expected = [256 * (component_bytes_per_token(2, 64, p.k_bits) +
                       component_bytes_per_token(2, 64, p.v_bits)) for p in plan]
    assert [logical_nbytes(c) for c in caches] == expected
    assert effective_bits(4, 64) == 4.5 and effective_bits(None, 64) == 16
    assert effective_bits(2, 128, 64) == 2.5


def test_install_is_idempotent_and_reversible():
    import sys

    m = tiny_llama()
    mod = sys.modules[type(m).__module__]
    uninstall(m)  # earlier tests may have installed it
    orig = mod.scaled_dot_product_attention
    install(m)
    patched = mod.scaled_dot_product_attention
    install(m)
    assert mod.scaled_dot_product_attention is patched is not orig
    uninstall(m)
    assert mod.scaled_dot_product_attention is orig


def test_fake_cache_is_dense():
    c = FakeQuantKVCache(2, 4)
    k = mx.random.normal((1, 2, 5, 64))
    kk, vv = c.update_and_fetch(k, k)
    assert kk.dtype == k.dtype and kk.shape == k.shape
    assert not mx.allclose(kk, k).item()


def test_plan_length_checked():
    with pytest.raises(ValueError):
        make_cache(tiny_llama(layers=2), uniform_plan(3, 4, 4))
