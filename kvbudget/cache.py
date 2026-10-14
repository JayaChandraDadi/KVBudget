"""KV caches with independent key / value precision per layer.

mlx-lm's ``QuantizedKVCache`` uses one bit-width for keys *and* values, and
one setting for every layer. Keys and values behave differently (keys carry
outlier channels that feed a softmax; values are averaged by it), and layers
differ in how much cache noise they tolerate, so KVBudget gives every layer
its own ``(k_bits, v_bits)`` with ``None`` meaning "keep 16-bit".

Two cache types:

* :class:`MixedKVCache` -- real storage. Quantized components are kept as MLX
  affine ``(packed, scales, biases)`` triplets and attention runs on them with
  ``mx.quantized_matmul`` (see :func:`mixed_sdpa`), so memory savings are real.
* :class:`FakeQuantKVCache` -- stores quantize->dequantize values in 16-bit.
  Numerically the cache error is the same, but it runs on the fused dense
  attention kernel. Used for sensitivity profiling and quality sweeps, where
  only accuracy matters.

Both quantize every token as it is written (prefill and decode alike) and
support ``trim`` for speculative-decoding rollback.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.cache import KVCache, _BaseCache, create_attention_mask

Bits = Optional[int]  # None == 16-bit (unquantized)


@dataclass(frozen=True)
class LayerKV:
    k_bits: Bits = None
    v_bits: Bits = None

    def label(self) -> str:
        return f"K{self.k_bits or 16}V{self.v_bits or 16}"


# --------------------------------------------------------------------------- #
# byte accounting
# --------------------------------------------------------------------------- #
def component_bytes_per_token(n_kv_heads: int, head_dim: int, bits: Bits,
                              group_size: int = 64, dtype_bytes: int = 2) -> float:
    """Bytes one token adds to one layer's K (or V) cache."""
    if bits is None:
        return n_kv_heads * head_dim * dtype_bytes
    groups = head_dim // group_size
    return n_kv_heads * (head_dim * bits / 8 + groups * 2 * dtype_bytes)


def effective_bits(bits: Bits, head_dim: int, group_size: int = 64, dtype_bytes: int = 2) -> float:
    """Bits per cached element including scale/bias metadata (4-bit gs64 -> 4.5)."""
    return component_bytes_per_token(1, head_dim, bits, group_size, dtype_bytes) * 8 / head_dim


# --------------------------------------------------------------------------- #
# caches
# --------------------------------------------------------------------------- #
def _qdq(x: mx.array, bits: Bits, group_size: int) -> mx.array:
    if bits is None:
        return x
    w, s, b = mx.quantize(x, group_size=group_size, bits=bits)
    return mx.dequantize(w, s, b, group_size=group_size, bits=bits).astype(x.dtype)


class FakeQuantKVCache(KVCache):
    """Dense cache holding quantize->dequantize values (accuracy simulation)."""

    def __init__(self, k_bits: Bits = None, v_bits: Bits = None, group_size: int = 64):
        super().__init__()
        self.k_bits, self.v_bits, self.group_size = k_bits, v_bits, group_size

    def update_and_fetch(self, keys, values):
        return super().update_and_fetch(_qdq(keys, self.k_bits, self.group_size),
                                        _qdq(values, self.v_bits, self.group_size))


class MixedKVCache(_BaseCache):
    """Quantized KV storage with separate key / value bit-widths."""

    step = 256

    def __init__(self, k_bits: Bits = 8, v_bits: Bits = 8, group_size: int = 64):
        self.k_bits, self.v_bits, self.group_size = k_bits, v_bits, group_size
        self.keys = None  # array (16-bit) or (packed, scales, biases)
        self.values = None
        self.offset = 0

    # -- storage helpers ------------------------------------------------------
    def _empty(self, like: mx.array, bits: Bits, n: int):
        B, H, _, D = like.shape
        if bits is None:
            return mx.zeros((B, H, n, D), like.dtype)
        if D * bits % 32:
            raise ValueError(f"head_dim {D} x {bits} bits is not a multiple of 32")
        words = D * bits // 32
        return (mx.zeros((B, H, n, words), mx.uint32),
                mx.zeros((B, H, n, D // self.group_size), like.dtype),
                mx.zeros((B, H, n, D // self.group_size), like.dtype))

    @staticmethod
    def _cat(a, b):
        if isinstance(a, tuple):
            return tuple(mx.concatenate([x, y], axis=2) for x, y in zip(a, b))
        return mx.concatenate([a, b], axis=2)

    @staticmethod
    def _slice(a, n):
        if isinstance(a, tuple):
            return tuple(x[..., :n, :] for x in a)
        return a[..., :n, :]

    def _write(self, store, new, bits, start, end):
        if bits is None:
            store[..., start:end, :] = new
        else:
            q = mx.quantize(new, group_size=self.group_size, bits=bits)
            for dst, src in zip(store, q):
                dst[..., start:end, :] = src

    # -- cache protocol ---------------------------------------------------------
    def update_and_fetch(self, keys, values):
        prev = self.offset
        n = keys.shape[2]
        cap = (self.keys[0] if isinstance(self.keys, tuple) else self.keys).shape[2] \
            if self.keys is not None else 0
        if self.keys is None or prev + n > cap:
            steps = (self.step + n - 1) // self.step * self.step
            new_k = self._empty(keys, self.k_bits, steps)
            new_v = self._empty(values, self.v_bits, steps)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step:
                    self.keys = self._slice(self.keys, prev)
                    self.values = self._slice(self.values, prev)
                self.keys = self._cat(self.keys, new_k)
                self.values = self._cat(self.values, new_v)
        self.offset += n
        self._write(self.keys, keys, self.k_bits, prev, self.offset)
        self._write(self.values, values, self.v_bits, prev, self.offset)
        return self._slice(self.keys, self.offset), self._slice(self.values, self.offset)

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.k_bits, self.v_bits, self.group_size

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.k_bits, self.v_bits, self.group_size = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        """Allocated bytes (includes up to ``step`` tokens of spare capacity)."""
        if self.keys is None:
            return 0
        leaves = lambda a: a if isinstance(a, tuple) else (a,)
        return sum(x.nbytes for x in leaves(self.keys) + leaves(self.values))

    def logical_nbytes(self) -> int:
        """Bytes actually holding the ``offset`` cached tokens."""
        if self.keys is None:
            return 0
        leaves = lambda a: a if isinstance(a, tuple) else (a,)
        cap = leaves(self.keys)[0].shape[2]
        return int(self.nbytes * self.offset / cap)


def logical_nbytes(cache) -> int:
    if isinstance(cache, MixedKVCache):
        return cache.logical_nbytes()
    if getattr(cache, "keys", None) is None:
        return 0
    cap = cache.keys.shape[2]
    return int(cache.nbytes * cache.offset / cap)


# --------------------------------------------------------------------------- #
# attention over mixed-precision caches
# --------------------------------------------------------------------------- #
def mixed_sdpa(queries, keys, values, scale, mask, k_bits: Bits, v_bits: Bits,
               group_size: int = 64):
    """Attention where keys and/or values may be quantized triplets.

    Mirrors ``mlx_lm.models.base.quantized_scaled_dot_product_attention`` but
    with independent bit-widths (and either side may be dense). Like the
    mlx-lm version it is not a fused kernel: it materializes the score
    matrix, which matters at long context (see the benchmarks).
    """
    B, n_q, L, D = queries.shape
    kv0 = keys[0] if isinstance(keys, tuple) else keys
    n_kv = kv0.shape[-3]
    rep = n_q // n_kv
    queries = queries * scale
    if rep > 1:
        queries = mx.reshape(queries, (B, n_kv, rep, L, D))
        expand = lambda x: mx.expand_dims(x, axis=-3)
        keys = tree_map(expand, keys) if isinstance(keys, tuple) else expand(keys)
        values = tree_map(expand, values) if isinstance(values, tuple) else expand(values)

    if k_bits is None:
        scores = queries @ mx.swapaxes(keys, -1, -2)
    else:
        scores = mx.quantized_matmul(queries, *keys, transpose=True,
                                     group_size=group_size, bits=k_bits)
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_idx = mx.arange(kL - qL, kL)
            k_idx = mx.arange(kL)
            mask = q_idx[:, None] >= k_idx[None]
        if rep > 1 and mask.ndim > 3:
            mask = mx.expand_dims(mask, -3)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    if v_bits is None:
        out = scores.astype(values.dtype) @ values
    else:
        out = mx.quantized_matmul(scores, *values, transpose=False,
                                  group_size=group_size, bits=v_bits)
    if rep > 1:
        out = mx.reshape(out, (B, n_q, L, D))
    return out


_ORIGINALS: dict = {}


def install(model) -> None:
    """Route attention for :class:`MixedKVCache` layers to :func:`mixed_sdpa`.

    mlx-lm model files import ``scaled_dot_product_attention`` into their own
    namespace, so the hook is installed in the model's module. Other caches
    (dense, mlx-lm's own quantized cache) go to the original function.
    Idempotent.
    """
    mod = sys.modules[type(model).__module__]
    if mod.__name__ in _ORIGINALS:
        return
    original = mod.scaled_dot_product_attention

    def dispatch(queries, keys, values, cache, scale, mask, sinks=None):
        if isinstance(cache, MixedKVCache):
            if sinks is not None:
                raise ValueError("attention sinks are not supported with MixedKVCache")
            return mixed_sdpa(queries, keys, values, scale, mask,
                              cache.k_bits, cache.v_bits, cache.group_size)
        return original(queries, keys, values, cache, scale, mask, sinks)

    _ORIGINALS[mod.__name__] = original
    mod.scaled_dot_product_attention = dispatch


def uninstall(model) -> None:
    mod = sys.modules[type(model).__module__]
    if mod.__name__ in _ORIGINALS:
        mod.scaled_dot_product_attention = _ORIGINALS.pop(mod.__name__)


def make_cache(model, plan: Sequence[LayerKV], group_size: int = 64, real: bool = True):
    """One cache per layer for a per-layer plan.

    16/16 layers get mlx-lm's plain ``KVCache`` (fused attention). With
    ``real=False`` quantized layers get :class:`FakeQuantKVCache`.
    """
    if len(plan) != len(model.layers):
        raise ValueError(f"plan has {len(plan)} layers, model has {len(model.layers)}")
    caches = []
    for p in plan:
        if p.k_bits is None and p.v_bits is None:
            caches.append(KVCache())
        elif real:
            caches.append(MixedKVCache(p.k_bits, p.v_bits, group_size))
        else:
            caches.append(FakeQuantKVCache(p.k_bits, p.v_bits, group_size))
    if real and any(isinstance(c, MixedKVCache) for c in caches):
        install(model)
    return caches


def uniform_plan(n_layers: int, k_bits: Bits, v_bits: Bits) -> list[LayerKV]:
    return [LayerKV(k_bits, v_bits) for _ in range(n_layers)]


@dataclass
class KVGeometry:
    n_layers: int
    n_kv_heads: int
    head_dim: int


def probe_geometry(model) -> KVGeometry:
    """Discover KV shapes by running one token through recording caches
    (works for any mlx-lm architecture that uses ``update_and_fetch``)."""
    shapes = []

    class Probe(KVCache):
        def update_and_fetch(self, k, v):
            shapes.append(k.shape)
            return super().update_and_fetch(k, v)

    model(mx.array([[0]]), cache=[Probe() for _ in model.layers])
    _, H, _, D = shapes[0]
    return KVGeometry(len(model.layers), H, D)
