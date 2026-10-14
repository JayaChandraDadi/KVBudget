"""Long-context scoring with chunked prefill.

Sequences are pushed through the model in chunks (as a real prefill would),
so later chunks attend to the *quantized* cache of earlier ones. Quality is
measured two ways at positions past ``score_from`` (so every scored token has
a long context behind it):

* NLL / perplexity of the next token, at every scored position;
* KL(fp16 || quantized cache) of the next-token distribution, at a strided
  subset of positions whose reference log-probs are kept in memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


@dataclass
class Reference:
    """fp16-cache reference for one batch of sequences."""

    tokens: mx.array  # (N, L)
    positions: list[mx.array]  # per sequence: scored positions for KL
    logprobs: list[mx.array]  # per sequence: (n_pos, V) float32
    nll: float  # mean NLL of the reference at all scored positions
    score_from: int
    chunk: int


def _positions(L: int, score_from: int, max_positions: int) -> mx.array:
    n = L - 1 - score_from
    stride = max(1, -(-n // max_positions))
    return mx.arange(score_from, L - 1, stride)


def _run(model, seq: mx.array, caches, chunk: int, score_from: int, kl_pos: mx.array,
         ref_lp: mx.array | None):
    """Chunked forward over one sequence. Returns (sum NLL, count, kl list, lp at kl_pos)."""
    L = seq.shape[0]
    nll_sum, count = 0.0, 0
    kls, lps = [], []
    pos_list = kl_pos.tolist()
    pi = 0
    for start in range(0, L - 1, chunk):
        end = min(start + chunk, L - 1)
        logits = model(seq[start:end][None], cache=caches)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        lo = max(score_from, start)
        if lo < end:
            tgt = seq[lo + 1 : end + 1]
            nll = -mx.take_along_axis(lp[lo - start : end - start], tgt[:, None], axis=1)
            nll_sum += nll.sum().item()
            count += end - lo
        # positions used for KL in this chunk
        sel = []
        while pi < len(pos_list) and pos_list[pi] < end:
            sel.append(pos_list[pi] - start)
            pi += 1
        if sel:
            cur = lp[mx.array(sel)]
            if ref_lp is None:
                lps.append(cur)
            else:
                r = ref_lp[len(kls) : len(kls) + len(sel)].astype(mx.float32)
                # KL >= 0 exactly; clamp float round-off (seen on Metal: ~-5e-8)
                kl = mx.maximum((mx.exp(r) * (r - cur)).sum(axis=-1), 0.0)
                kls.extend(kl.tolist())
        mx.eval(lps)
    return nll_sum, count, kls, (mx.concatenate(lps) if lps else None)


def build_reference(model: nn.Module, tokens: mx.array, chunk: int = 512,
                    score_from: int = 512, max_positions: int = 256) -> Reference:
    positions, lps = [], []
    nll_sum, count = 0.0, 0
    for seq in tokens:
        pos = _positions(seq.shape[0], score_from, max_positions)
        caches = [KVCache() for _ in model.layers]
        s, c, _, lp = _run(model, seq, caches, chunk, score_from, pos, None)
        nll_sum, count = nll_sum + s, count + c
        positions.append(pos)
        lps.append(lp)
    return Reference(tokens, positions, lps, nll_sum / count, score_from, chunk)


def score(model: nn.Module, ref: Reference, make_caches: Callable[[], list]) -> dict:
    """Quality of a cache configuration against the reference."""
    nll_sum, count, kls = 0.0, 0, []
    for seq, pos, rlp in zip(ref.tokens, ref.positions, ref.logprobs):
        s, c, kl, _ = _run(model, seq, make_caches(), ref.chunk, ref.score_from, pos, rlp)
        nll_sum, count = nll_sum + s, count + c
        kls.extend(kl)
    nll = nll_sum / count
    return {
        "kl": sum(kls) / len(kls),
        "kl_p95": sorted(kls)[int(0.95 * (len(kls) - 1))],
        "nll": nll,
        "ppl": math.exp(nll),
        "ref_ppl": math.exp(ref.nll),
        "delta_ppl": math.exp(nll) - math.exp(ref.nll),
        "scored_tokens": count,
    }
