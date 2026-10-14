"""Per-layer, per-component KV-cache sensitivity.

For every layer ``l``, component ``c`` in {K, V} and candidate bit-width
``b``, quantize *only* that component of that layer's cache (simulated, so no
kernel effects) and measure the KL divergence of the next-token distribution
from the fp16-cache model on long contexts. Under the usual additivity
assumption these curves are what the allocator trades off.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

from .cache import FakeQuantKVCache, KVGeometry, probe_geometry
from .scoring import Reference, build_reference, score


@dataclass
class Sensitivity:
    geometry: dict
    bits: list[int]
    group_size: int
    errors: dict[str, dict[int, float]]  # "12.k" -> {bits: KL}
    ref_ppl: float
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        d = asdict(self)
        d["errors"] = {k: {str(b): e for b, e in v.items()} for k, v in self.errors.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path) -> "Sensitivity":
        d = json.loads(Path(path).read_text())
        d["errors"] = {k: {int(b): e for b, e in v.items()} for k, v in d["errors"].items()}
        return cls(**d)

    def error(self, layer: int, comp: str, bits) -> float:
        return 0.0 if bits is None else self.errors[f"{layer}.{comp}"][bits]


def profile(model: nn.Module, tokens: mx.array, bits=(2, 3, 4, 8), group_size: int = 64,
            chunk: int = 512, score_from: int = 512, max_positions: int = 256,
            reference: Reference | None = None, verbose: bool = True) -> Sensitivity:
    geo: KVGeometry = probe_geometry(model)
    t0 = time.time()
    ref = reference or build_reference(model, tokens, chunk, score_from, max_positions)
    errors: dict[str, dict[int, float]] = {}
    for layer in range(geo.n_layers):
        tl = time.time()
        for comp in ("k", "v"):
            errors[f"{layer}.{comp}"] = {}
            for b in bits:
                kb, vb = (b, None) if comp == "k" else (None, b)

                def make(layer=layer, kb=kb, vb=vb):
                    return [FakeQuantKVCache(kb, vb, group_size) if i == layer else KVCache()
                            for i in range(geo.n_layers)]

                errors[f"{layer}.{comp}"][b] = score(model, ref, make)["kl"]
        if verbose:
            e = errors
            print(f"[kvbudget] layer {layer:3d}  K2 {e[f'{layer}.k'][min(bits)]:.4f}  "
                  f"V2 {e[f'{layer}.v'][min(bits)]:.4f}  ({time.time() - tl:.1f}s)", flush=True)
    return Sensitivity(
        geometry=asdict(geo), bits=list(bits), group_size=group_size, errors=errors,
        ref_ppl=float(mx.exp(mx.array(ref.nll)).item()),
        meta={"tokens_shape": list(tokens.shape), "chunk": chunk, "score_from": score_from,
              "max_positions": max_positions, "seconds": round(time.time() - t0, 1)},
    )
