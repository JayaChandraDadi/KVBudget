"""KVBudget: memory-budgeted mixed-precision KV cache for long-context on-device LLMs (MLX)."""

from .allocate import allocate, plan_avg_bits, predicted_error
from .cache import (
    FakeQuantKVCache,
    LayerKV,
    MixedKVCache,
    effective_bits,
    install,
    make_cache,
    probe_geometry,
    uniform_plan,
)
from .profile import Sensitivity, profile
from .scoring import build_reference, score

__all__ = [
    "FakeQuantKVCache", "LayerKV", "MixedKVCache", "Sensitivity", "allocate",
    "build_reference", "effective_bits", "install", "make_cache", "plan_avg_bits",
    "predicted_error", "probe_geometry", "profile", "score", "uniform_plan",
]
__version__ = "0.1.0"
