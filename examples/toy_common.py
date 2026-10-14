"""Shared pieces for the toy experiments: corpus, a JAX Llama that mirrors
``mlx_lm.models.llama`` (with any number of KV heads), training, and MLX
loading/evaluation."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# corpus (same deterministic file-level split as the BitBudget / QuantDraft toys)
# --------------------------------------------------------------------------- #
def load_corpus(root="/usr/lib/python3.13", max_bytes=8_000_000):
    root = Path(root)
    train, test = bytearray(), bytearray()
    for p in sorted(p for p in root.rglob("*.py") if "test" not in p.parts
                    and "site-packages" not in p.parts):
        held = int(hashlib.md5(str(p.relative_to(root)).encode()).hexdigest(), 16) % 10 == 0
        (test if held else train).extend(p.read_bytes() + b"\n")
        if len(train) > max_bytes:
            break
    as_np = lambda b: np.frombuffer(bytes(b), dtype=np.uint8).astype(np.int32)
    return as_np(train[:max_bytes]), as_np(test)


def config(layers=4, hidden=256, heads=8, kv_heads=8, inter=704):
    return {"model_type": "llama", "hidden_size": hidden, "num_hidden_layers": layers,
            "intermediate_size": inter, "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "head_dim": hidden // heads,
            "rms_norm_eps": 1e-5, "vocab_size": 256, "tie_word_embeddings": True}


# --------------------------------------------------------------------------- #
# JAX model + training
# --------------------------------------------------------------------------- #
def init_params(cfg, seed=0):
    import jax
    import jax.numpy as jnp

    H, I, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 8 * L + 2))
    n = lambda s, std=0.02: std * jax.random.normal(next(keys), s)
    out_std = 0.02 / math.sqrt(2 * L)
    p = {"model.embed_tokens.weight": n((cfg["vocab_size"], H)), "model.norm.weight": jnp.ones(H)}
    for i in range(L):
        pre = f"model.layers.{i}."
        p.update({pre + "input_layernorm.weight": jnp.ones(H),
                  pre + "post_attention_layernorm.weight": jnp.ones(H),
                  pre + "self_attn.q_proj.weight": n((nh * hd, H)),
                  pre + "self_attn.k_proj.weight": n((nkv * hd, H)),
                  pre + "self_attn.v_proj.weight": n((nkv * hd, H)),
                  pre + "self_attn.o_proj.weight": n((H, nh * hd), out_std),
                  pre + "mlp.gate_proj.weight": n((I, H)),
                  pre + "mlp.up_proj.weight": n((I, H)),
                  pre + "mlp.down_proj.weight": n((H, I), out_std)})
    return p


def forward(p, tokens, cfg):
    import jax
    import jax.numpy as jnp

    L = cfg["num_hidden_layers"]
    nh, nkv, hd, eps = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                        cfg["head_dim"], cfg["rms_norm_eps"])
    B, T = tokens.shape

    def rms(x, w):
        return w * (x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps))

    def rope(x):
        half = hd // 2
        inv = 10000.0 ** (-jnp.arange(half) / half)
        ang = jnp.arange(T)[:, None] * inv[None]
        c, s = jnp.cos(ang), jnp.sin(ang)
        a, b = x[..., :half], x[..., half:]
        return jnp.concatenate([a * c - b * s, a * s + b * c], -1)

    h = p["model.embed_tokens.weight"][tokens]
    mask = jnp.tril(jnp.ones((T, T), bool))
    for i in range(L):
        pre = f"model.layers.{i}."
        x = rms(h, p[pre + "input_layernorm.weight"])
        q = (x @ p[pre + "self_attn.q_proj.weight"].T).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        k = (x @ p[pre + "self_attn.k_proj.weight"].T).reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
        v = (x @ p[pre + "self_attn.v_proj.weight"].T).reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
        q, k = rope(q), rope(k)
        rep = nh // nkv  # query head j uses kv head j // rep (as in mlx sdpa)
        k, v = jnp.repeat(k, rep, axis=1), jnp.repeat(v, rep, axis=1)
        s = jnp.where(mask, (q @ k.transpose(0, 1, 3, 2)) * hd**-0.5, -jnp.inf)
        a = (jax.nn.softmax(s, -1) @ v).transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
        h = h + a @ p[pre + "self_attn.o_proj.weight"].T
        x = rms(h, p[pre + "post_attention_layernorm.weight"])
        g = jax.nn.silu(x @ p[pre + "mlp.gate_proj.weight"].T) * (x @ p[pre + "mlp.up_proj.weight"].T)
        h = h + g @ p[pre + "mlp.down_proj.weight"].T
    h = rms(h, p["model.norm.weight"])
    return h @ p["model.embed_tokens.weight"].T


def train(cfg, data, steps, lr, batch=8, seq_len=512, seed=0, params=None, warmup=None,
          log_every=100):
    """AdamW + cosine. ``params`` (dict of arrays) continues from a checkpoint
    (uptraining); otherwise trains from scratch."""
    import jax
    import jax.numpy as jnp
    import optax

    p = {k: jnp.asarray(v) for k, v in (params or init_params(cfg, seed)).items()}
    warmup = min(warmup if warmup is not None else 100, max(steps // 10, 1))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.05)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(sched, weight_decay=0.1,
                                 mask=lambda q: {k: v.ndim == 2 for k, v in q.items()}))
    state = tx.init(p)
    d = jnp.asarray(data)

    def loss_fn(q, x):
        logits = forward(q, x[:, :-1], cfg)
        return optax.softmax_cross_entropy_with_integer_labels(logits, x[:, 1:]).mean()

    @jax.jit
    def step(q, s, key):
        idx = jax.random.randint(key, (batch,), 0, d.size - seq_len - 1)
        x = d[idx[:, None] + jnp.arange(seq_len + 1)[None]]
        loss, g = jax.value_and_grad(loss_fn)(q, x)
        upd, s = tx.update(g, s, q)
        return optax.apply_updates(q, upd), s, loss

    key = jax.random.PRNGKey(seed + 1)
    t0, log = time.time(), []
    for i in range(1, steps + 1):
        key, sub = jax.random.split(key)
        p, state, loss = step(p, state, sub)
        if i % log_every == 0 or i == 1:
            log.append({"step": i, "loss": float(loss)})
            print(f"    step {i:5d} loss {float(loss):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {k: np.asarray(v, dtype=np.float32) for k, v in p.items()}, log


# --------------------------------------------------------------------------- #
# MLX side
# --------------------------------------------------------------------------- #
def mlx_model(cfg, weights: dict):
    import mlx.core as mx
    from mlx_lm.models import llama

    m = llama.Model(llama.ModelArgs(**cfg))
    m.load_weights([(k, mx.array(v)) for k, v in weights.items()])
    mx.eval(m.parameters())
    return m


def mlx_weights(model) -> dict:
    from mlx.utils import tree_flatten

    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


def heldout_ppl(model, test, seq_len=512, n=64, batch=8) -> float:
    import mlx.core as mx
    import mlx.nn as nn

    k = test.size // seq_len
    rows = mx.array(test[: k * seq_len].reshape(k, seq_len)[:n])
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], batch):
        x = rows[i : i + batch]
        lg = model(x[:, :-1]).astype(mx.float32)
        tot += nn.losses.cross_entropy(lg, x[:, 1:], reduction="sum").item()
        cnt += x[:, 1:].size
    return math.exp(tot / cnt)
