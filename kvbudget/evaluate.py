"""Long-context evaluation: data, needle-in-a-haystack, memory and speed."""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .cache import LayerKV, logical_nbytes, make_cache


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_text(dataset: str) -> str:
    if Path(dataset).exists():
        return Path(dataset).read_text(encoding="utf-8", errors="ignore")
    if dataset == "wikitext2":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"])
    raise ValueError(f"unknown dataset {dataset!r}")


def long_windows(tokenizer, text: str, seq_len: int, n: int) -> mx.array:
    toks = mx.array(tokenizer.encode(text)).reshape(-1)
    k = toks.size // seq_len
    if k == 0:
        raise ValueError(f"text has {toks.size} tokens, need >= {seq_len}")
    return toks[: k * seq_len].reshape(k, seq_len)[:n]


# --------------------------------------------------------------------------- #
# generation with an explicit cache list
# --------------------------------------------------------------------------- #
def prefill(model, tokens: list[int], caches, chunk: int = 512) -> mx.array:
    """Feed all tokens; return logits of the last position."""
    out = None
    for i in range(0, len(tokens), chunk):
        out = model(mx.array(tokens[i : i + chunk])[None], cache=caches)
        mx.eval([c.state for c in caches])
    return out[0, -1]


def greedy(model, prompt: list[int], caches, max_new: int, eos: set[int] = frozenset(),
           chunk: int = 512) -> list[int]:
    logits = prefill(model, prompt, caches, chunk)
    out = []
    for _ in range(max_new):
        t = int(mx.argmax(logits).item())
        out.append(t)
        if t in eos:
            break
        logits = model(mx.array([[t]]), cache=caches)[0, -1]
    return out


# --------------------------------------------------------------------------- #
# needle in a haystack
# --------------------------------------------------------------------------- #
NEEDLE = "The secret code for the blue vault is {code}."
QUESTION = ("What is the secret code for the blue vault mentioned in the document above? "
            "Reply with the number only.")


def needle_prompt(tokenizer, filler_tokens: list[int], context_len: int, depth: float,
                  code: str) -> list[int]:
    """Chat prompt of ~``context_len`` tokens with the needle at ``depth``."""
    needle = " " + NEEDLE.format(code=code) + " "
    overhead = len(tokenizer.encode(needle + QUESTION)) + 64
    n_fill = max(16, context_len - overhead)
    fill = filler_tokens[:n_fill]
    cut = int(depth * len(fill))
    doc = tokenizer.decode(fill[:cut]) + needle + tokenizer.decode(fill[cut:])
    msg = f"Document:\n{doc}\n\n{QUESTION}"
    if getattr(tokenizer, "chat_template", None):
        return list(tokenizer.apply_chat_template([{"role": "user", "content": msg}],
                                                  add_generation_prompt=True))
    return list(tokenizer.encode(msg + "\nAnswer:"))


def needle_eval(model, tokenizer, plan: list[LayerKV], filler: str, context_lens,
                depths=(0.1, 0.3, 0.5, 0.7, 0.9), trials: int = 2, group_size: int = 64,
                seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    fill = list(tokenizer.encode(filler))
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [])
    rows = []
    for L in context_lens:
        if len(fill) < L:
            fill = fill * (L // max(len(fill), 1) + 1)
        for d in depths:
            hits = 0
            for _ in range(trials):
                code = str(rng.randint(10000, 99999))
                prompt = needle_prompt(tokenizer, fill, L, d, code)
                out = greedy(model, prompt, make_cache(model, plan, group_size), 12, eos)
                hits += code in tokenizer.decode(out)
            rows.append({"context": L, "depth": d, "accuracy": hits / trials,
                         "prompt_tokens": len(prompt)})
    return rows


# --------------------------------------------------------------------------- #
# memory and speed
# --------------------------------------------------------------------------- #
def bench(model, plan: list[LayerKV], context_len: int, gen_tokens: int = 64,
          group_size: int = 64, chunk: int = 512, repeats: int = 2, seed: int = 0) -> dict:
    """Prefill ``context_len`` tokens into a real (quantized) cache, then decode.

    Reports the cache's logical size, process peak memory, prefill throughput
    and decode tokens/s (median over ``repeats`` after a warmup).
    """
    mx.random.seed(seed)
    vocab = model.args.vocab_size if hasattr(model, "args") else 32000
    prompt = mx.random.randint(0, vocab, (context_len,)).tolist()

    def once():
        caches = make_cache(model, plan, group_size, real=True)
        t0 = time.perf_counter()
        logits = prefill(model, prompt, caches, chunk)
        mx.eval(logits)
        t1 = time.perf_counter()
        tok = mx.argmax(logits)
        for _ in range(gen_tokens):
            logits = model(tok.reshape(1, 1), cache=caches)[0, -1]
            tok = mx.argmax(logits)
            mx.eval(tok)
        t2 = time.perf_counter()
        kv = sum(logical_nbytes(c) for c in caches)
        return t1 - t0, t2 - t1, kv

    once()  # warmup
    mx.clear_cache()
    mx.reset_peak_memory()
    runs = [once() for _ in range(repeats)]
    pre = statistics.median(r[0] for r in runs)
    dec = statistics.median(r[1] for r in runs)
    return {
        "context": context_len,
        "kv_bytes": runs[-1][2],
        "kv_mib": runs[-1][2] / 2**20,
        "peak_mem_gb": mx.get_peak_memory() / 1e9,
        "prefill_tps": context_len / pre,
        "decode_tps": gen_tokens / dec,
    }
