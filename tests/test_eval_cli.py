import json

import mlx.core as mx

from kvbudget.cache import make_cache, uniform_plan
from kvbudget.cli import main
from kvbudget.evaluate import bench, greedy, needle_prompt

from .conftest import tiny_llama, write_byte_model_dir

TEXT = ("The quick brown fox jumps over the lazy dog. " * 40 +
        "def add(a, b):\n    return a + b\n" * 30)


def test_greedy_with_fp_cache_matches_plain_model():
    m = tiny_llama(vocab=64, logit_scale=6.0)
    prompt = [1, 2, 3, 4, 5] * 20
    a = greedy(m, prompt, make_cache(m, uniform_plan(2, None, None)), 10, chunk=32)
    # reference: plain mlx forward each step
    seq, ref = list(prompt), []
    for _ in range(10):
        t = int(mx.argmax(m(mx.array([seq]))[0, -1]).item())
        ref.append(t)
        seq.append(t)
    assert a == ref


def test_bench_reports_memory(tmp_path):
    m = tiny_llama(vocab=64)
    r16 = bench(m, uniform_plan(2, None, None), 300, gen_tokens=4, repeats=1)
    r4 = bench(m, uniform_plan(2, 4, 4), 300, gen_tokens=4, repeats=1)
    assert r4["kv_bytes"] < r16["kv_bytes"] / 2
    assert r4["decode_tps"] > 0 and r4["prefill_tps"] > 0


def test_cli_end_to_end(tmp_path):
    model_dir = write_byte_model_dir(tmp_path / "tiny", tiny_llama(vocab=256, logit_scale=5.0))
    data = tmp_path / "data.txt"
    data.write_text(TEXT * 3)
    res = tmp_path / "res.jsonl"
    common = ["--model", str(model_dir), "--dataset", str(data), "--chunk", "64",
              "--score-from", "64", "--max-positions", "32"]
    main(["profile", *common, "--seq-len", "256", "--n-seqs", "2", "--bits", "2,4,8",
          "--out", str(tmp_path / "sens.json")])
    for mode in ("independent", "pairs"):
        main(["allocate", "--sens", str(tmp_path / "sens.json"), "--avg-bits", "4.0",
              "--mode", mode, "--out", str(tmp_path / f"{mode}.json")])
    main(["eval", *common, "--plan", str(tmp_path / "independent.json"),
          str(tmp_path / "pairs.json"), "--uniform", "4,4", "8,4", "16,16", "--sens",
          str(tmp_path / "sens.json"), "--seq-lens", "256", "--n-seqs", "2",
          "--results", str(res)])
    main(["bench", *common, "--uniform", "4,2", "--contexts", "200", "--gen-tokens", "4",
          "--results", str(res)])
    main(["needle", *common, "--uniform", "8,8", "--contexts", "300", "--trials", "1",
          "--results", str(res)])
    rows = [json.loads(l) for l in res.read_text().splitlines()]
    q = [r for r in rows if r["kind"] == "quality"]
    assert {r["mode"] for r in q} == {"independent", "pairs", "uniform"}
    fp = [r for r in q if r["config"] == "uniform-K16V16"][0]
    assert fp["kl"] < 1e-6 and fp["avg_bits"] == 16
    assert any(r["kind"] == "bench" for r in rows) and any(r["kind"] == "needle" for r in rows)
    main(["plot", "--results", str(res), "--sens", str(tmp_path / "sens.json"), "--plans",
          str(tmp_path / "independent.json"), "--out", str(tmp_path / "fig")])
    for f in ("kl_vs_memory.png", "sensitivity.png", "plan_independent.png", "bench.png",
              "needle.png"):
        assert (tmp_path / "fig" / f).exists(), f


def test_needle_prompt_contains_code():
    from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: F401

    class Tok:
        chat_template = None

        def encode(self, s):
            return list(s.encode())

        def decode(self, t):
            return bytes(t).decode(errors="ignore")

    tok = Tok()
    p = needle_prompt(tok, list(b"lorem ipsum " * 100), 600, 0.5, "48213")
    assert "48213" in tok.decode(p) and 400 < len(p) < 800


def test_works_as_mlx_lm_prompt_cache(tmp_path):
    """The per-layer cache list plugs into mlx-lm's own generation loop."""
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from kvbudget.cache import LayerKV, MixedKVCache

    write_byte_model_dir(tmp_path / "m", tiny_llama(vocab=256, logit_scale=5.0))
    m, tok = load(str(tmp_path / "m"))
    plan = [LayerKV(4, 2), LayerKV(8, None)]
    texts = []
    for real in (True, False):
        cache = make_cache(m, plan, real=real)
        texts.append("".join(r.text for r in stream_generate(
            m, tok, "def f(x):", max_tokens=12, prompt_cache=cache, sampler=make_sampler(0.0))))
        if real:
            assert isinstance(cache[0], MixedKVCache) and cache[0].offset > 9
    assert texts[0] == texts[1]
