"""Command line interface.

    kvbudget profile  --model M --seq-len 4096 --n-seqs 2 --out sens.json
    kvbudget allocate --sens sens.json --avg-bits 4.0 --mode independent --out plan.json
    kvbudget eval     --model M --plan plan.json --seq-lens 4096,16384 --results r.jsonl
    kvbudget needle   --model M --plan plan.json --contexts 4096,16384,32768 --results r.jsonl
    kvbudget bench    --model M --plan plan.json --contexts 4096,16384,32768 --results r.jsonl
    kvbudget plot     --results r.jsonl --sens sens.json --out figures
Several configurations can be given at once (``--plan a.json b.json --uniform 4,4 16,16``);
`eval` then shares one fp16 reference across all of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _append(path, rec):
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def _bits(s):
    return None if s in ("16", "none", "None") else int(s)


def _plans(a, n_layers):
    """All configurations named on the command line: ``[(plan, name, meta)]``."""
    from .allocate import plan_from_json
    from .cache import uniform_plan

    out = []
    for u in a.uniform or []:
        k, v = (_bits(x) for x in u.split(","))
        out.append((uniform_plan(n_layers, k, v), f"uniform-K{k or 16}V{v or 16}", None))
    for path in a.plan or []:
        d = json.loads(Path(path).read_text())
        out.append((plan_from_json(d["plan"]), d.get("name", Path(path).stem), d))
    if not out:
        raise SystemExit("give at least one --plan or --uniform")
    return out


def cmd_profile(a):
    from mlx_lm import load

    from .evaluate import load_text, long_windows
    from .profile import profile

    model, tok = load(a.model)
    toks = long_windows(tok, load_text(a.dataset), a.seq_len, a.n_seqs)
    sens = profile(model, toks, tuple(int(b) for b in a.bits.split(",")), a.group_size,
                   a.chunk, a.score_from, a.max_positions)
    sens.meta["model"] = a.model
    sens.save(a.out)
    print(f"saved {a.out}")


def cmd_allocate(a):
    from .allocate import allocate, describe, plan_avg_bits, plan_to_json, predicted_error
    from .profile import Sensitivity

    sens = Sensitivity.load(a.sens)
    opts = [int(b) for b in a.bits.split(",")] if a.bits else None
    plan = allocate(sens, a.avg_bits, a.mode, bits_options=opts)
    rec = {"name": a.name or f"{a.mode}-{a.avg_bits:g}", "mode": a.mode,
           "target_avg_bits": a.avg_bits, "avg_bits": plan_avg_bits(sens, plan),
           "predicted_kl": predicted_error(sens, plan), "plan": plan_to_json(plan),
           **describe(plan)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"{rec['name']}: {rec['avg_bits']:.3f} avg bits, predicted KL {rec['predicted_kl']:.4f}, "
          f"{rec['counts']}")


def _common_record(a, model, plan, name, meta):
    from .allocate import plan_avg_bits
    from .cache import probe_geometry
    from .profile import Sensitivity

    geo = probe_geometry(model)
    avg = None
    if a.sens:
        avg = plan_avg_bits(Sensitivity.load(a.sens), plan)
    else:
        from .cache import component_bytes_per_token as cb

        b = sum(cb(geo.n_kv_heads, geo.head_dim, p.k_bits, a.group_size) +
                cb(geo.n_kv_heads, geo.head_dim, p.v_bits, a.group_size) for p in plan)
        avg = b * 8 / (2 * geo.n_layers * geo.n_kv_heads * geo.head_dim)
    return {"tag": a.tag, "model": a.model, "config": name,
            "mode": (meta or {}).get("mode", "uniform"), "avg_bits": avg}


def cmd_eval(a):
    from mlx_lm import load

    from .cache import make_cache
    from .evaluate import load_text, long_windows
    from .scoring import build_reference, score

    model, tok = load(a.model)
    configs = _plans(a, len(model.layers))
    text = load_text(a.dataset)
    for L in (int(x) for x in a.seq_lens.split(",")):
        toks = long_windows(tok, text, L, a.n_seqs)
        # one fp16 reference per length, shared by every configuration
        ref = build_reference(model, toks, a.chunk, a.score_from, a.max_positions)
        for plan, name, meta in configs:
            res = score(model, ref,
                        lambda: make_cache(model, plan, a.group_size, real=not a.simulate))
            rec = {**_common_record(a, model, plan, name, meta), "kind": "quality",
                   "seq_len": L, "n_seqs": int(toks.shape[0]), **res}
            print(json.dumps(rec))
            _append(a.results, rec)
        del ref


def cmd_needle(a):
    from mlx_lm import load

    from .evaluate import load_text, needle_eval

    model, tok = load(a.model)
    text = load_text(a.dataset)
    for plan, name, meta in _plans(a, len(model.layers)):
        rows = needle_eval(model, tok, plan, text, [int(x) for x in a.contexts.split(",")],
                           trials=a.trials, group_size=a.group_size)
        base = _common_record(a, model, plan, name, meta)
        for r in rows:
            rec = {**base, "kind": "needle", **r}
            print(json.dumps(rec))
            _append(a.results, rec)


def cmd_bench(a):
    from mlx_lm import load

    from .evaluate import bench

    model, _ = load(a.model)
    for plan, name, meta in _plans(a, len(model.layers)):
        base = _common_record(a, model, plan, name, meta)
        for L in (int(x) for x in a.contexts.split(",")):
            rec = {**base, "kind": "bench", **bench(model, plan, L, a.gen_tokens, a.group_size)}
            print(json.dumps(rec))
            _append(a.results, rec)


def cmd_plot(a):
    from .plotting import plot_all

    for p in plot_all(a.results, a.sens, a.plans or [], a.out, a.title):
        print(p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kvbudget")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", required=True)
        p.add_argument("--dataset", default="wikitext2")
        p.add_argument("--group-size", type=int, default=64)
        p.add_argument("--chunk", type=int, default=512)
        p.add_argument("--score-from", type=int, default=512)
        p.add_argument("--max-positions", type=int, default=256)

    def plan_args(p):
        p.add_argument("--plan", nargs="*", help="plan JSON(s) from `kvbudget allocate`")
        p.add_argument("--uniform", nargs="*",
                       help="uniform K,V bits for every layer, e.g. 4,4 16,16 8,4")
        p.add_argument("--sens", help="sensitivity file (for exact bit accounting)")
        p.add_argument("--tag", default="run")
        p.add_argument("--results")

    p = sub.add_parser("profile", help="per-layer K/V sensitivity")
    model_args(p)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-seqs", type=int, default=2)
    p.add_argument("--bits", default="2,3,4,8")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_profile)

    al = sub.add_parser("allocate", help="per-layer K/V bits under a budget")
    al.add_argument("--sens", required=True)
    al.add_argument("--avg-bits", type=float, required=True,
                    help="average effective bits per cached element (uniform 4-bit gs64 = 4.5)")
    al.add_argument("--mode", choices=["independent", "pairs"], default="independent")
    al.add_argument("--bits", help="restrict candidate bit-widths (independent mode)")
    al.add_argument("--name")
    al.add_argument("--out", required=True)
    al.set_defaults(fn=cmd_allocate)

    e = sub.add_parser("eval", help="long-context perplexity and KL")
    model_args(e)
    plan_args(e)
    e.add_argument("--seq-lens", default="4096")
    e.add_argument("--n-seqs", type=int, default=4)
    e.add_argument("--simulate", action="store_true", help="fake-quant cache (accuracy only)")
    e.set_defaults(fn=cmd_eval)

    n = sub.add_parser("needle", help="needle-in-a-haystack retrieval")
    model_args(n)
    plan_args(n)
    n.add_argument("--contexts", default="4096,16384")
    n.add_argument("--trials", type=int, default=2)
    n.set_defaults(fn=cmd_needle)

    b = sub.add_parser("bench", help="KV memory, peak memory, prefill and decode speed")
    model_args(b)
    plan_args(b)
    b.add_argument("--contexts", default="4096,16384")
    b.add_argument("--gen-tokens", type=int, default=64)
    b.set_defaults(fn=cmd_bench)

    pl = sub.add_parser("plot")
    pl.add_argument("--results", required=True)
    pl.add_argument("--sens")
    pl.add_argument("--plans", nargs="*")
    pl.add_argument("--out", default="figures")
    pl.add_argument("--title")
    pl.set_defaults(fn=cmd_plot)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
