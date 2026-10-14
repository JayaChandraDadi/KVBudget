"""Toy-scale experiments (run anywhere; no downloads).

1. Train an MHA byte-level Llama (8 heads x 32 dims, 8 KV heads, 512 context)
   with JAX. (JAX only because MLX's Linux CPU backend has no BLAS; the JAX
   model is checked against mlx-lm's Llama to 1e-4.)
2. Track C -- GQA conversion to 4 / 2 / 1 KV heads with the paper's adjacent
   mean-pooling vs similarity grouping, before and after a 5% uptraining run,
   plus an "MHA + same extra training" control.
3. Track A -- KV-cache budgets on the MHA model: per-layer K/V sensitivity,
   then uniform vs layer-wise-pairs vs independent per-layer K/V allocation at
   matched average bits, measured (not predicted) KL and perplexity change.

    python examples/toy_experiments.py --out results/toy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import toy_common as tc  # noqa: E402


def parity_check(cfg):
    import jax.numpy as jnp
    import mlx.core as mx

    p = tc.init_params(cfg, seed=123)
    x = np.random.default_rng(0).integers(0, 256, (2, 64))
    ref = np.asarray(tc.forward(p, jnp.asarray(x), cfg))
    got = np.array(tc.mlx_model(cfg, {k: np.asarray(v) for k, v in p.items()})(mx.array(x)))
    err = float(np.abs(ref - got).max())
    assert err < 1e-3, err
    return err


def track_c(cfg, w_mha, train, test, args, out):
    import mlx.core as mx

    from kvbudget.gqa import convert_weights, grouping_cost, head_activations, plan_groups

    nh, hd, L = cfg["num_attention_heads"], cfg["head_dim"], cfg["num_hidden_layers"]
    mha = tc.mlx_model(cfg, w_mha)
    calib = mx.array(train[-1_000_000:][: 8 * 512].reshape(8, 512))
    acts = head_activations(mha, calib)
    rows = [{"variant": "MHA", "kv_heads": nh, "method": "-", "uptrain": 0,
             "ppl": tc.heldout_ppl(mha, test, n=args.n_eval)}]
    print(f"  MHA                         ppl {rows[0]['ppl']:.4f}", flush=True)
    w_ctrl, _ = tc.train(cfg, train, args.uptrain_steps, args.uptrain_lr, params=w_mha,
                         seed=7, warmup=10, log_every=10**9)
    rows.append({"variant": "MHA + extra training (control)", "kv_heads": nh, "method": "-",
                 "uptrain": args.uptrain_steps,
                 "ppl": tc.heldout_ppl(tc.mlx_model(cfg, w_ctrl), test, n=args.n_eval)})
    print(f"  MHA + {args.uptrain_steps} steps (control)   ppl {rows[-1]['ppl']:.4f}", flush=True)

    for G in args.groups:
        gcfg = {**cfg, "num_key_value_heads": G}
        for method in ("adjacent", "similarity"):
            groups = plan_groups(acts, G, method)
            cost = grouping_cost(acts, groups)
            w = convert_weights(w_mha, L, nh, hd, groups)
            ppl0 = tc.heldout_ppl(tc.mlx_model(gcfg, w), test, n=args.n_eval)
            t0 = time.time()
            w_up, _ = tc.train(gcfg, train, args.uptrain_steps, args.uptrain_lr, params=w,
                               seed=7, warmup=10, log_every=10**9)
            ppl1 = tc.heldout_ppl(tc.mlx_model(gcfg, w_up), test, n=args.n_eval)
            rows.append({"variant": f"GQA-{G}", "kv_heads": G, "method": method,
                         "grouping_cost": cost, "ppl_converted": ppl0, "ppl": ppl1,
                         "uptrain": args.uptrain_steps, "groups_layer0": groups[0]})
            print(f"  GQA {G} KV heads, {method:10s}  merge cost {cost:9.3f}  "
                  f"ppl converted {ppl0:.4f} -> uptrained {ppl1:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if G == 2 and method == "similarity":
                np.savez(out / "gqa2_similarity_uptrained.npz", **w_up)
    (out / "gqa_results.json").write_text(json.dumps(rows, indent=1))
    plot_gqa(rows, out / "toy_gqa.png", args.uptrain_steps)
    return rows


def plot_gqa(rows, path, steps):
    mha = next(r for r in rows if r["variant"] == "MHA")["ppl"]
    ctrl = next(r for r in rows if "control" in r["variant"])["ppl"]
    gq = [r for r in rows if r["variant"].startswith("GQA")]
    Gs = sorted({r["kv_heads"] for r in gq}, reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    w = 0.2
    styles = [("adjacent", "ppl_converted", "#9ca3af", "adjacent mean-pool (paper), converted"),
              ("adjacent", "ppl", "#4b5563", f"adjacent, + {steps} uptrain steps"),
              ("similarity", "ppl_converted", "#fca5a5", "similarity grouping, converted"),
              ("similarity", "ppl", "#dc2626", f"similarity, + {steps} uptrain steps")]
    for j, (m, key, c, lab) in enumerate(styles):
        ys = [next(r[key] for r in gq if r["kv_heads"] == G and r["method"] == m) for G in Gs]
        ax.bar(np.arange(len(Gs)) + (j - 1.5) * w, ys, width=w, color=c, label=lab)
    ax.axhline(mha, color="#111827", ls="--", lw=1, label=f"MHA ({mha:.3f})")
    ax.axhline(ctrl, color="#059669", ls=":", lw=1, label=f"MHA + {steps} steps ({ctrl:.3f})")
    ax.set_xticks(range(len(Gs)), [f"{G} KV heads\n({8 // G}x smaller cache)" for G in Gs])
    ax.set_ylabel("held-out perplexity (bytes, log scale)")
    ax.set_yscale("log")
    ax.set_ylim(min(mha, ctrl) * 0.9, max(r["ppl_converted"] for r in gq) * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([2.5, 3, 4, 5, 7, 10, 15, 20, 30, 40])
    for j, (m, key, _, _) in enumerate(styles):
        for i, G in enumerate(Gs):
            y = next(r[key] for r in gq if r["kv_heads"] == G and r["method"] == m)
            ax.text(i + (j - 1.5) * w, y * 1.03, f"{y:.2f}", ha="center", fontsize=6)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Toy MHA -> GQA conversion (8 query heads)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def track_a(cfg, w_mha, train, test, args, out):
    import mlx.core as mx

    from kvbudget.allocate import (allocate, describe, plan_avg_bits, plan_to_json,
                                   predicted_error)
    from kvbudget.cache import make_cache, uniform_plan
    from kvbudget.plotting import plot_plan, plot_quality_vs_memory, plot_sensitivity
    from kvbudget.profile import profile
    from kvbudget.scoring import build_reference, score

    gs = 32  # head_dim is 32
    model = tc.mlx_model(cfg, w_mha)
    k = test.size // args.ctx
    toks = mx.array(test[: k * args.ctx].reshape(k, args.ctx)[: args.n_kv_seqs])
    ref = build_reference(model, toks, chunk=128, score_from=128, max_positions=128)
    t0 = time.time()
    sens = profile(model, toks, bits=(2, 3, 4, 8), group_size=gs, chunk=128, score_from=128,
                   max_positions=128, reference=ref, verbose=False)
    sens.save(out / "kv_sensitivity.json")
    print(f"  sensitivity profiled in {time.time() - t0:.0f}s", flush=True)

    L = cfg["num_hidden_layers"]
    rows, plans = [], {}

    def measure(name, mode, plan):
        r = score(model, ref, lambda: make_cache(model, plan, gs, real=False))
        rec = {"kind": "quality", "config": name, "mode": mode, "seq_len": args.ctx,
               "avg_bits": plan_avg_bits(sens, plan),
               "predicted_kl": predicted_error(sens, plan), **r, **describe(plan)}
        rows.append(rec)
        print(f"  {name:22s} {rec['avg_bits']:6.3f} bits  KL {r['kl']:.5f}  "
              f"dppl {r['delta_ppl']:+.4f}", flush=True)

    for kb, vb in ((None, None), (8, 8), (8, 4), (4, 8), (4, 4), (4, 2), (2, 4), (2, 2)):
        measure(f"uniform-K{kb or 16}V{vb or 16}", "uniform", uniform_plan(L, kb, vb))
    for b in args.budgets:
        for mode in ("pairs", "independent"):
            plan = allocate(sens, b, mode)
            plans[f"{mode}-{b}"] = {"name": f"{mode}-{b}", "plan": plan_to_json(plan),
                                    "avg_bits": plan_avg_bits(sens, plan)}
            measure(f"{mode}-{b}", mode, plan)
    with (out / "kv_results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "kv_plans.json").write_text(json.dumps(plans, indent=1))
    plot_quality_vs_memory(rows, out / "toy_kv_kl_vs_bits.png", "kl",
                           "Toy MHA model, 512-token context")
    plot_sensitivity(json.loads((out / "kv_sensitivity.json").read_text()),
                     out / "toy_kv_sensitivity.png")
    mid = f"independent-{args.budgets[len(args.budgets) // 2]}"
    plot_plan(plans[mid], out / "toy_kv_plan.png")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--uptrain-steps", type=int, default=100)  # 5% of pretraining
    ap.add_argument("--uptrain-lr", type=float, default=5e-4)
    ap.add_argument("--groups", type=int, nargs="+", default=[4, 2, 1])
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--n-kv-seqs", type=int, default=6)
    ap.add_argument("--budgets", type=float, nargs="+", default=[3.5, 4.5, 6.0, 8.0])
    ap.add_argument("--skip", nargs="*", default=[], choices=["gqa", "kv"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = tc.config()
    print(f"JAX vs MLX parity: max |logit diff| {parity_check(cfg):.1e}", flush=True)
    train, test = tc.load_corpus()
    ckpt = out / "mha_model.npz"
    if ckpt.exists():
        w = dict(np.load(ckpt))
    else:
        print(f"training MHA toy ({args.steps} steps)", flush=True)
        w, log = tc.train(cfg, train, args.steps, 2e-3)
        np.savez(ckpt, **w)
        (out / "mha_train_log.json").write_text(json.dumps(log))
    if "gqa" not in args.skip:
        print("\nTrack C: MHA -> GQA conversion", flush=True)
        track_c(cfg, w, train, test, args, out)
    if "kv" not in args.skip:
        print("\nTrack A: KV-cache budgets", flush=True)
        track_a(cfg, w, train, test, args, out)


if __name__ == "__main__":
    main()
