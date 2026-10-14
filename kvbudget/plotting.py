"""Figures for KVBudget results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402

MODE_STYLE = {"uniform": ("#6b7280", "s"), "pairs": ("#2563eb", "D"),
              "independent": ("#dc2626", "o"), "independent (2/4/8 only)": ("#f59e0b", "v")}
MODE_LABEL = {"uniform": "uniform (K, V)", "pairs": "layer-wise pairs (KVTuner-style space)",
              "independent": "KVBudget: independent K/V per layer"}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def plot_quality_vs_memory(rows, out, metric="kl", title=None):
    q = [r for r in rows if r.get("kind") == "quality"]
    lens = sorted({r["seq_len"] for r in q})
    fig, axes = plt.subplots(1, len(lens), figsize=(5.2 * len(lens), 4), dpi=150, squeeze=False)
    for ax, L in zip(axes[0], lens):
        by = defaultdict(list)
        for r in q:
            if r["seq_len"] == L and r["avg_bits"] < 15.9:
                by[r["mode"]].append(r)
        for mode, rs in by.items():
            c, mk = MODE_STYLE.get(mode, ("#111827", "o"))
            rs.sort(key=lambda r: r["avg_bits"])
            if mode == "uniform":
                ax.scatter([r["avg_bits"] for r in rs], [r[metric] for r in rs], color=c,
                           marker=mk, label=MODE_LABEL[mode], zorder=3)
                for r in rs:
                    ax.annotate(r["config"].replace("uniform-", ""), (r["avg_bits"], r[metric]),
                                fontsize=6, xytext=(3, 3), textcoords="offset points")
            else:
                ls = ":" if "only" in mode else "-"
                ax.plot([r["avg_bits"] for r in rs], [r[metric] for r in rs], color=c,
                        marker=mk, ls=ls, label=MODE_LABEL.get(mode, mode))
        ax.set_yscale("log")
        ax.set_xlabel("KV cache: average bits per element (incl. scales)")
        ax.set_ylabel({"kl": "KL to fp16 cache (nats/token)",
                       "delta_ppl": "perplexity increase"}.get(metric, metric))
        ax.set_title(f"context {L}", fontsize=9)
        ax.grid(alpha=0.25, which="both")
    axes[0][0].legend(frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sensitivity(sens: dict, out, title=None):
    n = sens["geometry"]["n_layers"]
    bits = sorted(int(b) for b in sens["bits"])
    rows, labels = [], []
    for comp in ("k", "v"):
        for b in bits:
            rows.append([sens["errors"][f"{i}.{comp}"][str(b)] for i in range(n)])
            labels.append(f"{comp.upper()}{b}")
    m = np.maximum(np.array(rows), 1e-7)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.3), 3.4), dpi=150)
    im = ax.imshow(m, aspect="auto", cmap="magma_r", norm=LogNorm(vmin=m.min(), vmax=m.max()))
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(n), [str(i) for i in range(n)], fontsize=6)
    ax.set_xlabel("layer")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("KL when only this is quantized")
    ax.set_title(title or "KV-cache sensitivity per layer and component", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_plan(plan_json: dict, out):
    plan = plan_json["plan"]
    grid = np.array([[16 if k is None else k for k, _ in plan],
                     [16 if v is None else v for _, v in plan]], dtype=float)
    cmap = ListedColormap(["#7f1d1d", "#dc2626", "#f59e0b", "#93c5fd", "#1d4ed8"])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 6, 12, 17], cmap.N)
    fig, ax = plt.subplots(figsize=(max(6, len(plan) * 0.3), 1.9), dpi=150)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks([0, 1], ["K", "V"])
    ax.set_xticks(range(len(plan)), [str(i) for i in range(len(plan))], fontsize=6)
    ax.set_xlabel("layer")
    fig.colorbar(im, ax=ax, ticks=[2, 3, 4, 8, 16], fraction=0.03, pad=0.02).set_label("bits")
    ax.set_title(f"{plan_json.get('name', '')}: {plan_json.get('avg_bits', 0):.2f} avg bits",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_bench(rows, out, title=None):
    b = [r for r in rows if r.get("kind") == "bench"]
    if not b:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    by = defaultdict(list)
    for r in b:
        by[r["config"]].append(r)
    for i, (cfg, rs) in enumerate(sorted(by.items())):
        rs.sort(key=lambda r: r["context"])
        xs = [r["context"] for r in rs]
        axes[0].plot(xs, [r["decode_tps"] for r in rs], marker="o", label=cfg)
        axes[1].plot(xs, [r["kv_mib"] for r in rs], marker="o", label=cfg)
    axes[0].set_ylabel("decode tokens/s")
    axes[1].set_ylabel("KV cache (MiB)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_needle(rows, out, title=None):
    nrows = [r for r in rows if r.get("kind") == "needle"]
    if not nrows:
        return None
    cfgs = sorted({r["config"] for r in nrows})
    ctxs = sorted({r["context"] for r in nrows})
    acc = np.zeros((len(cfgs), len(ctxs)))
    for i, c in enumerate(cfgs):
        for j, L in enumerate(ctxs):
            v = [r["accuracy"] for r in nrows if r["config"] == c and r["context"] == L]
            acc[i, j] = np.mean(v) if v else np.nan
    fig, ax = plt.subplots(figsize=(1.6 * len(ctxs) + 3, 0.45 * len(cfgs) + 1.5), dpi=150)
    im = ax.imshow(acc, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ctxs)), [str(c) for c in ctxs], fontsize=7)
    ax.set_yticks(range(len(cfgs)), cfgs, fontsize=7)
    for i in range(len(cfgs)):
        for j in range(len(ctxs)):
            ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("context length")
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("needle retrieval accuracy")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_all(results, sens_path, plan_paths, out_dir, title=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(results)
    made = []
    if any(r.get("kind") == "quality" for r in rows):
        made.append(plot_quality_vs_memory(rows, out / "kl_vs_memory.png", "kl", title))
        made.append(plot_quality_vs_memory(rows, out / "ppl_vs_memory.png", "delta_ppl", title))
    if sens_path:
        made.append(plot_sensitivity(json.loads(Path(sens_path).read_text()),
                                     out / "sensitivity.png", title))
    for p in plan_paths:
        made.append(plot_plan(json.loads(Path(p).read_text()), out / f"plan_{Path(p).stem}.png"))
    for f in (plot_bench(rows, out / "bench.png", title),
              plot_needle(rows, out / "needle.png", title)):
        if f:
            made.append(f)
    return made
