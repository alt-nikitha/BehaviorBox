"""Overlay the mean learning curve of every content cluster on the task curve.

Reuses the exact selection + clustering of task_shape_content_clusters.py (same seed
and K => same clusters), then plots:

  panel A  all K cluster mean z-curves on one axis, task curve in black.
  panel B  same for the random control group.
  panel C  small multiples: each cluster's mean +/- 1sd vs the task curve.

The quantitative point is printed too: r(cluster mean, task) per cluster, and the
between-cluster spread compared with the within-cluster spread. If content clusters
carried trajectory information the curves would fan out; if not, they collapse onto
one line and only the group-level offset from the task curve survives.

Usage:
  python plot_cluster_vs_task_curves.py --task medmcqa --top-n 200000 --k 100
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

from task_shape_content_clusters import (
    CACHE_DIR, OUT_DIR, STEPS, open_memmap, task_curve,
)


def select(task, top_n, seed, control):
    z = np.load(CACHE_DIR / f"taskcorr_{task}.npz")
    r, valid = z["r"], z["valid"]
    if control:
        rows = np.random.default_rng(seed).choice(np.where(valid)[0], top_n, replace=False)
    else:
        rows = np.argpartition(-r, top_n)[:top_n]
        rows = rows[np.argsort(-r[rows])]
    return np.sort(rows), r


def gather(rows, batch=20000):
    """Return (embeddings, z-normed prob curves) for rows."""
    mm, edim, n_out = open_memmap()
    E = np.empty((len(rows), edim), np.float32)
    T = np.empty((len(rows), n_out), np.float32)
    for s in range(0, len(rows), batch):
        e = min(s + batch, len(rows))
        blk = np.asarray(mm[rows[s:e]], dtype=np.float32)
        E[s:e] = blk[:, :edim]
        p = blk[:, -n_out:]
        T[s:e] = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-9)
    return E, T


def cluster_means(E, T, k, seed):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                         n_init=3, max_iter=200).fit(E)
    lab = km.labels_
    M = np.stack([T[lab == c].mean(0) for c in range(k) if (lab == c).any()])
    S = np.stack([T[lab == c].std(0) for c in range(k) if (lab == c).any()])
    n = np.array([(lab == c).sum() for c in range(k) if (lab == c).any()])
    return M, S, n, lab


def stats(tag, M, T, c):
    rs = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9) @ c / len(c)
    between = M.std(0).mean()
    within = T.std(0).mean()
    print(f"{tag:<16} r(cluster mean, task): min={rs.min():.3f} median="
          f"{np.median(rs):.3f} max={rs.max():.3f}  | between-cluster sd={between:.3f} "
          f"vs within-group sd={within:.3f}  ({100*between/within:.1f}%)")
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--top-n", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-small", type=int, default=24, help="clusters in the small-multiples panel")
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    c = task_curve(a.task)
    data = {}
    for control in (False, True):
        rows, r = select(a.task, a.top_n, a.seed, control)
        E, T = gather(rows)
        M, S, n, lab = cluster_means(E, T, a.k, a.seed)
        tag = "random control" if control else "shape group"
        rs = stats(tag, M, T, c)
        data[control] = (M, S, n, T, rs)

    fig = plt.figure(figsize=(15, 4.6 + 2.6 * math.ceil(a.n_small / 6)))
    gs = fig.add_gridspec(1 + math.ceil(a.n_small / 6), 6, height_ratios=[1.5] +
                          [1] * math.ceil(a.n_small / 6), hspace=.55, wspace=.3)

    for col, control in ((0, False), (3, True)):
        M, S, n, T, rs = data[control]
        ax = fig.add_subplot(gs[0, col:col + 3])
        for m in M:
            ax.plot(STEPS, m, color="#4a7fb5", lw=.7, alpha=.35)
        ax.plot(STEPS, M.mean(0), color="#c0392b", lw=2, label="mean of clusters")
        ax.plot(STEPS, c, color="black", lw=2.2, ls="--", label=f"{a.task} task curve")
        ax.set_xscale("log"); ax.set_xticks([1, 8, 68, 256])
        ax.set_xticklabels(["1k", "8k", "68k", "256k"])
        ax.axhline(0, color="#ddd", lw=.6); ax.set_ylim(-2.6, 2.6)
        ax.set_title(f"{'random control' if control else 'shape group (r>0.88)'}: "
                     f"{len(M)} cluster mean curves\n"
                     f"r(cluster,task) {rs.min():.2f}–{rs.max():.2f}", fontsize=10)
        ax.set_xlabel("checkpoint"); ax.set_ylabel("z-normed prob")
        ax.legend(fontsize=8, loc="lower right")

    M, S, n, T, rs = data[False]
    order = np.argsort(-n)[:a.n_small]
    for i, ci in enumerate(order):
        ax = fig.add_subplot(gs[1 + i // 6, i % 6])
        ax.fill_between(STEPS, M[ci] - S[ci], M[ci] + S[ci], color="#4a7fb5", alpha=.18)
        ax.plot(STEPS, M[ci], color="#4a7fb5", lw=1.6)
        ax.plot(STEPS, c, color="black", lw=1.2, ls="--")
        ax.set_xscale("log"); ax.set_xticks([1, 68, 256]); ax.set_xticklabels(["1k", "68k", "256k"], fontsize=6)
        ax.tick_params(labelsize=6); ax.set_ylim(-2.6, 2.6)
        ax.set_title(f"C{ci} n={n[ci]} r={rs[ci]:.2f}", fontsize=7)

    fig.suptitle(f"Content-cluster mean curves vs the {a.task} curve "
                 f"(top {a.top_n:,} of 36.7M tokens, K={a.k})", fontsize=13)
    out = OUT_DIR / f"{a.task}_cluster_curves_k{a.k}.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
