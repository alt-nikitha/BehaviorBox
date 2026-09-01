"""Cluster the highest-Kendall-tau tokens by content, and show each cluster's tokens,
contexts, and trajectories against the task curve.

Selection is by tau (rank agreement with the task's checkpoint ordering), not Pearson r
-- see kendall_shape_match.py for why r is degenerate here.

For each cluster the report shows:
  * mean tau, and what fraction of the cluster are strict ordering matches
  * an SVG with the task curve, the cluster mean, AND every displayed token's own curve
    drawn faintly -- the individual curves matter because averaging thousands of tokens
    washes out minority shapes (a mistake made earlier in this analysis)
  * top tokens by tau with +/-10 words of context and a per-token sparkline

Usage:
  python cluster_tau_tokens.py --task medmcqa --top-n 200000 --k 100 --per-cluster 12
"""
import argparse, html, json, math, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, OUT_DIR, STEPS, open_memmap
from kendall_shape_match import raw_task_perf

WORD_IDS = CACHE_DIR / "word_ids.pkl"
INPUT_FEATURES = "/data/user_data/nsrikant/bbox_data/output/olmo_256000_unseen/input_features"
LOGX = np.log(np.array(STEPS, dtype=float))


def svg(curves, mean, task, w=300, h=90, pad=6):
    """Inline SVG: faint per-token curves, cluster mean, task curve."""
    xs = (LOGX - LOGX.min()) / (LOGX.max() - LOGX.min()) * (w - 2 * pad) + pad

    def path(y, lo=-2.5, hi=2.5):
        yy = np.clip(y, lo, hi)
        py = h - pad - (yy - lo) / (hi - lo) * (h - 2 * pad)
        return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, py))

    mid = h - pad - (0 - -2.5) / 5.0 * (h - 2 * pad)
    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             f'<line x1="{pad}" y1="{mid:.1f}" x2="{w-pad}" y2="{mid:.1f}" '
             f'stroke="#e5e5e5" stroke-width="1"/>']
    for c in curves:
        parts.append(f'<path d="{path(c)}" fill="none" stroke="#4a7fb5" '
                     f'stroke-width="0.8" opacity="0.28"/>')
    parts.append(f'<path d="{path(mean)}" fill="none" stroke="#1f5c99" stroke-width="2.2"/>')
    parts.append(f'<path d="{path(task)}" fill="none" stroke="#111" stroke-width="2" '
                 f'stroke-dasharray="5,3"/>')
    parts.append("</svg>")
    return "".join(parts)


def spark(y, w=110, h=26, pad=3):
    xs = (LOGX - LOGX.min()) / (LOGX.max() - LOGX.min()) * (w - 2 * pad) + pad
    yy = np.clip(y, -2.5, 2.5)
    py = h - pad - (yy + 2.5) / 5.0 * (h - 2 * pad)
    d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, py))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<path d="{d}" fill="none" stroke="#4a7fb5" stroke-width="1.4"/></svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--top-n", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--per-cluster", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    z = np.load(CACHE_DIR / f"kendall_{a.task}.npz")
    tau, strict = z["tau"], z["strict"]
    rows = np.argpartition(-tau, a.top_n)[:a.top_n]
    rows = np.sort(rows)
    print(f"top {a.top_n:,} by tau: {tau[rows].min():.3f}..{tau[rows].max():.3f}; "
          f"{strict[rows].sum():,} are strict ordering matches")

    mm, edim, n_out = open_memmap()
    E = np.empty((len(rows), edim), np.float32)
    T = np.empty((len(rows), n_out), np.float32)
    for s in range(0, len(rows), 20000):
        e = min(s + 20000, len(rows))
        blk = np.asarray(mm[rows[s:e]], dtype=np.float32)
        E[s:e] = blk[:, :edim]
        p = blk[:, -n_out:]
        T[s:e] = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-9)
    print("clustering ...")
    lab = MiniBatchKMeans(n_clusters=a.k, random_state=a.seed, batch_size=4096,
                          n_init=3, max_iter=200).fit(E).labels_

    perf = raw_task_perf(a.task)
    task_z = (perf - perf.mean()) / perf.std()

    picks = {}
    for c in range(a.k):
        idx = np.where(lab == c)[0]
        if len(idx):
            picks[c] = idx[np.argsort(-tau[rows[idx]])[:a.per_cluster]]

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    need = np.concatenate(list(picks.values()))
    print(f"resolving {len(need):,} contexts ...")
    wic = get_words_in_context(INPUT_FEATURES, [str(w) for w in wids[rows[need]]], N=10)

    clusters = []
    for c, idx in picks.items():
        m = lab == c
        samples = []
        for i in idx:
            wid = str(wids[rows[i]])
            e = wic.get(wid)
            if e:
                samples.append({"word_id": wid, **e, "tau": float(tau[rows[i]]),
                                "curve": T[i], "strict": bool(strict[rows[i]])})
        clusters.append({"id": int(c), "n": int(m.sum()), "mean": T[m].mean(0),
                         "tau": float(tau[rows[m]].mean()),
                         "strict_frac": float(strict[rows[m]].mean()), "samples": samples})
    clusters.sort(key=lambda x: -x["tau"])

    esc = html.escape
    out = [f"<title>{esc(a.task)}: content clusters of high-tau tokens</title>", """<style>
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a;max-width:1180px}
h1{font-size:20px;margin:0 0 4px}.meta{color:#666;font-size:13px;margin-bottom:20px}
.c{background:#fff;border:1px solid #e3e3e3;border-radius:8px;margin-bottom:16px;padding:14px 16px;
display:grid;grid-template-columns:320px 1fr;gap:16px;align-items:start}
.ch{font-weight:600;margin-bottom:6px}.ch .n{color:#888;font-weight:400;font-size:12px}
.row{display:grid;grid-template-columns:118px 1fr;gap:10px;align-items:center;
border-top:1px solid #f0f0f0;padding:4px 0}
.tok{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}
.w{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}
.ctx{color:#777}.tag{color:#999;font-size:11px}.s{color:#1f5c99;font-weight:600}
.legend{font-size:11px;color:#888;margin-top:4px}
</style>"""]
    out.append(f"<h1>{esc(a.task)}: content clusters among the highest-tau tokens</h1>")
    out.append(f'<div class="meta">top {a.top_n:,} of 36.7M tokens by Kendall tau '
               f'&middot; K={a.k} &middot; clusters sorted by mean tau &middot; '
               f'solid blue = cluster mean, faint blue = the individual tokens shown, '
               f'dashed black = {esc(a.task)} performance</div>')
    for cl in clusters:
        out.append('<div class="c"><div>')
        out.append(f'<div class="ch">C{cl["id"]} <span class="n">n={cl["n"]:,} &middot; '
                   f'mean &tau;={cl["tau"]:.3f} &middot; {100*cl["strict_frac"]:.1f}% strict</span></div>')
        out.append(svg([s["curve"] for s in cl["samples"]], cl["mean"], task_z))
        out.append('<div class="legend">1k &rarr; 256k (log)</div></div><div>')
        for s in cl["samples"]:
            out.append('<div class="row"><div>' + spark(s["curve"]) +
                       f'<div class="tag">&tau;={s["tau"]:.2f}'
                       + (' <span class="s">strict</span>' if s["strict"] else '') + '</div></div>')
            out.append(f'<div class="tok"><span class="ctx">{esc(s["before"])}</span>'
                       f'<span class="w">{esc(s["word"])}</span>'
                       f'<span class="ctx">{esc(s["after"])}</span></div></div>')
        out.append("</div></div>")

    fp = OUT_DIR / f"{a.task}_tau_clusters_k{a.k}.html"
    fp.write_text("\n".join(out))
    print(f"wrote {fp}")
    for cl in clusters[:15]:
        print(f"  C{cl['id']:<3} n={cl['n']:<6} tau={cl['tau']:.3f} "
              f"strict={100*cl['strict_frac']:4.1f}%  "
              + ", ".join(s["word"].strip()[:14] for s in cl["samples"][:8]))


if __name__ == "__main__":
    main()
