"""Full-corpus shape->content pass: match ALL 36.7M tokens to a task curve, take the
best-matching ones, cluster them on Longformer content, and dump top tokens in context.

Stage 1  pearson r of every token's z-normed 11-checkpoint curve vs the task z-curve.
         Reads the 57GB memmap once (~2 min) and caches r to disk per task.
Stage 2  take the top-N tokens by r (the "shape group").
Stage 3  MiniBatchKMeans (default K=100) on their 768-d Longformer embeddings.
Stage 4  per cluster, the most prototypical tokens (nearest the centroid) with their
         left/right context, written to an HTML report + a JSON dump.

Usage:
  python task_shape_content_clusters.py --task medmcqa --top-n 200000 --k 100
  python task_shape_content_clusters.py --task gsm8k --top-n 200000 --k 100 --per-cluster 15
  python task_shape_content_clusters.py --task medmcqa --r-only     # just cache r + stats
"""
import argparse, html, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from sample_by_task_centroids import DEFAULT_STEM, build_task_centroids

CACHE_DIR = Path("/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen")
MEMMAP_INFO = CACHE_DIR / "ofw=0.5_pznorm=0.01/cached_data_info.json"
WORD_IDS = CACHE_DIR / "word_ids.pkl"
INPUT_FEATURES = "/data/user_data/nsrikant/bbox_data/output/olmo_256000_unseen/input_features"
OUT_DIR = Path(__file__).resolve().parent / "shape_content"
STEPS = [1, 2, 4, 8, 16, 32, 68, 103, 154, 205, 256]


def open_memmap():
    info = json.load(open(MEMMAP_INFO))
    mm = np.memmap(info["filename"], dtype=np.float16, mode="r", shape=tuple(info["shape"]))
    return mm, int(info["embedding_dim"]), int(info["output_feature_dim"])


def task_curve(task):
    cents = build_task_centroids(DEFAULT_STEM, len(STEPS))
    if task not in cents:
        raise SystemExit(f"no complete curve for {task}; have: {sorted(cents)}")
    c = cents[task].astype(np.float64)
    return (c - c.mean()) / c.std()


def corr_all(task, chunk=2_000_000):
    """Pearson r of every row's z-curve with the task curve. Cached per task."""
    cache = CACHE_DIR / f"taskcorr_{task}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"loaded cached r from {cache}")
        return z["r"], z["valid"]
    c = task_curve(task)
    mm, edim, n_out = open_memmap()
    N = mm.shape[0]
    r = np.empty(N, np.float32)
    valid = np.zeros(N, bool)
    t = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -n_out:], dtype=np.float32)
        mu = b.mean(1, keepdims=True)
        sd = b.std(1, keepdims=True)
        z = (b - mu) / (sd + 1e-9)
        r[s:e] = (z @ c) / n_out
        valid[s:e] = sd[:, 0] > 1e-6
        print(f"  {e:,}/{N:,} ({time.time()-t:.0f}s)", flush=True)
    r[~valid] = -np.inf
    np.savez(cache, r=r, valid=valid)
    print(f"cached -> {cache}")
    return r, valid


def read_embeddings(rows, edim, batch=20000):
    """Gather the 768-d content block for `rows` (sorted for locality)."""
    mm, _, _ = open_memmap()
    E = np.empty((len(rows), edim), np.float32)
    t = time.time()
    for s in range(0, len(rows), batch):
        e = min(s + batch, len(rows))
        E[s:e] = np.asarray(mm[rows[s:e], :edim], dtype=np.float32)
        if (s // batch) % 20 == 0:
            print(f"  emb {e:,}/{len(rows):,} ({time.time()-t:.0f}s)", flush=True)
    return E


def render_html(task, args, clusters, out):
    esc = html.escape
    parts = [f"<title>{esc(task)} shape &rarr; content clusters</title>", """<style>
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a;max-width:1100px}
h1{font-size:20px;margin:0 0 4px} .meta{color:#666;font-size:13px;margin-bottom:20px}
.c{background:#fff;border:1px solid #e3e3e3;border-radius:8px;margin-bottom:14px;padding:12px 16px}
.ch{font-weight:600;margin-bottom:8px} .ch .n{color:#888;font-weight:400}
.tok{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;padding:3px 0;
border-top:1px solid #f0f0f0;white-space:pre-wrap;word-break:break-word}
.w{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}
.ctx{color:#777} .top{color:#444;font-size:13px;margin-bottom:6px}
</style>"""]
    parts.append(f"<h1>{esc(task)}: content clusters within the shape-matched group</h1>")
    parts.append(f'<div class="meta">top {args.top_n:,} of 36.7M tokens by curve correlation '
                 f'&middot; K={args.k} &middot; {args.per_cluster} most prototypical tokens each</div>')
    for cl in clusters:
        parts.append('<div class="c">')
        parts.append(f'<div class="ch">C{cl["id"]} <span class="n">n={cl["n"]:,} '
                     f'({cl["pct"]:.1f}%) &middot; mean r={cl["mean_r"]:.3f}</span></div>')
        parts.append(f'<div class="top">{esc(", ".join(cl["top_words"]))}</div>')
        for s in cl["samples"]:
            parts.append(f'<div class="tok"><span class="ctx">{esc(s["before"])}</span>'
                         f'<span class="w">{esc(s["word"])}</span>'
                         f'<span class="ctx">{esc(s["after"])}</span></div>')
        parts.append("</div>")
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--top-n", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--per-cluster", type=int, default=12, help="tokens shown per cluster")
    ap.add_argument("--r-only", action="store_true", help="cache correlations and exit")
    ap.add_argument("--random-control", action="store_true",
                    help="cluster a random token sample instead of the shape group")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    r, valid = corr_all(a.task)
    fin = r[np.isfinite(r)]
    qs = np.percentile(fin, [50, 90, 99, 99.9])
    print(f"\nvalid={valid.sum():,}/{len(r):,}  r median={qs[0]:.3f} p90={qs[1]:.3f} "
          f"p99={qs[2]:.3f} p99.9={qs[3]:.3f} max={fin.max():.3f}")
    if a.r_only:
        return

    if a.random_control:
        rows = np.random.default_rng(a.seed).choice(np.where(valid)[0], a.top_n, replace=False)
        print(f"RANDOM CONTROL: {a.top_n:,} rows, r {r[rows].min():.3f}..{r[rows].max():.3f}")
    else:
        rows = np.argpartition(-r, a.top_n)[:a.top_n]
        rows = rows[np.argsort(-r[rows])]
        print(f"shape group: {a.top_n:,} rows, r {r[rows].min():.3f}..{r[rows].max():.3f}")
    rows_sorted = np.sort(rows)

    mm, edim, _ = open_memmap()
    E = read_embeddings(rows_sorted, edim)

    print(f"clustering K={a.k} ...")
    km = MiniBatchKMeans(n_clusters=a.k, random_state=a.seed, batch_size=4096,
                         n_init=3, max_iter=200).fit(E)
    labels = km.labels_
    d = np.linalg.norm(E - km.cluster_centers_[labels], axis=1)

    # pick prototypical rows per cluster, then resolve all their contexts in one pass
    picks = {}
    for c in range(a.k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        picks[c] = idx[np.argsort(d[idx])[:a.per_cluster]]

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    need = np.concatenate(list(picks.values()))
    need_wids = [str(w) for w in wids[rows_sorted[need]]]
    print(f"resolving {len(need_wids):,} contexts ...")
    wic = get_words_in_context(INPUT_FEATURES, need_wids, N=10)

    clusters = []
    for c, idx in picks.items():
        samples, words = [], []
        for i in idx:
            w = str(wids[rows_sorted[i]])
            e = wic.get(w)
            if not e:
                continue
            samples.append({"word_id": w, **e})
            words.append(e["word"].strip())
        clusters.append({
            "id": int(c), "n": int((labels == c).sum()),
            "pct": 100.0 * (labels == c).sum() / len(labels),
            "mean_r": float(r[rows_sorted[labels == c]].mean()),
            "top_words": words, "samples": samples,
        })
    clusters.sort(key=lambda x: -x["n"])

    sizes = np.bincount(labels, minlength=a.k)
    print(f"\ncluster sizes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()} "
          f"CV={sizes.std()/sizes.mean():.3f}  (CV~0 => embeddings are one blob, no content "
          f"structure for KMeans to find)")
    print(f"mean dist to own centroid={d.mean():.2f}  vs mean dist to global centroid="
          f"{np.linalg.norm(E - E.mean(0), axis=1).mean():.2f}")

    stem = f"{a.task}_top{a.top_n}_k{a.k}" + ("_randctrl" if a.random_control else "")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(clusters, indent=1))
    render_html(a.task, a, clusters, OUT_DIR / f"{stem}.html")
    for cl in clusters[:15]:
        print(f"  C{cl['id']:<3} n={cl['n']:<6} r={cl['mean_r']:.3f}  "
              + ", ".join(cl["top_words"][:8]))


if __name__ == "__main__":
    main()
