"""Cluster the (already-sampled) exclusive train+test tokens in the logistic regression's
CLASSIFIER-WEIGHTED embedding space, not raw Longformer similarity — scale each of the 768
scaled-embedding dims by |coef_[target_family]| before KMeans, so clusters form around
whatever the model actually uses to separate families, not generic semantic similarity
(plain unweighted KMeans on raw embeddings was already tried in family_semantic_clusters.py
and found no coherent theme).

Reuses the SAME exclusive sample/split/model as predict_family_from_embedding_highconf.py
(r>0.85, L1<0.3, medmcqa excluded, seed=42 by default) — no corpus scan, train+test tokens
only, per the user's request. K=100 clusters, MiniBatchKMeans, same convention as
task_shape_content_clusters.py (nearest-to-centroid samples + word-context lookup). Each
cluster also reports its TRUE family composition, to see whether classifier-weighted
clusters line up with family boundaries or cut across them.

Usage:
  python weighted_cluster_logreg.py --target-family 0 --k 100
"""
import argparse, html, json, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DIR = Path(__file__).resolve().parent


def render_html(args, clusters, out):
    esc = html.escape
    parts = [f"<title>Weighted clusters (fam{args.target_family}, K={args.k})</title>", """<style>
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a;max-width:1100px}
h1{font-size:20px;margin:0 0 4px} .meta{color:#666;font-size:13px;margin-bottom:20px}
.c{background:#fff;border:1px solid #e3e3e3;border-radius:8px;margin-bottom:14px;padding:12px 16px}
.ch{font-weight:600;margin-bottom:8px} .ch .n{color:#888;font-weight:400}
.comp{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#555;margin-bottom:6px}
.tok{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;padding:3px 0;
border-top:1px solid #f0f0f0;white-space:pre-wrap;word-break:break-word}
.w{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}
.ctx{color:#777} .top{color:#444;font-size:13px;margin-bottom:6px}
</style>"""]
    parts.append(f"<h1>Classifier-weighted clusters &mdash; fam{args.target_family} "
                 f"({esc(FAM_LABELS.get(args.target_family, str(args.target_family)))}) "
                 f"direction, K={args.k}</h1>")
    parts.append(f'<div class="meta">{args.k} clusters over {args.n_total:,} exclusive '
                 f'train+test tokens, embeddings scaled by |logreg coef| before KMeans '
                 f'&middot; {args.per_cluster} most prototypical tokens each</div>')
    for cl in clusters:
        parts.append('<div class="c">')
        parts.append(f'<div class="ch">C{cl["id"]} <span class="n">n={cl["n"]:,} '
                     f'({cl["pct"]:.1f}%)</span></div>')
        parts.append(f'<div class="comp">true-family mix: {esc(cl["family_mix_str"])}</div>')
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
    ap.add_argument("--min-r", type=float, default=0.85)
    ap.add_argument("--max-l1", type=float, default=0.3)
    ap.add_argument("--per-family", type=int, default=548_236)
    ap.add_argument("--exclude-family", type=int, nargs="*", default=[3])
    ap.add_argument("--target-family", type=int, default=0)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--per-cluster", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fam_names, F, members = build_families(0.45)
    R, D, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D, valid, a.min_r, a.max_l1, exclude=set(a.exclude_family))
    rows, labels_sel = sample_equal_exclusive(mask, labels, a.per_family, a.seed, fam_names)
    print(f"sampled {len(rows):,} tokens total; reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=a.seed, stratify=labels_sel)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = labels_sel[idx_tr], labels_sel[idx_te]

    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    print(f"training logistic regression on {len(y_tr):,} rows...")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    classes = list(clf.classes_)
    w = np.abs(clf.coef_[classes.index(a.target_family)])  # (768,)
    print(f"weight magnitude: min={w.min():.4f} median={np.median(w):.4f} max={w.max():.4f} "
          f"top-10-dims-share-of-total={np.sort(w)[-10:].sum()/w.sum():.3f}")

    # train + test tokens ONLY (no corpus scan), same order as rows/labels_sel via idx
    X_all = np.concatenate([X_tr, X_te], axis=0)
    labels_all = np.concatenate([y_tr, y_te], axis=0)
    rows_all = rows[np.concatenate([idx_tr, idx_te])]
    X_weighted = X_all * w  # scale each scaled-embedding dim by the classifier's |weight|

    print(f"clustering K={a.k} on classifier-weighted embeddings ({X_weighted.shape[0]:,} tokens)...")
    km = MiniBatchKMeans(n_clusters=a.k, random_state=a.seed, batch_size=4096,
                          n_init=3, max_iter=200).fit(X_weighted)
    km_labels = km.labels_
    d = np.linalg.norm(X_weighted - km.cluster_centers_[km_labels], axis=1)

    sizes = np.bincount(km_labels, minlength=a.k)
    print(f"cluster sizes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()} "
          f"CV={sizes.std()/sizes.mean():.3f}")

    picks = {}
    for c in range(a.k):
        idxs = np.where(km_labels == c)[0]
        if len(idxs) == 0:
            continue
        picks[c] = idxs[np.argsort(d[idxs])[: a.per_cluster]]

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    need = np.concatenate(list(picks.values()))
    need_wids = [str(w_) for w_ in wids[rows_all[need]]]
    print(f"resolving {len(need_wids):,} contexts...")
    wic = get_words_in_context(INPUT_FEATURES, need_wids, N=10)

    clusters = []
    for c, idxs in picks.items():
        member_mask = km_labels == c
        fam_counts = {f: int((labels_all[member_mask] == f).sum())
                      for f in sorted(set(labels_all.tolist()))}
        n_c = int(member_mask.sum())
        mix_str = " / ".join(f"fam{f}={100*cnt/n_c:.0f}%" for f, cnt in fam_counts.items())
        samples, words = [], []
        for i in idxs:
            wid = str(wids[rows_all[i]])
            e = wic.get(wid)
            if not e:
                continue
            samples.append({"word_id": wid, **e})
            words.append(e["word"].strip())
        clusters.append({
            "id": int(c), "n": n_c, "pct": 100.0 * n_c / len(km_labels),
            "family_mix": fam_counts, "family_mix_str": mix_str,
            "top_words": words, "samples": samples,
        })
    clusters.sort(key=lambda x: -x["n"])

    a.n_total = len(km_labels)
    (OUT_DIR / f"weighted_clusters_fam{a.target_family}_k{a.k}.json").write_text(
        json.dumps(clusters, indent=1))
    render_html(a, clusters, OUT_DIR / f"weighted_clusters_fam{a.target_family}_k{a.k}.html")

    print(f"\ntop 20 largest clusters:")
    for cl in clusters[:20]:
        print(f"  C{cl['id']:<3} n={cl['n']:<7} [{cl['family_mix_str']}]  "
              + ", ".join(cl["top_words"][:8]))


if __name__ == "__main__":
    main()
