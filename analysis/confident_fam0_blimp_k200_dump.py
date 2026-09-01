"""Same confidently-correct fam0 / fam2-blimp token pools as confident_fam0_blimp_llm_cluster.py,
but clustered at a FIXED K=200 on the full 768-d embedding (no elbow, no LLM labeling -- just a
raw dump), with a wider +/-20 word context window per token (vs the 10-word default used
elsewhere this session). Output: one HTML, two sections (fam0 | fam2-blimp), each cluster shown
as a block with every member token highlighted in its context, ordered nearest-centroid-first,
clusters ordered largest-first.

Loads the saved clf+scaler (no retraining), reproduces the same exclusive sample/split.

Usage:
  python confident_fam0_blimp_k200_dump.py --model fam0_logreg_weights.pkl --n-confident 2000 --k 200
"""
import argparse, html, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DIR = Path(__file__).resolve().parent
SIDE_COLOR = {0: "#2a78d6", 2: "#1baf7a"}


def render_html(side_clusters, out_path, context_words, max_show):
    esc = html.escape
    parts = [f"""<title>K=200 clusters: confident fam0 vs confident blimp</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px;position:sticky;top:0;
background:#fafafa;padding:8px 0}}
.meta-top{{color:#666;font-size:13px;margin-bottom:16px}}
.twocol{{display:flex;gap:20px;align-items:flex-start}}
.side{{flex:1;min-width:0}}
.cluster{{background:#fff;border:1px solid #e3e3e3;border-radius:6px;margin-bottom:10px;padding:10px 14px}}
.chead{{font-size:12.5px;color:#555;font-weight:600;margin-bottom:6px}}
.tok{{font-family:ui-monospace,Menlo,monospace;font-size:12px;padding:2px 0;
border-top:1px solid #f0f0f0;white-space:pre-wrap;word-break:break-word}}
.tok:first-of-type{{border-top:none}}
.before,.after{{color:#777}}
.w{{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}}
</style>
<h1>K=200 clusters on full 768-d embedding, +/-{context_words} word context</h1>
<div class="meta-top">Confidently AND correctly classified tokens only (true label == predicted
label, ranked by predicted probability). Two families side by side; clusters ordered
largest-first within each.</div>
<div class="twocol">
"""]
    for c, clusters in side_clusters.items():
        parts.append('<div class="side">')
        parts.append(f'<h2 style="color:{SIDE_COLOR[c]}">fam{c} ({esc(FAM_LABELS.get(c, str(c)))}) '
                     f'-- {len(clusters)} clusters</h2>')
        for rank, cl in enumerate(clusters, 1):
            shown = cl["examples"][:max_show]
            parts.append(f'<div class="cluster"><div class="chead">cluster {rank}/{len(clusters)} '
                         f'n={cl["n"]} (showing {len(shown)})</div>')
            for e in shown:
                parts.append(f'<div class="tok">'
                             f'<span class="before">{esc(e["before"])}</span>'
                             f'<span class="w">{esc(e["word"])}</span>'
                             f'<span class="after">{esc(e["after"])}</span></div>')
            parts.append('</div>')
        parts.append('</div>')
    parts.append('</div>')
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--n-confident", type=int, default=2000)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--context-words", type=int, default=20)
    ap.add_argument("--max-show", type=int, default=10, help="max members shown per cluster")
    ap.add_argument("--out", default="confident_fam0_blimp_k200_dump.html")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}")

    fam_names, F, members = build_families(0.45)
    R, D_, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D_, valid, sargs["min_r"], sargs["max_l1"],
                                     exclude=set(sargs["exclude_family"]))
    rows, labels_sel = sample_equal_exclusive(mask, labels, sargs["per_family"], sargs["seed"], fam_names)
    print(f"sampled {len(rows):,} tokens total (reproducing saved split); reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=sargs["seed"], stratify=labels_sel)
    X_te = scaler.transform(X[idx_te])
    y_te = labels_sel[idx_te]
    rows_te = rows[idx_te]

    proba = clf.predict_proba(X_te)
    pred = np.array(classes)[np.argmax(proba, axis=1)]
    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))

    side_clusters = {}
    for target_class in [0, 2]:
        col = classes.index(target_class)
        m = (y_te == target_class) & (pred == target_class)
        cand = np.where(m)[0]
        cand = cand[np.argsort(-proba[cand, col])[: a.n_confident]]
        print(f"\nfam{target_class}: {m.sum():,} confidently+correctly classified; "
              f"using top {len(cand):,} by p (range {proba[cand, col].min():.3f}-"
              f"{proba[cand, col].max():.3f})")

        X_side = X_te[cand]
        k = min(a.k, len(cand))
        print(f"  KMeans K={k} on {len(cand):,} points (full 768-d embedding)...")
        km = KMeans(n_clusters=k, random_state=a.seed, n_init=10).fit(X_side)
        lbl = km.labels_
        dist = np.linalg.norm(X_side - km.cluster_centers_[lbl], axis=1)
        sizes = np.bincount(lbl, minlength=k)
        order = np.argsort(-sizes)

        word_ids_side = [str(wids_all[rows_te[i]]) for i in cand]
        print(f"  fetching +/-{a.context_words}w context for {len(word_ids_side):,} tokens...")
        ctx = get_words_in_context(INPUT_FEATURES, word_ids_side, N=a.context_words)

        clusters = []
        for cl in order:
            mm = np.where(lbl == cl)[0]
            m_sorted = mm[np.argsort(dist[mm])]
            examples = [ctx.get(word_ids_side[i], {"before": "?", "word": "?", "after": "?"})
                       for i in m_sorted]
            clusters.append({"n": len(mm), "examples": examples})
        side_clusters[target_class] = clusters

    render_html(side_clusters, OUT_DIR / a.out, a.context_words, a.max_show)


if __name__ == "__main__":
    main()
