"""Cluster tokens WITHIN each family separately on their (d1, d2) coordinates from the
non-redundant 2D weight-space projection (d1 = (w_fam1-w_fam0).x, d2 = (w_fam2-w_fam0).x, see
logreg_three_class_projection.py). K per family is chosen automatically via the elbow method
(max-distance-from-chord on the inertia curve) unless overridden with --k. Produces one
self-contained HTML report: an elbow-curve plot (all 3 families, chosen K marked) followed by,
for each family, one column per cluster listing EVERY member token's context with the target
word highlighted (not just centroid-nearest examples), ordered nearest-centroid-first.

Loads the saved clf+scaler (no retraining), reproduces the same exclusive sample/split.

Usage:
  python logreg_dim_cluster_per_family.py --model fam0_logreg_weights.pkl --n-per-family 2000
"""
import argparse, base64, html, io, pickle, re, sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}  # dataviz reference palette slots 1-3
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize_window(e):
    text = e["before"] + " " + e["word"] + " " + e["after"]
    return [t.lower() for t in TOKEN_RE.findall(text)]


def label_clusters(examples_by_cluster, min_count=5, top_n=4, k=0.5):
    """Automatic per-cluster label: top terms by log-odds ratio of this cluster's context
    vocabulary vs the POOLED vocabulary of every other cluster in the same family. Returns
    {cluster_idx: [(term, lor, count_in_cluster, count_in_rest), ...]}."""
    counts = {}
    totals = {}
    for ci, exs in examples_by_cluster.items():
        cnt = Counter()
        tot = 0
        for e in exs:
            toks = tokenize_window(e)
            cnt.update(toks)
            tot += len(toks)
        counts[ci] = cnt
        totals[ci] = tot

    labels = {}
    for ci in examples_by_cluster:
        rest_counts = Counter()
        rest_total = 0
        for cj, cnt in counts.items():
            if cj == ci:
                continue
            rest_counts.update(cnt)
            rest_total += totals[cj]
        rows = []
        for term, cnt_in in counts[ci].items():
            cnt_out = rest_counts.get(term, 0)
            if cnt_in + cnt_out < min_count:
                continue
            lor = (np.log(cnt_in + k) - np.log(totals[ci] - cnt_in + k)) - \
                  (np.log(cnt_out + k) - np.log(rest_total - cnt_out + k))
            rows.append((term, lor, cnt_in, cnt_out))
        rows.sort(key=lambda r: -r[1])
        labels[ci] = rows[:top_n]
    return labels


def find_knee(ks, inertias):
    ks = np.asarray(ks, float)
    inertias = np.asarray(inertias, float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (inertias - inertias.min()) / (inertias.max() - inertias.min() + 1e-12)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = p2 - p1
    line_norm = line / np.linalg.norm(line)
    dists = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi]) - p1
        proj = np.dot(p, line_norm) * line_norm
        dists.append(np.linalg.norm(p - proj))
    return int(ks[int(np.argmax(dists))])


def elbow_plot_b64(elbow_data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c, (ks, inertias, knee) in elbow_data.items():
        ax.plot(ks, inertias, "o-", color=COLORS[c], label=f"fam{c} ({FAM_LABELS.get(c, c)})",
                markersize=5)
        ki = ks.index(knee)
        ax.plot(knee, inertias[ki], "o", color=COLORS[c], markersize=12,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
    ax.set_xlabel("K")
    ax.set_ylabel("inertia")
    ax.set_title("Elbow curves per family (marked point = chosen K)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(elbow_png_b64, family_clusters, out_path, max_show):
    esc = html.escape
    parts = [f"""<title>Per-family clustering on (d1,d2) weight-space projection</title>
<style>
body{{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;
background:#fafafa;color:#1a1a1a}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px}}
.meta-top{{color:#666;font-size:13px;margin-bottom:16px}}
.elbow{{max-width:700px;display:block;margin-bottom:8px}}
.cols{{display:flex;gap:14px;overflow-x:auto;padding-bottom:8px}}
.col{{flex:1;min-width:260px;max-width:340px}}
.col h3{{font-size:13px;margin:0 0 8px;position:sticky;top:0;background:#fafafa;padding:6px 0;
color:#333}}
.tok{{background:#fff;border:1px solid #e3e3e3;border-radius:6px;margin-bottom:6px;padding:6px 8px}}
.ctx{{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}}
.before,.after{{color:#777}}
.w{{background:#ffe9a8;font-weight:600;padding:0 2px;border-radius:2px}}
.label{{font-weight:400;color:#a05a00;font-size:11.5px}}
</style>
<h1>Per-family clustering on the non-redundant (d1, d2) weight-space projection</h1>
<div class="meta-top">K chosen per family via elbow method (max distance from chord); clusters
ordered left-to-right by centroid position along the d1+d2 axis; up to {max_show} nearest-
to-centroid tokens shown per cluster (cluster n = true size). Label = top terms by log-odds
ratio of this cluster's context vocabulary vs every OTHER cluster in the same family
(automatic, min count 5; term(log-odds)) &mdash; empty means nothing cleared the count
threshold, i.e. no distinguishing vocabulary was found.</div>
<img class="elbow" src="data:image/png;base64,{elbow_png_b64}">
"""]
    for c, clusters in family_clusters.items():
        parts.append(f'<h2>family {c} &mdash; {esc(FAM_LABELS.get(c, str(c)))}</h2>')
        parts.append('<div class="cols">')
        for rank, cl in enumerate(clusters, 1):
            shown = cl["examples"][:max_show]
            parts.append(f'<div class="col"><h3>cluster {rank}/{len(clusters)} '
                         f'n={cl["n"]:,} (showing {len(shown)}) '
                         f'centroid=({cl["cx"]:+.2f},{cl["cy"]:+.2f})<br>'
                         f'<span class="label">{esc(cl["label"])}</span></h3>')
            for e in shown:
                parts.append(f'<div class="tok"><div class="ctx">'
                             f'<span class="before">{esc(e["before"])}</span>'
                             f'<span class="w">{esc(e["word"])}</span>'
                             f'<span class="after">{esc(e["after"])}</span></div></div>')
            parts.append('</div>')
        parts.append('</div>')
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--ref-class", type=int, default=0)
    ap.add_argument("--k", type=int, default=None, help="override auto-elbow K for all families")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-per-family", type=int, default=2000, help="cap tokens clustered per family")
    ap.add_argument("--max-show", type=int, default=20, help="max tokens shown per cluster")
    ap.add_argument("--context-words", type=int, default=10)
    ap.add_argument("--out", default="logreg_dim_cluster_per_family.html")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}, classes={classes}")

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

    S = clf.decision_function(X_te)
    ref_col = classes.index(a.ref_class)
    other_cols = [c for c in range(len(classes)) if c != ref_col]
    other_classes = [classes[c] for c in other_cols]
    Dproj = S[:, other_cols] - S[:, [ref_col]]  # (N, 2)
    print(f"reference class: fam{a.ref_class}  ->  d1=fam{other_classes[0]}-fam{a.ref_class}, "
          f"d2=fam{other_classes[1]}-fam{a.ref_class}")

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    rng = np.random.default_rng(a.seed)
    ks_range = list(range(a.k_min, a.k_max + 1))

    elbow_data = {}
    chosen_k = {}
    cand_per_fam = {}
    for c in classes:
        cand = np.where(y_te == c)[0]
        if a.n_per_family < len(cand):
            cand = rng.choice(cand, a.n_per_family, replace=False)
        cand_per_fam[c] = cand
        Dc = Dproj[cand]
        inertias = []
        for k in ks_range:
            km = KMeans(n_clusters=k, random_state=a.seed, n_init=5).fit(Dc)
            inertias.append(km.inertia_)
        knee = a.k if a.k is not None else find_knee(ks_range, inertias)
        elbow_data[c] = (ks_range, inertias, knee)
        chosen_k[c] = knee
        print(f"family {c}: inertias={[f'{v:.0f}' for v in inertias]}  chosen K={knee}")

    elbow_png = elbow_plot_b64(elbow_data)

    family_clusters = {}
    for c in classes:
        cand = cand_per_fam[c]
        Dc = Dproj[cand]
        k = chosen_k[c]
        km = KMeans(n_clusters=k, random_state=a.seed, n_init=10).fit(Dc)
        lbl = km.labels_
        dist = np.linalg.norm(Dc - km.cluster_centers_[lbl], axis=1)
        order = np.argsort(km.cluster_centers_[:, 0] + km.cluster_centers_[:, 1])

        all_wids = [str(wids_all[rows_te[cand[i]]]) for i in range(len(cand))]
        print(f"fetching context for family {c}: {len(all_wids):,} tokens...")
        ctx = get_words_in_context(INPUT_FEATURES, all_wids, N=a.context_words)

        examples_by_cluster = {}
        cluster_meta = {}
        for cl in order:
            m = np.where(lbl == cl)[0]
            m_sorted = m[np.argsort(dist[m])]
            examples = []
            for i in m_sorted:
                wid = str(wids_all[rows_te[cand[i]]])
                examples.append(ctx.get(wid, {"before": "?", "word": "?", "after": "?"}))
            cx, cy = km.cluster_centers_[cl]
            examples_by_cluster[cl] = examples  # ALL members, for labeling
            cluster_meta[cl] = (len(m), cx, cy)

        print(f"labeling {k} clusters for family {c} via log-odds vs rest-of-family...")
        term_labels = label_clusters(examples_by_cluster, min_count=5, top_n=4)

        clusters = []
        for cl in order:
            n, cx, cy = cluster_meta[cl]
            label_terms = term_labels[cl]
            label_str = ", ".join(f"{t!r}({lor:+.1f})" for t, lor, ci_, co_ in label_terms) \
                if label_terms else "(no term clears min-count)"
            print(f"  family{c} cluster centroid=({cx:+.2f},{cy:+.2f}) n={n}: {label_str}")
            clusters.append({"n": n, "cx": cx, "cy": cy, "label": label_str,
                             "examples": examples_by_cluster[cl][: a.max_show]})
        family_clusters[c] = clusters

    render_html(elbow_png, family_clusters, OUT_DIR / a.out, a.max_show)


if __name__ == "__main__":
    main()
