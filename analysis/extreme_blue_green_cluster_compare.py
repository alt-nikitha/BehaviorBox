"""Cluster WITHIN the extreme tails of the fam0("blue")/fam2-blimp("green") scatter from
logreg_three_class_projection.png, and compare them side by side. Same (d1,d2) projection as
logreg_dim_cluster_per_family.py, but restricted to only the most extreme members of each
class first (controlling for "how prototypical") before clustering, instead of clustering
each family's whole sample.

  BLUE extreme = true label fam0, ranked by (d1+d2) most NEGATIVE (purest prose-like fam0).
  GREEN extreme = true label fam2/blimp, ranked by (d1+d2) most POSITIVE (purest code-like blimp).

K per side chosen via elbow method; clusters auto-labeled by log-odds vs the rest of that
side's tokens (same method as logreg_dim_cluster_per_family.py). Output: one HTML with two
columns (BLUE extreme clusters | GREEN extreme clusters) for direct comparison.

Loads the saved clf+scaler (no retraining), reproduces the same exclusive sample/split.

Usage:
  python extreme_blue_green_cluster_compare.py --model fam0_logreg_weights.pkl --n-pool 8000 --n-extreme 1500
"""
import argparse, base64, html, io, pickle, sys
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
from logreg_dim_cluster_per_family import find_knee, label_clusters

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DIR = Path(__file__).resolve().parent
SIDE_COLOR = {"blue": "#2a78d6", "green": "#1baf7a"}


def elbow_plot_b64(elbow_data):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for side, (ks, inertias, knee) in elbow_data.items():
        ax.plot(ks, inertias, "o-", color=SIDE_COLOR[side], label=side, markersize=5)
        ki = ks.index(knee)
        ax.plot(knee, inertias[ki], "o", color=SIDE_COLOR[side], markersize=12,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
    ax.set_xlabel("K")
    ax.set_ylabel("inertia (on d1,d2)")
    ax.set_title("Elbow curves: BLUE extreme vs GREEN extreme (marked = chosen K)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(elbow_png_b64, side_clusters, out_path, max_show, n_extreme):
    esc = html.escape
    parts = [f"""<title>Extreme-tail clustering: fam0(blue) vs fam2-blimp(green)</title>
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
<h1>Clustering WITHIN the extreme tails (top {n_extreme} most extreme each side)</h1>
<div class="meta-top">BLUE = true fam0, most negative (d1+d2) (purest prose-like fam0).
GREEN = true fam2/blimp, most positive (d1+d2) (purest code-like blimp). Clustered on (d1,d2)
only, K chosen via elbow per side. Label = top log-odds terms vs rest of
that side; up to {max_show} nearest-centroid tokens shown per cluster.</div>
<img class="elbow" src="data:image/png;base64,{elbow_png_b64}">
"""]
    for side, clusters in side_clusters.items():
        parts.append(f'<h2 style="color:{SIDE_COLOR[side]}">{side.upper()} extreme '
                     f'({"fam0" if side=="blue" else "fam2_blimp"})</h2>')
        parts.append('<div class="cols">')
        for rank, cl in enumerate(clusters, 1):
            shown = cl["examples"][:max_show]
            parts.append(f'<div class="col"><h3>cluster {rank}/{len(clusters)} '
                         f'n={cl["n"]:,} (showing {len(shown)})<br>'
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


def cluster_side(X_side, word_ids_side, ctx, k_min, k_max, seed, max_show):
    inertias = []
    ks_range = list(range(k_min, k_max + 1))
    for k in ks_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(X_side)
        inertias.append(km.inertia_)
    knee = find_knee(ks_range, inertias)
    print(f"  inertias={[f'{v:.0f}' for v in inertias]}  chosen K={knee}")

    km = KMeans(n_clusters=knee, random_state=seed, n_init=10).fit(X_side)
    lbl = km.labels_
    dist = np.linalg.norm(X_side - km.cluster_centers_[lbl], axis=1)
    sizes = np.bincount(lbl, minlength=knee)
    order = np.argsort(-sizes)  # largest cluster first

    examples_by_cluster = {}
    for cl in range(knee):
        m = np.where(lbl == cl)[0]
        m_sorted = m[np.argsort(dist[m])]
        examples_by_cluster[cl] = [ctx.get(word_ids_side[i],
                                           {"before": "?", "word": "?", "after": "?"})
                                   for i in m_sorted]

    term_labels = label_clusters(examples_by_cluster, min_count=5, top_n=4)

    clusters = []
    for cl in order:
        label_terms = term_labels[cl]
        label_str = ", ".join(f"{t!r}({lor:+.1f})" for t, lor, ci_, co_ in label_terms) \
            if label_terms else "(no term clears min-count)"
        clusters.append({"n": len(examples_by_cluster[cl]), "label": label_str,
                         "examples": examples_by_cluster[cl][:max_show]})
    return clusters, (ks_range, inertias, knee)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--ref-class", type=int, default=0)
    ap.add_argument("--green-class", type=int, default=2)
    ap.add_argument("--n-pool", type=int, default=8000, help="candidate pool per side before ranking")
    ap.add_argument("--n-extreme", type=int, default=1500, help="most extreme tokens kept per side")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--max-show", type=int, default=20)
    ap.add_argument("--context-words", type=int, default=10)
    ap.add_argument("--out", default="extreme_blue_green_cluster_compare.html")
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

    S = clf.decision_function(X_te)
    ref_col = classes.index(a.ref_class)
    other_cols = [c for c in range(len(classes)) if c != ref_col]
    Dproj = S[:, other_cols] - S[:, [ref_col]]
    dsum = Dproj.sum(axis=1)

    rng = np.random.default_rng(a.seed)

    def pick_extreme(target_class, most_negative):
        cand = np.where(y_te == target_class)[0]
        if a.n_pool < len(cand):
            cand = rng.choice(cand, a.n_pool, replace=False)
        order = np.argsort(dsum[cand]) if most_negative else np.argsort(-dsum[cand])
        return cand[order[: a.n_extreme]]

    blue_idx = pick_extreme(a.ref_class, most_negative=True)
    green_idx = pick_extreme(a.green_class, most_negative=False)
    print(f"BLUE extreme (fam{a.ref_class}): n={len(blue_idx):,}  "
          f"dsum range [{dsum[blue_idx].min():.2f}, {dsum[blue_idx].max():.2f}]")
    print(f"GREEN extreme (fam{a.green_class}): n={len(green_idx):,}  "
          f"dsum range [{dsum[green_idx].min():.2f}, {dsum[green_idx].max():.2f}]")

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    side_data = {}
    for side, sel in [("blue", blue_idx), ("green", green_idx)]:
        word_ids = [str(wids_all[rows_te[i]]) for i in sel]
        print(f"fetching context for {side} extreme: {len(word_ids):,} tokens...")
        ctx = get_words_in_context(INPUT_FEATURES, word_ids, N=a.context_words)
        side_data[side] = (Dproj[sel], word_ids, ctx)

    elbow_data = {}
    side_clusters = {}
    for side, (D_side, word_ids, ctx) in side_data.items():
        print(f"\nclustering {side} extreme on (d1,d2) only...")
        clusters, elbow = cluster_side(D_side, word_ids, ctx, a.k_min, a.k_max, a.seed, a.max_show)
        elbow_data[side] = elbow
        side_clusters[side] = clusters

    elbow_png = elbow_plot_b64(elbow_data)
    render_html(elbow_png, side_clusters, OUT_DIR / a.out, a.max_show, a.n_extreme)


if __name__ == "__main__":
    main()
