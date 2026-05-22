"""Render the aligned curve plots from task_shape_groups.py as a static HTML.

Companion to task_shape_groups_to_html.py. That file gives the feature/sample
tables; this file gives the visual overlays — for every cluster (tab 1) and every
task centroid (tab 2), task curves and matched feature curves are z-normalized
(negative features flipped) and drawn on a shared axis so the shape alignment is
visible at a glance.
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}

TASK_METADATA = {
    "arc_challenge":        {"type": "reasoning",  "format": "MCQ"},
    "bbh":                  {"type": "reasoning",  "format": "generation"},
    "blimp":                {"type": "linguistic", "format": "forced-choice"},
    "coqa":                 {"type": "reading",    "format": "generation"},
    "csqa":                 {"type": "reasoning",  "format": "MCQ"},
    "gsm8k":                {"type": "math",       "format": "generation"},
    "hellaswag":            {"type": "reasoning",  "format": "MCQ"},
    "lambada":              {"type": "LM",         "format": "generation"},
    "medmcqa":              {"type": "knowledge",  "format": "MCQ"},
    "mmlu_other":           {"type": "knowledge",  "format": "MCQ"},
    "mmlu_social_sciences": {"type": "knowledge",  "format": "MCQ"},
    "mmlu_stem":            {"type": "knowledge",  "format": "MCQ"},
    "naturalqs":            {"type": "knowledge",  "format": "generation"},
    "piqa":                 {"type": "reasoning",  "format": "MCQ"},
    "winogrande":           {"type": "reasoning",  "format": "MCQ"},
}

TYPE_COLORS = {
    "reasoning":  "#2196F3",
    "math":       "#FF5722",
    "knowledge":  "#4CAF50",
    "reading":    "#9C27B0",
    "linguistic": "#FFC107",
    "LM":         "#00BCD4",
}

METRIC_SLUGS = {"Z-norm Area between": "area", "Z-norm MSE": "mse"}
PRECOMPUTE_DIR = os.path.join(ANALYSIS_DIR, "precomputed_task_shape")


def precompute_path(dataset_name, metric_name):
    return os.path.join(PRECOMPUTE_DIR, f"{dataset_name}__{METRIC_SLUGS[metric_name]}.pkl")


def normalize_curve(op):
    valid = [(i, p) for i, p in enumerate(op) if p is not None]
    if len(valid) < 3:
        return None
    idxs = [i for i, _ in valid]
    vals = np.array([p for _, p in valid], dtype=float)
    sd = vals.std()
    if sd < 1e-9:
        return None
    return idxs, (vals - vals.mean()) / sd


def _shared_diff(a, b):
    ia, va = a
    ib, vb = b
    shared = sorted(set(ia) & set(ib))
    if len(shared) < 3:
        return None
    da, db = dict(zip(ia, va)), dict(zip(ib, vb))
    return np.array([da[k] for k in shared]) - np.array([db[k] for k in shared])


def area_distance(a, b):
    d = _shared_diff(a, b)
    return float("nan") if d is None else float(np.trapezoid(np.abs(d)))


def mse_distance(a, b):
    d = _shared_diff(a, b)
    return float("nan") if d is None else float(np.mean(d ** 2))


DISTANCE_FNS = {"area": area_distance, "mse": mse_distance}


def load_all_tasks_for_dataset(dataset_name):
    out = {}
    for d in sorted(glob.glob(os.path.join(ANALYSIS_DIR, "precomputed_data_*"))):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        if tname not in VALID_TASKS:
            continue
        fpath = os.path.join(d, f"{dataset_name}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                out[tname] = json.load(f)
    return out


def compute_clusters(tsg_data, distance_fn):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    curves, ckpts = {}, None
    for t, td in tsg_data.items():
        n = normalize_curve(td.get("overall_performance", []))
        if n is not None:
            curves[t] = n
            if ckpts is None:
                ckpts = td.get("checkpoints", [])
    names = sorted(curves)
    nn = len(names)
    M = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(i + 1, nn):
            d = distance_fn(curves[names[i]], curves[names[j]])
            if np.isnan(d): d = 0.0
            M[i, j] = M[j, i] = d
    cond = squareform((M + M.T) / 2, checks=False)
    Z = linkage(cond, method="average")
    heights = Z[:, 2]
    max_k = min(8, nn - 1)
    auto_k, best = 2, -np.inf
    for k in range(3 if max_k >= 3 else 2, max_k + 1):
        g = float(heights[nn - k] - heights[nn - k - 1])
        if g > best:
            best, auto_k = g, k
    labels = fcluster(Z, t=auto_k, criterion="maxclust")
    clusters = defaultdict(list)
    for t, l in zip(names, labels):
        clusters[int(l)].append(t)
    return clusters, curves, ckpts


def cluster_feature_scores(clusters, tsg_data, distance_fn):
    out = {}
    for ci, members in clusters.items():
        fs = {}
        for tname in members:
            td = tsg_data.get(tname, {})
            opn = normalize_curve(td.get("overall_performance", []))
            if opn is None:
                continue
            for feat in td.get("features", []):
                mpn = normalize_curve(feat.get("median_probs", []))
                if mpn is None:
                    continue
                d_pos = distance_fn(opn, mpn)
                d_neg = distance_fn(opn, (mpn[0], -mpn[1]))
                if np.isnan(d_pos) and np.isnan(d_neg):
                    continue
                fid = feat["feature_id"]
                if fid not in fs:
                    fs[fid] = {"pos": [], "neg": [], "curve": mpn,
                               "desc": feat.get("description", "")}
                if not np.isnan(d_pos): fs[fid]["pos"].append(d_pos)
                if not np.isnan(d_neg): fs[fid]["neg"].append(d_neg)
        out[ci] = fs
    return out


def select_top(fs, members, side, top_n):
    key = "pos" if side == "positive" else "neg"
    min_pres = max(1, (len(members) + 1) // 2)
    rows = []
    for fid, info in fs.items():
        if len(info[key]) < min_pres:
            continue
        rows.append((fid, float(np.mean(info[key])), info["curve"], info["desc"]))
    rows.sort(key=lambda r: r[1])
    return rows[:top_n]


def fmt_x(ckpts, idxs):
    return [ckpts[i] if ckpts and i < len(ckpts) else str(i) for i in idxs]


# ── Plot builders ──────────────────────────────────────────────────────────


def plot_cluster(ci, members, task_curves, ckpts, pos_rows, neg_rows):
    """One figure per cluster: task curves on the left, positive features in middle,
    negative features (un-flipped, so they oppose the task curve) on the right — all
    on the z-scale."""
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=(
            f"Tasks (n={len(members)})",
            f"Positive features (top {len(pos_rows)}) — aligned",
            f"Negative features (top {len(neg_rows)}) — opposing",
        ),
    )
    for m in members:
        idxs, vals = task_curves[m]
        color = TYPE_COLORS.get(TASK_METADATA.get(m, {}).get("type", "?"), "#888")
        fig.add_trace(go.Scatter(
            x=fmt_x(ckpts, idxs), y=vals, mode="lines+markers",
            name=m, line=dict(color=color, width=2),
            marker=dict(size=5, color=color), legendgroup=f"task_{m}",
        ), row=1, col=1)
    # mean task curve, dashed black, all three panels
    sums, counts = {}, {}
    for m in members:
        idxs, vals = task_curves[m]
        for i, v in zip(idxs, vals):
            sums[i] = sums.get(i, 0.0) + v
            counts[i] = counts.get(i, 0) + 1
    xs_m = sorted(sums)
    ys_m = [sums[i] / counts[i] for i in xs_m]
    for col in (1, 2, 3):
        fig.add_trace(go.Scatter(
            x=fmt_x(ckpts, xs_m), y=ys_m, mode="lines",
            name="cluster mean", line=dict(color="black", width=2.5, dash="dash"),
            legendgroup="cmean", showlegend=(col == 1),
        ), row=1, col=col)
    for fid, d, curve, desc in pos_rows:
        idxs, vals = curve
        fig.add_trace(go.Scatter(
            x=fmt_x(ckpts, idxs), y=vals, mode="lines",
            name=f"f{fid} (avg={d:.3f})",
            line=dict(color="#1976D2", width=1), opacity=0.45,
            hovertemplate=f"<b>f{fid}</b><br>{desc[:120]}<br>%{{x}}<br>z=%{{y:.3f}}<extra></extra>",
            legendgroup=f"pos_{ci}", showlegend=False,
        ), row=1, col=2)
    for fid, d, curve, desc in neg_rows:
        idxs, vals = curve
        fig.add_trace(go.Scatter(
            x=fmt_x(ckpts, idxs), y=vals, mode="lines",
            name=f"f{fid} (avg={d:.3f})",
            line=dict(color="#C62828", width=1), opacity=0.45,
            hovertemplate=f"<b>f{fid} (anti-correlated)</b><br>{desc[:120]}<br>%{{x}}<br>z=%{{y:.3f}}<extra></extra>",
            legendgroup=f"neg_{ci}", showlegend=False,
        ), row=1, col=3)
    fig.update_layout(
        height=420, margin=dict(l=40, r=10, t=50, b=80),
        legend=dict(font=dict(size=10), orientation="v", x=1.02, y=1),
        title=f"Cluster {ci}: {', '.join(members)}",
        title_font=dict(size=13),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(title_text="z-score", row=1, col=1)
    return fig


def plot_task_assignment(t, pre, pos, neg, top_n_pos, top_n_neg):
    """One figure per task centroid: centroid + assigned positive features (left,
    aligned) and assigned negative features (right, un-flipped, so they oppose the
    centroid)."""
    ckpts = pre["task_ckpts"]
    t_idxs, t_vals = pre["task_curves"][t]

    pos = pos[:top_n_pos]
    neg = neg[:top_n_neg]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=(
            f"Positive (aligned) — top {len(pos)} of {len(pre['assignments_pos'].get(t, []))}",
            f"Negative (opposing) — top {len(neg)} of {len(pre['assignments_neg'].get(t, []))}",
        ),
    )

    for col, fids in [(1, pos), (2, neg)]:
        is_neg = (col == 2)
        sums, counts = {}, {}
        for fid, d in fids:
            fc = pre["feature_curves"].get(fid)
            if fc is None: continue
            idxs, vals = fc
            arr = np.asarray(vals)
            for i, v in zip(idxs, arr):
                sums[i] = sums.get(i, 0.0) + v
                counts[i] = counts.get(i, 0) + 1
            color = "#C62828" if is_neg else "#1976D2"
            desc = pre["feature_meta"].get(fid, {}).get("desc", "")
            fig.add_trace(go.Scatter(
                x=fmt_x(ckpts, idxs), y=arr, mode="lines",
                name=f"f{fid} (d={d:.3f})",
                line=dict(color=color, width=1), opacity=0.35,
                hovertemplate=f"<b>f{fid}{' (anti-correlated)' if is_neg else ''}</b><br>"
                              f"{desc[:120]}<br>%{{x}}<br>z=%{{y:.3f}}<extra></extra>",
                showlegend=False,
            ), row=1, col=col)
        # mean curve
        if sums:
            xs_m = sorted(sums)
            ys_m = [sums[i] / counts[i] for i in xs_m]
            fig.add_trace(go.Scatter(
                x=fmt_x(ckpts, xs_m), y=ys_m, mode="lines+markers",
                name="mean feature curve",
                line=dict(color="#FF5722", width=2.5),
                marker=dict(size=5, color="#FF5722"),
                legendgroup="mean_feat", showlegend=(col == 1),
            ), row=1, col=col)
        # task centroid on both sides
        fig.add_trace(go.Scatter(
            x=fmt_x(ckpts, t_idxs), y=t_vals, mode="lines+markers",
            name=f"task: {t}",
            line=dict(color="black", width=2.5, dash="dash"),
            marker=dict(size=6, color="black"),
            legendgroup="centroid", showlegend=(col == 1),
        ), row=1, col=col)

    fig.update_layout(
        height=420, margin=dict(l=40, r=10, t=50, b=80),
        legend=dict(font=dict(size=10), orientation="v", x=1.02, y=1),
        title=f"{t} — centroid vs assigned features",
        title_font=dict(size=13),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(title_text="z-score", row=1, col=1)
    return fig


# ── HTML assembly ──────────────────────────────────────────────────────────


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 0; background: #fafafa; color: #222; }
header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #ddd;
         position: sticky; top: 0; z-index: 10; }
h1 { margin: 0 0 4px; font-size: 18px; }
header .meta { color: #666; font-size: 12px; }
.tabs { display: flex; gap: 4px; margin-top: 12px; }
.tabs button { padding: 8px 16px; background: #eee; border: 1px solid #ccc;
               border-bottom: none; cursor: pointer; font-size: 13px;
               border-radius: 4px 4px 0 0; }
.tabs button.active { background: #fff; font-weight: 600; }
.tab-panel { display: none; padding: 24px; }
.tab-panel.active { display: block; }
.plot-card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
             padding: 12px; margin-bottom: 18px; }
"""

JS = """
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('btn-' + name).classList.add('active');
  // trigger plotly resize on newly-visible plots
  window.dispatchEvent(new Event('resize'));
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm")
    ap.add_argument("--metric", default="area", choices=list(DISTANCE_FNS))
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR, "task_shape_groups_curves.html"))
    ap.add_argument("--top-n", type=int, default=20,
                    help="Top features per cluster column (tab 1).")
    ap.add_argument("--top-n-tab2", type=int, default=30,
                    help="Top features per task column (tab 2).")
    args = ap.parse_args()

    metric_name = "Z-norm Area between" if args.metric == "area" else "Z-norm MSE"
    distance_fn = DISTANCE_FNS[args.metric]

    print(f"loading task json for dataset={args.dataset!r}…")
    tsg_data = load_all_tasks_for_dataset(args.dataset)
    print(f"  {len(tsg_data)} tasks")

    print("clustering…")
    clusters, task_curves, ckpts = compute_clusters(tsg_data, distance_fn)
    print(f"  k = {len(clusters)}")

    print("scoring features per cluster…")
    fscores = cluster_feature_scores(clusters, tsg_data, distance_fn)

    print("building tab 1 cluster plots…")
    tab1_html = []
    for ci in sorted(clusters):
        members = clusters[ci]
        pos = select_top(fscores[ci], members, "positive", args.top_n)
        neg = select_top(fscores[ci], members, "negative", args.top_n)
        fig = plot_cluster(ci, members, task_curves, ckpts, pos, neg)
        tab1_html.append(
            f"<div class='plot-card'>"
            f"{fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'tab1_c{ci}')}"
            f"</div>"
        )

    pkl = precompute_path(args.dataset, metric_name)
    if not os.path.exists(pkl):
        raise SystemExit(f"missing pickle: {pkl}")
    print(f"loading {pkl}…")
    with open(pkl, "rb") as f:
        pre = pickle.load(f)

    print("building tab 2 task plots…")
    tab2_html = []
    for t in sorted(pre["task_curves"]):
        pos = sorted(pre["assignments_pos"].get(t, []), key=lambda x: x[1])
        neg = sorted(pre["assignments_neg"].get(t, []), key=lambda x: x[1])
        fig = plot_task_assignment(t, pre, pos, neg,
                                   args.top_n_tab2, args.top_n_tab2)
        tab2_html.append(
            f"<div class='plot-card'>"
            f"{fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'tab2_{t}')}"
            f"</div>"
        )

    summary = (
        f"{len(tsg_data)} tasks · k={len(clusters)} clusters · "
        f"{len(pre['feature_curves'])} features · "
        f"tab1 top-{args.top_n} per side, tab2 top-{args.top_n_tab2} per side"
    )

    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Task Shape Groups — Aligned Curves — {args.dataset}</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>{CSS}</style></head><body>
<header>
  <h1>Task Shape Groups — Aligned Curves</h1>
  <div class='meta'>Dataset: <code>{args.dataset}</code> · metric: <code>{metric_name}</code> · {summary}</div>
  <div class='tabs'>
    <button id='btn-tab1' class='active' onclick="showTab('tab1')">Task clusters (curves + matched features)</button>
    <button id='btn-tab2' onclick="showTab('tab2')">Per-task centroid (assigned features)</button>
  </div>
</header>
<div id='panel-tab1' class='tab-panel active'>{''.join(tab1_html)}</div>
<div id='panel-tab2' class='tab-panel'>{''.join(tab2_html)}</div>
<script>{JS}</script>
</body></html>"""

    with open(args.out, "w") as f:
        f.write(html_doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
