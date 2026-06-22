"""Cluster benchmark tasks by the shape of their performance trajectory over training.

Each task's overall_performance curve is z-normalized; distance between two tasks is
either pointwise MSE or integrated area between the curves (user-selectable). Tasks
in the same cluster have approximately the same acquisition shape. Hover/legend
annotations include task type (reasoning/math/knowledge/…) and format (MCQ/generation/…).

Run: streamlit run analysis/task_shape_groups.py
"""

import sys

if "streamlit" in sys.modules or "streamlit.runtime" in sys.modules:
    import streamlit.config as _cfg
    _cfg.set_option("server.address", "0.0.0.0")
    _cfg.set_option("server.headless", True)

import streamlit as st
import json
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Task Shape Groups", layout="wide")

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}

TASK_METADATA = {
    "arc_challenge":        {"type": "reasoning",  "domain": "science",      "format": "MCQ"},
    "bbh":                  {"type": "reasoning",  "domain": "mixed",        "format": "generation"},
    "blimp":                {"type": "linguistic", "domain": "syntax",       "format": "forced-choice"},
    "coqa":                 {"type": "reading",    "domain": "conversation", "format": "generation"},
    "csqa":                 {"type": "reasoning",  "domain": "commonsense",  "format": "MCQ"},
    "gsm8k":                {"type": "math",       "domain": "arithmetic",   "format": "generation"},
    "hellaswag":            {"type": "reasoning",  "domain": "commonsense",  "format": "MCQ"},
    "lambada":              {"type": "LM",         "domain": "narrative",    "format": "generation"},
    "medmcqa":              {"type": "knowledge",  "domain": "medical",      "format": "MCQ"},
    "minerva_math":         {"type": "math",       "domain": "competition",  "format": "generation"},
    "mmlu_other":           {"type": "knowledge",  "domain": "misc",         "format": "MCQ"},
    "mmlu_social_sciences": {"type": "knowledge",  "domain": "social_sci",   "format": "MCQ"},
    "mmlu_stem":            {"type": "knowledge",  "domain": "STEM",         "format": "MCQ"},
    "naturalqs":            {"type": "knowledge",  "domain": "factual",      "format": "generation"},
    "piqa":                 {"type": "reasoning",  "domain": "physical",     "format": "MCQ"},
    "winogrande":           {"type": "reasoning", "domain":  "coreference",  "format": "MCQ"},
}

TYPE_COLORS = {
    "reasoning":  "#2196F3",
    "math":       "#FF5722",
    "knowledge":  "#4CAF50",
    "reading":    "#9C27B0",
    "linguistic": "#FFC107",
    "LM":         "#00BCD4",
}

TASK_DESCRIPTIONS = {
    "arc_challenge": "ARC Challenge — grade-school science questions requiring reasoning (multiple choice).",
    "bbh": "BIG-Bench Hard — 23 challenging BIG-Bench tasks where LMs previously fell below average human performance.",
    "blimp": "BLiMP — minimal-pair grammaticality judgments testing knowledge of English syntax, morphology, and semantics.",
    "coqa": "CoQA — conversational question answering over passages requiring coreference and pragmatic reasoning.",
    "csqa": "CommonsenseQA — multiple-choice questions requiring commonsense knowledge and reasoning.",
    "gsm8k": "GSM8K — grade-school math word problems requiring multi-step arithmetic reasoning.",
    "hellaswag": "HellaSwag — sentence completion requiring commonsense natural language inference.",
    "lambada": "LAMBADA — predicting the last word of a passage, testing long-range contextual understanding.",
    "medmcqa": "MedMCQA — medical entrance exam multiple-choice questions across 21 medical subjects.",
    "minerva_math": "Minerva Math — competition-level mathematics problems requiring multi-step quantitative reasoning.",
    "mmlu_other": "MMLU (Other) — miscellaneous knowledge domains including business, health, and professional topics.",
    "mmlu_social_sciences": "MMLU (Social Sciences) — economics, geography, psychology, sociology, and related fields.",
    "mmlu_stem": "MMLU (STEM) — science, technology, engineering, and mathematics questions.",
    "naturalqs": "Natural Questions — real Google search queries with answers from Wikipedia.",
    "piqa": "PIQA — physical intuition QA, testing understanding of everyday physical processes.",
    "winogrande": "WinoGrande — large-scale Winograd schema challenge for coreference resolution.",
}


@st.cache_data
def discover_tasks():
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    out = {}
    for d in sorted(glob.glob(pattern)):
        task = os.path.basename(d).replace("precomputed_data_", "")
        if task not in VALID_TASKS:
            continue
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        out[task] = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    return out


@st.cache_data
def load_all_tasks_for_dataset(dataset_name):
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    result = {}
    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        if tname not in VALID_TASKS:
            continue
        fpath = os.path.join(d, f"{dataset_name}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                result[tname] = json.load(f)
    return result


def normalize_curve(op):
    """Z-normalize a curve. Returns (valid_idxs, z_values) or None if degenerate."""
    valid = [(i, p) for i, p in enumerate(op) if p is not None]
    if len(valid) < 3:
        return None
    idxs = [i for i, _ in valid]
    vals = np.array([p for _, p in valid], dtype=float)
    sd = vals.std()
    if sd < 1e-9:
        return None
    return idxs, (vals - vals.mean()) / sd


def _shared_diff(curve_a, curve_b):
    idx_a, v_a = curve_a
    idx_b, v_b = curve_b
    shared = sorted(set(idx_a) & set(idx_b))
    if len(shared) < 3:
        return None
    da = dict(zip(idx_a, v_a))
    db = dict(zip(idx_b, v_b))
    return np.array([da[k] for k in shared]) - np.array([db[k] for k in shared])


def mse_distance(curve_a, curve_b):
    """Pointwise MSE between two z-normalized curves on shared checkpoints."""
    diff = _shared_diff(curve_a, curve_b)
    if diff is None:
        return float("nan")
    return float(np.mean(diff ** 2))


def area_distance(curve_a, curve_b):
    """Integrated absolute difference (trapezoid) between two z-normalized curves."""
    diff = _shared_diff(curve_a, curve_b)
    if diff is None:
        return float("nan")
    return float(np.trapezoid(np.abs(diff)))


DISTANCE_METRICS = {
    # "Z-norm MSE": mse_distance,
    "Z-norm Area between": area_distance,
}


# METRIC_SLUGS = {"Z-norm MSE": "mse", "Z-norm Area between": "area"}
METRIC_SLUGS = {"Z-norm Area between": "area"}

PRECOMPUTE_DIR = os.path.join(ANALYSIS_DIR, "precomputed_task_shape")


def precompute_path(dataset_name, metric_name):
    slug = METRIC_SLUGS[metric_name]
    return os.path.join(PRECOMPUTE_DIR, f"{dataset_name}__{slug}.pkl")


@st.cache_data
def load_precomputed_view(dataset_name, metric_name, _mtime):
    """Reads the pickle written by precompute_task_shape_groups.py. The _mtime
    arg only exists to bust the cache when the file is regenerated."""
    import pickle
    with open(precompute_path(dataset_name, metric_name), "rb") as f:
        return pickle.load(f)


# ── UI ──────────────────────────────────────────────────────────────────────

tasks = discover_tasks()

st.title("Task Shape Groups")
st.caption("Cluster benchmark tasks by the shape of their performance trajectory over training. "
           "Each task's overall_performance curve is z-normalized; distance between two tasks "
           "is either pointwise MSE or integrated area between the curves. Tasks in the same "
           "cluster have approximately the same acquisition shape.")

all_datasets = sorted(set().union(*(tpaths.keys() for tpaths in tasks.values())))
all_datasets = [d for d in all_datasets if "early" in d and "late" in d and "0.8" in d and "znorm" in d and "log" in d]
col_ds, col_metric = st.columns([2, 1])
with col_ds:
    dataset_name = st.selectbox("Dataset (model)", all_datasets, key="tsg_dataset")
with col_metric:
    metric_name = st.selectbox("Shape distance metric", list(DISTANCE_METRICS.keys()), key="tsg_metric",
                               help="Both metrics operate on z-normalized curves; lower = more similar shape.")
distance_fn = DISTANCE_METRICS[metric_name]

tsg_data = load_all_tasks_for_dataset(dataset_name)

tab_tasks, tab_feats = st.tabs(["Task clusters", "Feature clusters"])

# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — original task clustering
# ════════════════════════════════════════════════════════════════════════════
with tab_tasks:
    task_curves = {}
    task_ckpts = None
    for tname, td in tsg_data.items():
        op = td.get("overall_performance", [])
        norm = normalize_curve(op)
        if norm is not None:
            task_curves[tname] = norm
            if task_ckpts is None:
                task_ckpts = td.get("checkpoints", [])

    if len(task_curves) < 2:
        st.info("Not enough tasks with valid performance data.")
        st.stop()

    task_names_ord = sorted(task_curves.keys())
    n = len(task_names_ord)

    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(task_curves[task_names_ord[i]], task_curves[task_names_ord[j]])
            if np.isnan(d):
                d = 0.0
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # ── Heatmap ────────────────────────────────────────────────────────────────
    st.markdown(f"#### Pairwise {metric_name} (lower = more similar shape)")
    label_with_meta = []
    for t in task_names_ord:
        md = TASK_METADATA.get(t, {})
        label_with_meta.append(f"{t} [{md.get('type', '?')}/{md.get('format', '?')}]")

    _dmax = float(dist_mat.max()) if dist_mat.size else 1.0
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=dist_mat, x=label_with_meta, y=label_with_meta,
        colorscale="Viridis_r", zmin=0, zmax=_dmax if _dmax > 0 else 1.0,
        text=np.round(dist_mat, 2), texttemplate="%{text}", textfont={"size": 9},
    ))
    heatmap_fig.update_layout(height=80 + 35 * n, margin=dict(l=10, r=10, t=10, b=120),
                              xaxis=dict(tickangle=45))
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # ── Hierarchical clustering ─────────────────────────────────────────────────
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        sym_dist = (dist_mat + dist_mat.T) / 2
        np.fill_diagonal(sym_dist, 0)
        condensed = squareform(sym_dist, checks=False)

        col_link, col_k = st.columns([1, 2])
        with col_link:
            linkage_method = st.selectbox("Linkage", ["average", "complete", "single", "ward"],
                                          index=0, key="tsg_linkage",
                                          help="ward typically yields more balanced/separated clusters; "
                                               "average is more sensitive to outliers under MSE.")
        Z = linkage(condensed, method=linkage_method)

        max_k = min(8, n - 1)

        # Auto-pick k by largest gap in linkage merge heights — the natural dendrogram cut.
        # Skip k=2 since the k=2→1 merge is almost always the biggest absolute jump in any
        # tree (the trivial first split) and overwhelms the more interesting structural gaps.
        heights = Z[:, 2]
        auto_k, best_gap = 2, -np.inf
        start = 3 if max_k >= 3 else 2
        for k in range(start, max_k + 1):
            gap = float(heights[n - k] - heights[n - k - 1])
            if gap > best_gap:
                best_gap = gap
                auto_k = k
        gap_caption = f"dendrogram-gap pick: k = {auto_k} (Δ = {best_gap:.3f}); k=2 excluded as trivial split"

        with col_k:
            n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=max_k,
                                   value=int(auto_k), step=1, key="tsg_k",
                                   help=f"Manual override. {gap_caption}.")
        st.caption(gap_caption)

        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        clusters = {}
        for tname, lbl in zip(task_names_ord, labels):
            clusters.setdefault(int(lbl), []).append(tname)

        # ── Cluster members with type/format ────────────────────────────────────
        st.markdown("#### Clusters")
        cluster_cols = st.columns(n_clusters)
        for col_idx, (ci, members) in enumerate(sorted(clusters.items())):
            with cluster_cols[col_idx]:
                st.markdown(f"**Cluster {ci}** ({len(members)})")
                rows = []
                for m in members:
                    md = TASK_METADATA.get(m, {})
                    rows.append({
                        "Task": m,
                        "Type": md.get("type", "?"),
                        "Domain": md.get("domain", "?"),
                        "Format": md.get("format", "?"),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                             height=min(300, 35 * len(rows) + 40))

        # ── Normalized curves overlaid by cluster ───────────────────────────────
        st.markdown("#### Normalized curves by cluster")
        n_cols_plot = min(n_clusters, 4)
        n_rows_plot = (n_clusters + n_cols_plot - 1) // n_cols_plot
        curve_fig = make_subplots(
            rows=n_rows_plot, cols=n_cols_plot,
            subplot_titles=[f"Cluster {ci}" for ci in sorted(clusters.keys())]
                           + [""] * (n_rows_plot * n_cols_plot - n_clusters),
            shared_yaxes=True,
        )
        for plot_idx, (ci, members) in enumerate(sorted(clusters.items())):
            r = plot_idx // n_cols_plot + 1
            c = plot_idx % n_cols_plot + 1
            for m in members:
                idxs, vals = task_curves[m]
                x_lbls = [task_ckpts[i] if task_ckpts and i < len(task_ckpts) else str(i) for i in idxs]
                typ = TASK_METADATA.get(m, {}).get("type", "?")
                fmt = TASK_METADATA.get(m, {}).get("format", "?")
                color = TYPE_COLORS.get(typ, "#888")
                curve_fig.add_trace(
                    go.Scatter(
                        x=x_lbls, y=vals, mode="lines+markers",
                        name=f"{m} [{typ}/{fmt}]",
                        line=dict(color=color, width=1.5),
                        marker=dict(size=6, color=color),
                        legendgroup=f"c{ci}",
                    ),
                    row=r, col=c,
                )
        curve_fig.update_layout(
            height=350 * n_rows_plot + 50,
            margin=dict(l=10, r=10, t=40, b=80),
            legend=dict(font=dict(size=9)),
        )
        curve_fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        st.plotly_chart(curve_fig, use_container_width=True)

        # ── Raw (unnormalized) curves overlaid by cluster ───────────────────────
        st.markdown("#### Raw curves by cluster (unscaled)")
        raw_fig = make_subplots(
            rows=n_rows_plot, cols=n_cols_plot,
            subplot_titles=[f"Cluster {ci}" for ci in sorted(clusters.keys())]
                           + [""] * (n_rows_plot * n_cols_plot - n_clusters),
        )
        for plot_idx, (ci, members) in enumerate(sorted(clusters.items())):
            r = plot_idx // n_cols_plot + 1
            c = plot_idx % n_cols_plot + 1
            for m in members:
                td = tsg_data[m]
                op = td.get("overall_performance", [])
                ck = td.get("checkpoints", task_ckpts or [])
                pts = [(ck[i] if i < len(ck) else str(i), p) for i, p in enumerate(op) if p is not None]
                if not pts:
                    continue
                x_lbls = [x for x, _ in pts]
                y_vals = [y for _, y in pts]
                typ = TASK_METADATA.get(m, {}).get("type", "?")
                fmt = TASK_METADATA.get(m, {}).get("format", "?")
                color = TYPE_COLORS.get(typ, "#888")
                raw_fig.add_trace(
                    go.Scatter(
                        x=x_lbls, y=y_vals, mode="lines+markers",
                        name=f"{m} [{typ}/{fmt}]",
                        line=dict(color=color, width=1.5),
                        marker=dict(size=6, color=color),
                        legendgroup=f"raw_c{ci}",
                    ),
                    row=r, col=c,
                )
        raw_fig.update_layout(
            height=350 * n_rows_plot + 50,
            margin=dict(l=10, r=10, t=40, b=80),
            legend=dict(font=dict(size=9)),
        )
        raw_fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        st.plotly_chart(raw_fig, use_container_width=True)

        # ── Features matching each cluster's shape ──────────────────────────────
        st.markdown("---")
        st.markdown("#### Features matching each cluster's shape")
        st.caption(f"For each cluster, SAE features whose median_probs curves most closely match "
                   f"the cluster's task performance curves. Score = mean {metric_name} across "
                   f"the cluster's tasks (lower = better; features must appear in at least half "
                   f"the cluster's tasks). The **Negative** sub-tab flips the feature curve "
                   f"(negates the z-values) and scores against task performance — surfacing "
                   f"features that anti-correlate with the cluster (same shape, opposite direction).")

        feat_top_n = st.number_input("Top features per cluster", min_value=5, max_value=100,
                                     value=20, step=5, key="tsg_feat_topn")

        def _typical_std(feat):
            """Median of std_probs across checkpoints — robust to a few outlier checkpoints,
            captures whether most checkpoints have tightly-agreeing samples."""
            sps = [s for s in (feat.get("std_probs") or []) if s is not None]
            return float(np.median(sps)) if sps else float("nan")

        # For each (cluster, feature) collect distances against the original curve AND against
        # the flipped feature curve (z-values negated). Small flipped-distance = anti-correlated.
        cluster_feat_scores = {}  # ci -> {fid -> {"pos_scores","neg_scores","stds","desc"}}
        for ci, members in sorted(clusters.items()):
            feat_scores = {}
            for tname in members:
                td = tsg_data.get(tname, {})
                op_norm = normalize_curve(td.get("overall_performance", []))
                if op_norm is None:
                    continue
                for feat in td.get("features", []):
                    mp_norm = normalize_curve(feat.get("median_probs", []))
                    if mp_norm is None:
                        continue
                    d_pos = distance_fn(op_norm, mp_norm)
                    mp_flipped = (mp_norm[0], -mp_norm[1])
                    d_neg = distance_fn(op_norm, mp_flipped)
                    if np.isnan(d_pos) and np.isnan(d_neg):
                        continue
                    fid = feat["feature_id"]
                    if fid not in feat_scores:
                        feat_scores[fid] = {"pos_scores": [], "neg_scores": [],
                                            "stds": [], "desc": feat.get("description", "")}
                    if not np.isnan(d_pos):
                        feat_scores[fid]["pos_scores"].append(d_pos)
                    if not np.isnan(d_neg):
                        feat_scores[fid]["neg_scores"].append(d_neg)
                    ts = _typical_std(feat)
                    if not np.isnan(ts):
                        feat_scores[fid]["stds"].append(ts)
            cluster_feat_scores[ci] = feat_scores

        all_avgs = [float(np.mean(info[k]))
                    for fs in cluster_feat_scores.values()
                    for info in fs.values()
                    for k in ("pos_scores", "neg_scores") if info[k]]
        slider_max = float(np.quantile(all_avgs, 0.95)) if all_avgs else 1.0
        slider_max = max(slider_max, 0.01)
        all_stds = [float(np.median(info["stds"]))
                    for fs in cluster_feat_scores.values()
                    for info in fs.values() if info["stds"]]
        std_slider_max = float(np.quantile(all_stds, 0.95)) if all_stds else 1.0
        std_slider_max = max(std_slider_max, 0.01)

        fc1, fc2 = st.columns(2)
        with fc1:
            feat_max_dist = st.slider(f"Max avg {metric_name} (lower = stricter)", 0.0, slider_max,
                                      min(slider_max * 0.5, slider_max), slider_max / 100,
                                      key="tsg_feat_maxdist")
        with fc2:
            feat_max_std = st.slider("Max median std (within-feature sample spread)",
                                     0.0, std_slider_max, std_slider_max, std_slider_max / 100,
                                     key="tsg_feat_maxstd",
                                     help="Filters out features whose activated samples disagree on the curve. "
                                          "Uses median(std_probs) across checkpoints — robust to a few outlier "
                                          "checkpoints, requires the typical checkpoint to have tight agreement.")

        def _render_feat_match_panel(ci, members, feat_scores, direction):
            """Render the table + inspector for one cluster, one direction.
            direction: 'positive' (correlated) or 'negative' (anti-correlated; feature flipped)."""
            score_key = "pos_scores" if direction == "positive" else "neg_scores"
            min_presence = max(1, (len(members) + 1) // 2)
            rows = []
            for fid, info in feat_scores.items():
                scores = info[score_key]
                if len(scores) < min_presence:
                    continue
                avg = float(np.mean(scores))
                if avg > feat_max_dist:
                    continue
                med_std = float(np.median(info["stds"])) if info["stds"] else float("nan")
                if not np.isnan(med_std) and med_std > feat_max_std:
                    continue
                rows.append({
                    "Feature ID": fid,
                    f"Avg {metric_name}": avg,
                    f"Max {metric_name}": float(np.max(scores)),
                    "Median Std": med_std,
                    "N tasks": len(scores),
                    "Description": info["desc"],
                })

            if not rows:
                st.info(f"No {direction} features below threshold for this cluster.")
                return

            avg_col = f"Avg {metric_name}"
            max_col = f"Max {metric_name}"
            df = (pd.DataFrame(rows)
                    .sort_values(avg_col, ascending=True)
                    .head(int(feat_top_n))
                    .reset_index(drop=True))
            event = st.dataframe(
                df,
                hide_index=True, use_container_width=True,
                height=min(500, 35 * len(df) + 40),
                column_config={
                    avg_col: st.column_config.NumberColumn(format="%.4f"),
                    max_col: st.column_config.NumberColumn(format="%.4f"),
                    "Median Std": st.column_config.NumberColumn(format="%.4f"),
                },
                on_select="rerun",
                selection_mode="single-row",
                key=f"tsg_table_{ci}_{direction}",
            )
            st.caption("Click a row above to inspect that feature's curve + samples.")

            selected_rows = event.selection.rows if hasattr(event, "selection") else []
            if not selected_rows:
                return
            sel_fid = df.iloc[selected_rows[0]]["Feature ID"]

            feat_obj, src_task = None, None
            for tname in members:
                for f in tsg_data.get(tname, {}).get("features", []):
                    if f["feature_id"] == sel_fid:
                        if feat_obj is None or (f.get("samples") and not feat_obj.get("samples")):
                            feat_obj, src_task = f, tname
                if feat_obj and feat_obj.get("samples"):
                    break

            if feat_obj is None:
                return

            flip_note = " — feature curve flipped for display" if direction == "negative" else ""
            st.caption(f"Feature {sel_fid} — pulled from task `{src_task}`{flip_note}. "
                       f"{feat_obj.get('description', '')}")

            ins_fig = go.Figure()
            for m in members:
                if m not in task_curves:
                    continue
                idxs, vals = task_curves[m]
                x_lbls = [task_ckpts[i] if task_ckpts and i < len(task_ckpts) else str(i) for i in idxs]
                color = TYPE_COLORS.get(TASK_METADATA.get(m, {}).get("type", "?"), "#888")
                ins_fig.add_trace(go.Scatter(
                    x=x_lbls, y=vals, mode="lines+markers", name=f"task: {m}",
                    line=dict(color=color, width=1.5), marker=dict(size=5),
                    opacity=0.75,
                ))
            mp_norm = normalize_curve(feat_obj.get("median_probs", []))
            if mp_norm is not None:
                f_idxs, f_vals = mp_norm
                if direction == "negative":
                    f_vals = -f_vals
                f_x = [task_ckpts[i] if task_ckpts and i < len(task_ckpts) else str(i) for i in f_idxs]
                label = f"feature {sel_fid}" + (" (flipped)" if direction == "negative" else "")
                ins_fig.add_trace(go.Scatter(
                    x=f_x, y=f_vals, mode="lines+markers", name=label,
                    line=dict(color="black", width=3, dash="dash"),
                    marker=dict(size=8, color="black"),
                ))
            ins_fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                yaxis_title="Z-score", hovermode="x unified",
            )
            ins_fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
            st.plotly_chart(ins_fig, use_container_width=True,
                            key=f"tsg_insfig_{ci}_{direction}")

            samples = feat_obj.get("samples", [])
            if samples:
                import html as _html
                sort_by = st.radio(
                    "Sort samples by", ["cos_sim", "activation"], horizontal=True, index=0,
                    key=f"tsg_sortsamp_{ci}_{direction}",
                )
                samp_df = (pd.DataFrame(samples)
                           .sort_values(sort_by, ascending=False)
                           .reset_index(drop=True))
                rows_html = []
                for _, rr in samp_df.iterrows():
                    before = _html.escape(str(rr.get("before", "")))
                    word = _html.escape(str(rr.get("word", "")))
                    after = _html.escape(str(rr.get("after", "")))
                    cos = rr.get("cos_sim", 0) or 0
                    act = rr.get("activation", 0) or 0
                    rows_html.append(
                        "<tr>"
                        f"<td style='text-align:right;color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{before}</td>"
                        f"<td style='font-weight:bold;padding:4px 6px;white-space:nowrap;'>{word}</td>"
                        f"<td style='color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{after}</td>"
                        f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{cos:.4f}</td>"
                        f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{act:.4f}</td>"
                        "</tr>"
                    )
                table_html = (
                    "<div style='max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:4px;'>"
                    "<table style='width:100%;border-collapse:collapse;'>"
                    "<thead style='position:sticky;top:0;background:#f0f0f0;z-index:1;'><tr>"
                    "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Before</th>"
                    "<th style='border-bottom:2px solid #ddd;padding:6px;'>Word</th>"
                    "<th style='border-bottom:2px solid #ddd;padding:6px;'>After</th>"
                    "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Cos Sim</th>"
                    "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Activation</th>"
                    "</tr></thead><tbody>" + "\n".join(rows_html) + "</tbody></table></div>"
                )
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("No samples available for this feature.")

        for ci, members in sorted(clusters.items()):
            st.markdown(f"##### Cluster {ci} — {', '.join(members)}")
            feat_scores = cluster_feat_scores.get(ci, {})
            n_pos = sum(1 for info in feat_scores.values() if info["pos_scores"])
            n_neg = sum(1 for info in feat_scores.values() if info["neg_scores"])
            pos_sub, neg_sub = st.tabs([
                f"Positive (correlated) — {n_pos}",
                f"Negative (flipped) — {n_neg}",
            ])
            with pos_sub:
                _render_feat_match_panel(ci, members, feat_scores, "positive")
            with neg_sub:
                _render_feat_match_panel(ci, members, feat_scores, "negative")

        # ── Cluster x type/format cross-tab ─────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Cluster composition by type/format")
        xtab_rows = []
        for ci, members in sorted(clusters.items()):
            types = [TASK_METADATA.get(m, {}).get("type", "?") for m in members]
            formats = [TASK_METADATA.get(m, {}).get("format", "?") for m in members]
            xtab_rows.append({
                "Cluster": ci,
                "N": len(members),
                "Types": ", ".join(sorted(set(types))),
                "Formats": ", ".join(sorted(set(formats))),
                "Tasks": ", ".join(members),
            })
        st.dataframe(pd.DataFrame(xtab_rows), hide_index=True, use_container_width=True)

    except ImportError:
        st.warning("Install `scipy` (and optionally `scikit-learn`) for clustering: "
                   "`pip install scipy scikit-learn`")


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — cluster the feature curves themselves
# ════════════════════════════════════════════════════════════════════════════
with tab_feats:
    st.markdown("### Features grouped by nearest task curve")
    st.caption("Each task's z-normalized overall_performance curve is treated as a centroid. "
               "Every SAE feature (deduped by feature_id) is assigned to the **single task "
               "whose curve it most closely matches** under the selected shape distance. The "
               "**Negative** sub-tab does the same after flipping the feature's z-curve for "
               "matching — i.e. features whose shape is the inverse of the task's. (The plot "
               "still draws the feature curve un-flipped so the anti-correlation is visible.) "
               "Selected feature's samples are the same regardless of task — they're sorted "
               "by their z-curve distance to the **chosen task** centroid."
               "\n\n*Backed by `precomputed_task_shape/{dataset}__{metric}.pkl`. Regenerate "
               "with `python precompute_task_shape_groups.py --dataset … --metric …`.*")

    pkl_path = precompute_path(dataset_name, metric_name)
    if not os.path.exists(pkl_path):
        st.warning(
            f"No precomputed file found at `{pkl_path}`.\n\n"
            f"Run:\n```bash\npython precompute_task_shape_groups.py "
            f"--dataset {dataset_name} --metric {METRIC_SLUGS[metric_name]}\n```"
        )
        st.stop()

    pre = load_precomputed_view(dataset_name, metric_name, os.path.getmtime(pkl_path))
    task_curves_p = pre["task_curves"]
    task_ckpts_p = pre["task_ckpts"]
    feature_curves_p = pre["feature_curves"]
    feature_meta_p = pre["feature_meta"]
    feature_samples_p = pre["feature_samples"]
    assignments_pos_p = pre["assignments_pos"]
    assignments_neg_p = pre["assignments_neg"]
    sample_mode_p = pre["metric"]

    if not task_curves_p or not feature_curves_p:
        st.info("No precomputed task centroids / features in the loaded file.")
    else:
        n_feat_total = len(feature_curves_p)
        n_centroids = len(task_curves_p)
        st.write(f"**{n_feat_total}** unique features → assigned to one of "
                 f"**{n_centroids}** task centroids.")

        # ── Filters (cheap; applied to precomputed assignments) ──────────
        valid_stds = [m["median_std"] for m in feature_meta_p.values()
                      if not np.isnan(m["median_std"])]
        fc_std_max_slider = float(np.quantile(valid_stds, 0.95)) if valid_stds else 1.0
        fc_std_max_slider = max(fc_std_max_slider, 0.01)

        col_ff1, col_ff2 = st.columns(2)
        with col_ff1:
            fc_std_max = st.slider("Max median std (within-feature spread)",
                                   0.0, fc_std_max_slider, fc_std_max_slider,
                                   fc_std_max_slider / 100, key="tsg_fc_std",
                                   help="Drops features whose samples disagree on the curve.")
        with col_ff2:
            fc_min_tasks = st.number_input("Min tasks the feature appears in",
                                           min_value=1, max_value=max(1, n_centroids),
                                           value=1, step=1, key="tsg_fc_mintasks")

        def _filter_assignments(assignments):
            out = {}
            for t, fids in assignments.items():
                kept = []
                for fid, d in fids:
                    m = feature_meta_p.get(fid)
                    if not m:
                        continue
                    if m["n_tasks"] < int(fc_min_tasks):
                        continue
                    ms = m["median_std"]
                    if not np.isnan(ms) and ms > fc_std_max:
                        continue
                    kept.append((fid, d))
                out[t] = kept
            return out

        a_pos = _filter_assignments(assignments_pos_p)
        a_neg = _filter_assignments(assignments_neg_p)

        # Distance threshold — based on observed combined distribution
        all_dists = ([d for v in a_pos.values() for _, d in v]
                   + [d for v in a_neg.values() for _, d in v])
        fc_dist_max_slider = float(np.quantile(all_dists, 0.95)) if all_dists else 1.0
        fc_dist_max_slider = max(fc_dist_max_slider, 0.01)
        fc_dist_max = st.slider(f"Max {metric_name} to centroid (lower = stricter)",
                                0.0, fc_dist_max_slider, fc_dist_max_slider,
                                fc_dist_max_slider / 100, key="tsg_fc_distmax",
                                help="Features with distance above this are dropped "
                                     "(consider them shape-unaligned to any task).")

        a_pos = {t: [(f, d) for f, d in v if d <= fc_dist_max] for t, v in a_pos.items()}
        a_neg = {t: [(f, d) for f, d in v if d <= fc_dist_max] for t, v in a_neg.items()}

        task_names_centroid = sorted(task_curves_p.keys())

        def _render_assignment_view(assignments, direction, key_suffix):
            total = sum(len(v) for v in assignments.values())

            # ── Sizes table ─────────────────────────────────────────────
            size_rows = []
            for t in task_names_centroid:
                n = len(assignments[t])
                md = TASK_METADATA.get(t, {})
                size_rows.append({
                    "Task (centroid)": t,
                    "Type": md.get("type", "?"),
                    "Format": md.get("format", "?"),
                    "N features": n,
                    "% of assigned": 100.0 * n / total if total > 0 else 0.0,
                })
            size_rows.sort(key=lambda r: -r["N features"])
            st.markdown("##### Cluster sizes (features per task centroid)")
            st.dataframe(pd.DataFrame(size_rows), hide_index=True,
                         use_container_width=True,
                         column_config={"% of assigned":
                                        st.column_config.NumberColumn(format="%.1f%%")})

            active_tasks = [t for t in task_names_centroid if assignments[t]]
            if not active_tasks:
                st.info(f"No {direction} assignments above threshold.")
                return

            # ── Faceted: task centroid + mean assigned feature curve ────
            st.markdown("##### Task centroid (black, dashed) vs mean assigned feature curve "
                        "(orange)")
            if direction == "negative":
                st.caption("Negative view: feature curves are kept un-flipped, so the mean "
                           "curve runs opposite to the task centroid.")
            ncols_p = min(len(active_tasks), 4)
            nrows_p = (len(active_tasks) + ncols_p - 1) // ncols_p
            facet_fig = make_subplots(
                rows=nrows_p, cols=ncols_p,
                subplot_titles=[f"{t} (n={len(assignments[t])})" for t in active_tasks]
                               + [""] * (nrows_p * ncols_p - len(active_tasks)),
                shared_yaxes=True,
            )
            for pi, t in enumerate(active_tasks):
                r = pi // ncols_p + 1
                c = pi % ncols_p + 1
                t_idxs, t_vals = task_curves_p[t]
                t_x = [task_ckpts_p[i] if task_ckpts_p and i < len(task_ckpts_p) else str(i)
                       for i in t_idxs]
                facet_fig.add_trace(go.Scatter(
                    x=t_x, y=t_vals, mode="lines+markers", name="task centroid",
                    legendgroup="centroid", showlegend=(pi == 0),
                    line=dict(color="black", width=2.5, dash="dash"),
                    marker=dict(size=6, color="black"),
                ), row=r, col=c)
                sums, counts = {}, {}
                for fid, _ in assignments[t]:
                    idxs, vals = feature_curves_p[fid]
                    for i, v in zip(idxs, vals):
                        sums[i] = sums.get(i, 0.0) + v
                        counts[i] = counts.get(i, 0) + 1
                if sums:
                    xs_m = sorted(sums.keys())
                    ys_m = [sums[i] / counts[i] for i in xs_m]
                    x_m = [task_ckpts_p[i] if task_ckpts_p and i < len(task_ckpts_p) else str(i)
                           for i in xs_m]
                    facet_fig.add_trace(go.Scatter(
                        x=x_m, y=ys_m, mode="lines+markers", name="mean feature curve",
                        legendgroup="mean_feat", showlegend=(pi == 0),
                        line=dict(color="#FF5722", width=2),
                        marker=dict(size=5, color="#FF5722"),
                    ), row=r, col=c)
            facet_fig.update_layout(
                height=260 * nrows_p + 90,
                margin=dict(l=10, r=10, t=60, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
            )
            facet_fig.update_xaxes(tickangle=45, tickfont=dict(size=7))
            st.plotly_chart(facet_fig, use_container_width=True,
                            key=f"tsg_fc_facet_{key_suffix}")

            # ── Browse features for a chosen task ────────────────────────
            st.markdown("##### Browse features assigned to a task")
            sel_t = st.selectbox("Task centroid to inspect", active_tasks,
                                 key=f"tsg_fc_browse_{key_suffix}")
            sel_desc = TASK_DESCRIPTIONS.get(sel_t)
            if sel_desc:
                st.caption(sel_desc)
            rows = []
            for fid, d in sorted(assignments[sel_t], key=lambda x: x[1]):
                meta = feature_meta_p[fid]
                rows.append({
                    "Feature ID": fid,
                    metric_name: d,
                    "Median Std": meta["median_std"],
                    "N tasks": meta["n_tasks"],
                    "Description": meta["desc"],
                })
            browse_df = pd.DataFrame(rows)
            event = st.dataframe(
                browse_df, hide_index=True, use_container_width=True,
                height=min(500, 35 * len(browse_df) + 40),
                column_config={
                    metric_name: st.column_config.NumberColumn(format="%.4f"),
                    "Median Std": st.column_config.NumberColumn(format="%.4f"),
                },
                on_select="rerun",
                selection_mode="single-row",
                key=f"tsg_fc_table_{key_suffix}_{sel_t}",
            )
            st.caption("Click a row to inspect that feature's curve + samples.")

            sel_rows = event.selection.rows if hasattr(event, "selection") else []
            if not sel_rows:
                return
            sel_fid = browse_df.iloc[sel_rows[0]]["Feature ID"]

            meta = feature_meta_p.get(sel_fid, {})
            samples = feature_samples_p.get(sel_fid, [])
            st.caption(f"Feature {sel_fid}. {meta.get('desc', '')}")

            # ── Inspector plot: task centroid + (un-flipped) feature curve ──
            ins_fig = go.Figure()
            t_idxs, t_vals = task_curves_p[sel_t]
            t_x = [task_ckpts_p[i] if task_ckpts_p and i < len(task_ckpts_p) else str(i)
                   for i in t_idxs]
            ins_fig.add_trace(go.Scatter(
                x=t_x, y=t_vals, mode="lines+markers",
                name=f"task centroid: {sel_t}",
                line=dict(color="#2196F3", width=2.5),
                marker=dict(size=6, color="#2196F3"),
            ))
            fcurve = feature_curves_p.get(sel_fid)
            if fcurve is not None:
                f_idxs, f_vals = fcurve
                f_x = [task_ckpts_p[i] if task_ckpts_p and i < len(task_ckpts_p) else str(i)
                       for i in f_idxs]
                ins_fig.add_trace(go.Scatter(
                    x=f_x, y=f_vals, mode="lines+markers", name=f"feature {sel_fid}",
                    line=dict(color="#FF5722", width=2.5, dash="dash"),
                    marker=dict(size=7, color="#FF5722"),
                ))
            ins_fig.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
                yaxis_title="Z-score", hovermode="x unified",
            )
            ins_fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
            st.plotly_chart(ins_fig, use_container_width=True,
                            key=f"tsg_fc_insfig_{key_suffix}_{sel_t}_{sel_fid}")

            # ── Samples table — sorted by z-norm distance to chosen task ──
            if samples:
                import html as _html
                # Samples whose curve couldn't be resolved sink to the bottom.
                def _samp_key(s):
                    d = (s.get("_task_dists") or {}).get(sel_t)
                    return (d is None, d if d is not None else float("inf"))
                ordered = sorted(samples, key=_samp_key)

                dist_header = f"z-{sample_mode_p} to {sel_t}"
                rows_html = []
                for s in ordered:
                    before = _html.escape(str(s.get("before", "")))
                    word = _html.escape(str(s.get("word", "")))
                    after = _html.escape(str(s.get("after", "")))
                    cos = s.get("cos_sim", 0) or 0
                    act = s.get("activation", 0) or 0
                    d = (s.get("_task_dists") or {}).get(sel_t)
                    d_cell = f"{d:.4f}" if d is not None else "—"
                    rows_html.append(
                        "<tr>"
                        f"<td style='text-align:right;color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{before}</td>"
                        f"<td style='font-weight:bold;padding:4px 6px;white-space:nowrap;'>{word}</td>"
                        f"<td style='color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{after}</td>"
                        f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{d_cell}</td>"
                        f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{cos:.4f}</td>"
                        f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{act:.4f}</td>"
                        "</tr>"
                    )
                table_html = (
                    "<div style='max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:4px;'>"
                    "<table style='width:100%;border-collapse:collapse;'>"
                    "<thead style='position:sticky;top:0;background:#f0f0f0;z-index:1;'><tr>"
                    "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Before</th>"
                    "<th style='border-bottom:2px solid #ddd;padding:6px;'>Word</th>"
                    "<th style='border-bottom:2px solid #ddd;padding:6px;'>After</th>"
                    f"<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>{dist_header}</th>"
                    "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Cos Sim</th>"
                    "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Activation</th>"
                    "</tr></thead><tbody>" + "\n".join(rows_html) + "</tbody></table></div>"
                )
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("No samples available for this feature.")

        n_pos_total = sum(len(v) for v in a_pos.values())
        n_neg_total = sum(len(v) for v in a_neg.values())
        pos_tab, neg_tab = st.tabs([
            f"Positive — {n_pos_total} features",
            f"Negative (flipped) — {n_neg_total} features",
        ])
        with pos_tab:
            _render_assignment_view(a_pos, "positive", "pos")
        with neg_tab:
            _render_assignment_view(a_neg, "negative", "neg")
