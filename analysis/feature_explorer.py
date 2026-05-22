import sys

# Auto-configure to bind on 0.0.0.0 when run via `streamlit run`
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
import html as html_mod

st.set_page_config(page_title="Feature Explorer", layout="wide")

# ── Data loading ──────────────────────────────────────────────────────────────

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_FILE = os.path.join(ANALYSIS_DIR, "cross_model_topics.json")

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


def znorm_curve_metrics(median_probs, overall_perf):
    """Z-normalize both curves, then return (signed MSE, signed area).
    Magnitude: integrated/pointwise distance after picking the better of (curve) vs (flipped curve).
    Sign: positive when the original curve fits better (correlated), negative when the flipped curve
    fits better (anti-correlated). Lower |value| = more similar shape (possibly after flipping).
    Returns (nan, nan) if degenerate (constant curve or too few points)."""
    if not median_probs or not overall_perf:
        return float("nan"), float("nan")
    n = min(len(median_probs), len(overall_perf))
    pairs = [(m, p) for m, p in zip(median_probs[:n], overall_perf[:n])
             if m is not None and p is not None]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    mp = np.array([m for m, _ in pairs], dtype=float)
    op = np.array([p for _, p in pairs], dtype=float)
    mp_std = mp.std()
    op_std = op.std()
    if mp_std == 0 or op_std == 0:
        return float("nan"), float("nan")
    mp_z = (mp - mp.mean()) / mp_std
    op_z = (op - op.mean()) / op_std
    diff_pos = mp_z - op_z       # original alignment
    diff_neg = mp_z + op_z       # flipped feature curve (equivalent to negating mp_z)
    area_pos = float(np.trapezoid(np.abs(diff_pos)))
    area_neg = float(np.trapezoid(np.abs(diff_neg)))
    mse_pos = float(np.mean(diff_pos ** 2))
    mse_neg = float(np.mean(diff_neg ** 2))
    if area_neg < area_pos:
        return -mse_neg, -area_neg
    return mse_pos, area_pos


LOWER_IS_BETTER_METRICS = {"znorm_mse", "znorm_area_between"}


@st.cache_data
def discover_tasks():
    """Find all precomputed_data_* directories and extract task names + all datasets."""
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    tasks = {}
    for d in sorted(glob.glob(pattern)):
        task = os.path.basename(d).replace("precomputed_data_", "")
        json_files = sorted(glob.glob(os.path.join(d, "*.json")))
        if json_files:
            datasets = {}
            for f in json_files:
                name = os.path.splitext(os.path.basename(f))[0]
                datasets[name] = f
            tasks[task] = datasets
    return tasks


@st.cache_data
def load_task_data(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_topic_data(_mtime):
    """Load topic data. _mtime param busts cache when file changes."""
    if os.path.exists(TOPICS_FILE):
        with open(TOPICS_FILE) as f:
            return json.load(f)
    return None


@st.cache_data
def load_all_tasks_for_dataset(dataset_name):
    """Load the same dataset across all tasks. Returns {task_name: data}."""
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    result = {}
    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        fpath = os.path.join(d, f"{dataset_name}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                result[tname] = json.load(f)
    return result


@st.cache_data
def compute_feature_spread(dataset_name, corr_key, current_task):
    """For each feature, compute the std/range of correlations across all tasks.

    Low std → feature behaves consistently across tasks (generic when |curr| is large).
    High std → at least one task differs (specific somewhere).

    Returns {feature_id: {"std", "range", "mean", "current", "n_tasks",
                          "min_task", "min_val", "max_task", "max_val"}}.
    """
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    task_corrs = {}
    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        fpath = os.path.join(d, f"{dataset_name}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            tdata = json.load(f)
        tc = {}
        for feat in tdata["features"]:
            c = feat.get(corr_key)
            if c is not None:
                tc[feat["feature_id"]] = c
        task_corrs[tname] = tc

    if current_task not in task_corrs:
        return {}

    result = {}
    for fid, curr in task_corrs[current_task].items():
        pairs = [(t, task_corrs[t][fid]) for t in task_corrs if fid in task_corrs[t]]
        if len(pairs) < 2:
            continue
        vals = np.array([v for _, v in pairs], dtype=float)
        min_idx = int(np.argmin(vals))
        max_idx = int(np.argmax(vals))
        result[fid] = {
            "std": float(np.std(vals, ddof=0)),
            "range": float(vals.max() - vals.min()),
            "mean": float(vals.mean()),
            "current": float(curr),
            "n_tasks": int(len(vals)),
            "min_task": pairs[min_idx][0], "min_val": float(vals[min_idx]),
            "max_task": pairs[max_idx][0], "max_val": float(vals[max_idx]),
        }
    return result


@st.cache_data
def build_sample_overlap_index(task_path):
    """Build word_id -> list of feature_ids mapping. Keyed by file path for fast caching."""
    with open(task_path) as f:
        features = json.load(f)["features"]
    word_to_feats = {}
    for feat in features:
        fid = feat["feature_id"]
        for s in feat.get("samples", []):
            wid = s.get("word_id")
            if wid:
                word_to_feats.setdefault(wid, []).append(fid)
    return word_to_feats


def find_overlapping_features(sel_feat_id, sel_samples, word_to_feats, features_by_id):
    """Find features sharing >50% of samples with the selected feature."""
    n_samples = len(sel_samples)
    if n_samples == 0:
        return []
    overlap_counts = {}
    for s in sel_samples:
        wid = s.get("word_id")
        if wid and wid in word_to_feats:
            for fid in word_to_feats[wid]:
                if fid != sel_feat_id:
                    overlap_counts[fid] = overlap_counts.get(fid, 0) + 1
    threshold = n_samples * 0.5
    rows = []
    for fid, count in sorted(overlap_counts.items(), key=lambda x: -x[1]):
        if count < threshold:
            continue
        f = features_by_id.get(fid)
        if f:
            rows.append({
                "Partial Spearman": f.get("partial_spearman_corr", 0),
                "Partial Pearson": f.get("partial_pearson_corr", 0),
                "Spearman": f.get("spearman_corr", 0),
                "Pearson": f.get("pearson_corr", 0),
                "Description": f.get("description", ""),
                "Shared Samples": f"{count}/{n_samples}",
                "Feature ID": fid,
            })
    return rows


# ── Sidebar ───────────────────────────────────────────────────────────────────

tasks = discover_tasks()

st.sidebar.title("Feature Explorer")
task_name = st.sidebar.selectbox("Task", list(tasks.keys()))
dataset_name = st.sidebar.selectbox("Dataset", [d for d in tasks[task_name].keys() if "early" in d and "late" in d and "0.7" in d])

data = load_task_data(tasks[task_name][dataset_name])
features = data["features"]
checkpoints = data["checkpoints"]
n_ckpts = len(checkpoints)

# Build correlation table
corr_type = st.sidebar.selectbox(
    "Sort features by",
    # ["partial_pearson_corr", "partial_spearman_corr", "pearson_corr", "spearman_corr", "residual_pearson_corr", "residual_spearman_corr"],
    ["partial_pearson_corr", "partial_spearman_corr", "pearson_corr", "spearman_corr", "znorm_mse", "znorm_area_between", "partial_diff_pearson_corr", "diff_pearson_corr", "partial_diff_spearman_corr", "diff_spearman_corr", ],

    index=0,
)

overall_perf_global = data.get("overall_performance", [])
rows = []
for feat in features:
    median_probs = feat.get("median_probs", [])
    max_median = max(median_probs) if median_probs else 0
    min_median = min(median_probs) if median_probs else 0
    range_median = max_median - min_median
    max_err = feat.get("max_error", 0)
    std_probs = feat.get("std_probs", []) or []
    valid_stds = [s for s in std_probs if s is not None]
    typical_err = float(np.median(valid_stds)) if valid_stds else 0.0
    rel_err = typical_err / range_median if range_median > 1e-6 else float("inf")
    op_for_feat = feat.get("overall_performance", overall_perf_global)
    rows.append(
        {
            "feature_id": feat["feature_id"],
            "description": feat["description"],
            "pearson_corr": feat.get("pearson_corr", 0),
            "spearman_corr": feat.get("spearman_corr", 0),
            # "residual_pearson_corr": feat.get("residual_pearson_corr", 0),
            # "residual_spearman_corr": feat.get("residual_spearman_corr", 0),
            "partial_pearson_corr": feat.get("partial_pearson_corr", 0),
            "partial_spearman_corr": feat.get("partial_spearman_corr", 0),
            "partial_diff_pearson_corr": feat.get("partial_diff_pearson_corr", 0),
            "diff_pearson_corr": feat.get("diff_pearson_corr", 0),
            "partial_diff_spearman_corr": feat.get("partial_diff_spearman_corr", 0),
            "diff_spearman_corr": feat.get("diff_spearman_corr", 0),
            **dict(zip(("znorm_mse", "znorm_area_between"), znorm_curve_metrics(median_probs, op_for_feat))),
            "trend": feat.get("trend_median_probs", ""),
            "max_error": max_err,
            "relative_error": rel_err,
            "acquired": max_median >= 0.5,
        }
    )

corr_df = pd.DataFrame(rows)

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

acquired_filter = st.sidebar.selectbox(
    "Acquisition status",
    ["All", "Acquired (median prob >= 0.5)", "Not acquired (median prob < 0.5)"],
    index=0,
)
if acquired_filter == "Acquired (median prob >= 0.5)":
    corr_df = corr_df[corr_df["acquired"]].reset_index(drop=True)
elif acquired_filter == "Not acquired (median prob < 0.5)":
    corr_df = corr_df[~corr_df["acquired"]].reset_index(drop=True)

rel_err_vals = corr_df["relative_error"].replace([np.inf, -np.inf], np.nan).dropna().values
if len(rel_err_vals) > 0:
    rel_err_cap = float(min(np.quantile(rel_err_vals, 0.99), 5.0))
    rel_err_cap = max(rel_err_cap, 0.1)  # ensure slider is usable even if all values are tiny
    rel_error_threshold = st.sidebar.slider(
        "Max (median std) / (curve range) <=",
        min_value=0.0,
        max_value=rel_err_cap,
        value=rel_err_cap,
        step=rel_err_cap / 100,
        help="Filter features by median error bar width (robust to a few outlier checkpoints) "
             "relative to the curve's own range. Lower = within-checkpoint noise is small "
             "compared to the across-checkpoint signal.",
    )
    corr_df = corr_df[corr_df["relative_error"] <= rel_error_threshold].reset_index(drop=True)
else:
    st.sidebar.caption("No relative-error data available (curves may be flat).")

# ── Spread column (std of corr across all tasks for this dataset) ─────────────
spread_map = compute_feature_spread(dataset_name, corr_type, task_name)
corr_df["spread"] = corr_df["feature_id"].map(
    lambda fid: spread_map[fid]["std"] if fid in spread_map else np.nan
)

# ── Sort mode ─────────────────────────────────────────────────────────────────
sort_mode = st.sidebar.radio(
    "Sort mode",
    ["Correlation strength", "Task-specific first", "Generic first"],
    index=0,
    help="Spread = std of this feature's correlation across tasks. "
         "Low spread + strong |curr| = generic; high spread = task-specific somewhere.",
)

_lower_is_better = corr_type in LOWER_IS_BETTER_METRICS
# For "best first": lower-is-better metrics sort by |value| ascending (small |distance| = better fit,
# sign just indicates correlated vs anti-correlated); correlations sort by |value| descending.
_score = corr_df[corr_type].abs()
if sort_mode == "Correlation strength":
    corr_df["_score"] = _score
    corr_df = corr_df.sort_values("_score", ascending=_lower_is_better).drop(columns="_score").reset_index(drop=True)
elif sort_mode == "Task-specific first":
    # Higher spread first; break ties by stronger correlation (or lower MSE/area)
    corr_df["_sort_key"] = corr_df["spread"].fillna(-1.0)
    corr_df["_score"] = _score
    corr_df = corr_df.sort_values(["_sort_key", "_score"], ascending=[False, _lower_is_better]).drop(
        columns=["_sort_key", "_score"]
    ).reset_index(drop=True)
else:  # Generic first: low spread first
    corr_df["_sort_key"] = corr_df["spread"].fillna(np.inf)
    corr_df["_score"] = _score
    corr_df = corr_df.sort_values(["_sort_key", "_score"], ascending=[True, _lower_is_better]).drop(
        columns=["_sort_key", "_score"]
    ).reset_index(drop=True)

# Split into positive and negative (by the current correlation metric's sign).
# Lower-is-better metrics (MSE, area) are always >= 0, so everything lands in pos_df.
pos_df = corr_df[corr_df[corr_type] >= 0].reset_index(drop=True)
neg_df = corr_df[corr_df[corr_type] < 0].reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(corr_df)}** features shown ({len(pos_df)} pos, {len(neg_df)} neg)")

# ── Task description header ──────────────────────────────────────────────────

task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
st.markdown(f"### {task_name}")
st.caption(task_desc)

_topics_mtime = os.path.getmtime(TOPICS_FILE) if os.path.exists(TOPICS_FILE) else 0
topic_data = load_topic_data(_topics_mtime)
has_topics = topic_data is not None and task_name in topic_data

MATCHES_FILE = os.path.join(ANALYSIS_DIR, "cross_model_matches.json")

@st.cache_data
def load_match_data(_mtime):
    if os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE) as f:
            return json.load(f)
    return None

_matches_mtime = os.path.getmtime(MATCHES_FILE) if os.path.exists(MATCHES_FILE) else 0
match_data = load_match_data(_matches_mtime)
has_matches = match_data is not None

JUMP_FILE = os.path.join(ANALYSIS_DIR, "jump_alignment.json")

@st.cache_data
def load_jump_data(_mtime):
    if os.path.exists(JUMP_FILE):
        with open(JUMP_FILE) as f:
            return json.load(f)
    return None

_jump_mtime = os.path.getmtime(JUMP_FILE) if os.path.exists(JUMP_FILE) else 0
jump_data = load_jump_data(_jump_mtime)
has_jumps = jump_data is not None

tab_names = ["Feature Explorer", "Correlation Distributions", "Task Grouping", "Shared Task Features", "Suppressor Features", "Difficult Features", "Feature Specificity"]
if has_topics:
    tab_names.append("Cross-Model Topics")
if has_matches:
    tab_names.append("Cross-Model Matches")
if has_jumps:
    tab_names.append("Jump Groups")

tabs = st.tabs(tab_names)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Feature Explorer
# ══════════════════════════════════════════════════════════════════════════════

def render_feature_detail(sel_feat, sel_feat_id, data, checkpoints, n_ckpts, features, dataset_name, task_name, task_path, tab_key=""):
    """Render the right-column detail view for a selected feature."""
    st.subheader(f"Feature {sel_feat_id}: {sel_feat['description']}")

    # ── Dual-axis plot ────────────────────────────────────────────────────
    median_probs_raw = sel_feat["median_probs"]
    overall_perf_raw = sel_feat["overall_performance"]
    std_probs_raw = sel_feat.get("std_probs", [0] * n_ckpts)

    valid_mask = [p is not None for p in overall_perf_raw[:len(median_probs_raw)]]
    plot_ckpts = [c for c, v in zip(checkpoints[:len(median_probs_raw)], valid_mask) if v]
    mp = np.array([m for m, v in zip(median_probs_raw, valid_mask) if v])
    op = np.array([p for p, v in zip(overall_perf_raw, valid_mask) if v])
    sp = np.array([s for s, v in zip(std_probs_raw, valid_mask) if v])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=plot_ckpts, y=op, name=f"Task Performance ({data.get('metric', '')})",
                   mode="lines+markers", line=dict(color="#2196F3", width=2), marker=dict(size=8)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=plot_ckpts, y=mp, name="Median Prob (feature)",
                   mode="lines+markers", line=dict(color="#FF5722", width=2), marker=dict(size=8),
                   error_y=dict(type="data", array=sp, arrayminus=sp, visible=True,
                                color="rgba(255,87,34,0.3)", thickness=1.5, width=4)),
        secondary_y=True,
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Task Performance", secondary_y=False, color="#2196F3",
                     range=[min(op) * 0.9, max(op) * 1.1] if len(op) > 0 else None)
    fig.update_yaxes(title_text="Median Probability", secondary_y=True, color="#FF5722",
                     range=[max(0, min(mp) - max(sp) * 1.2), min(1, max(mp) + max(sp) * 1.2)] if len(mp) > 0 else None)

    # ── Z-normalized overlay (shared y-axis) ─────────────────────────────
    znorm_fig = go.Figure()
    if len(mp) >= 3 and mp.std() > 0 and op.std() > 0:
        mp_z = (mp - mp.mean()) / mp.std()
        op_z = (op - op.mean()) / op.std()
        znorm_fig.add_trace(go.Scatter(
            x=plot_ckpts, y=op_z, name="Task Performance (z)",
            mode="lines+markers", line=dict(color="#2196F3", width=2), marker=dict(size=8),
        ))
        znorm_fig.add_trace(go.Scatter(
            x=plot_ckpts, y=mp_z, name="Median Prob (z)",
            mode="lines+markers", line=dict(color="#FF5722", width=2), marker=dict(size=8),
        ))
        znorm_fig.add_trace(go.Scatter(
            x=plot_ckpts + plot_ckpts[::-1],
            y=list(np.maximum(mp_z, op_z)) + list(np.minimum(mp_z, op_z))[::-1],
            fill="toself", fillcolor="rgba(128,128,128,0.18)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="|diff| area", showlegend=True,
        ))
    else:
        znorm_fig.add_annotation(text="Curve too short or constant — z-norm undefined",
                                 xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    znorm_fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            hovermode="x unified",
                            yaxis_title="Z-score (shared scale)")

    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.caption("Raw curves (dual-axis)")
        st.plotly_chart(fig, use_container_width=True)
    with plot_cols[1]:
        st.caption("Z-normalized overlay (shared axis) — shaded = pointwise |diff|")
        st.plotly_chart(znorm_fig, use_container_width=True)

    # ── Correlation stats ─────────────────────────────────────────────────
    znorm_mse, znorm_area = znorm_curve_metrics(median_probs_raw, overall_perf_raw)
    stat_cols = st.columns(6)
    stat_cols[0].metric("Pearson", f"{sel_feat.get('pearson_corr', 0):.4f}")
    stat_cols[1].metric("Spearman", f"{sel_feat.get('spearman_corr', 0):.4f}")
    stat_cols[2].metric("Partial Pearson", f"{sel_feat.get('partial_pearson_corr', 0):.4f}")
    stat_cols[3].metric("Partial Spearman", f"{sel_feat.get('partial_spearman_corr', 0):.4f}")
    stat_cols[4].metric("Z-norm MSE", f"{znorm_mse:+.4f}" if not np.isnan(znorm_mse) else "—",
                        help="Pointwise MSE between z-normalized curves, signed: positive = correlated, "
                             "negative = anti-correlated (flipped curve fits better). Lower |value| = more similar shape.")
    stat_cols[5].metric("Z-norm Area", f"{znorm_area:+.4f}" if not np.isnan(znorm_area) else "—",
                        help="Integrated absolute difference (trapezoid) between z-normalized curves, signed: "
                             "positive = correlated, negative = anti-correlated (flipped curve fits better). "
                             "Lower |value| = more similar shape.")

    # ── Samples table ─────────────────────────────────────────────────────
    st.markdown("---")
    samples = sel_feat.get("samples", [])
    sort_by = st.radio("Sort samples by", ["cos_sim", "activation"], horizontal=True, index=0,
                       help="cos_sim = embedding cosine distance to centroid; activation = SAE activation value",
                       key=f"sort_{tab_key}")
    if samples:
        sample_df = pd.DataFrame(samples).sort_values(sort_by, ascending=False).reset_index(drop=True)
        html_rows = []
        for _, r in sample_df.iterrows():
            before = html_mod.escape(str(r.get("before", "")))
            word = html_mod.escape(str(r.get("word", "")))
            after = html_mod.escape(str(r.get("after", "")))
            cos = r.get("cos_sim", 0)
            act = r.get("activation", 0)
            html_rows.append(
                f"<tr>"
                f"<td style='text-align:right;color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{before}</td>"
                f"<td style='font-weight:bold;padding:4px 6px;white-space:nowrap;'>{word}</td>"
                f"<td style='color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{after}</td>"
                f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{cos:.4f}</td>"
                f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{act:.4f}</td>"
                f"</tr>"
            )
        table_html = (
            "<div style='max-height:400px;overflow-y:auto;border:1px solid #ddd;border-radius:4px;'>"
            "<table style='width:100%;border-collapse:collapse;'>"
            "<thead style='position:sticky;top:0;background:#f0f0f0;z-index:1;'><tr>"
            "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Before Context</th>"
            "<th style='border-bottom:2px solid #ddd;padding:6px;'>Word</th>"
            "<th style='border-bottom:2px solid #ddd;padding:6px;'>After Context</th>"
            "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Cos Sim</th>"
            "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Activation</th>"
            "</tr></thead><tbody>" + "\n".join(html_rows) + "</tbody></table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No samples available for this feature.")

    # ── Correlation across all tasks ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### Correlation across all tasks")
    all_task_data = load_all_tasks_for_dataset(dataset_name)
    cross_task_rows = []
    for tname in sorted(all_task_data.keys()):
        tdata = all_task_data[tname]
        feat_match = next((f for f in tdata["features"] if f["feature_id"] == sel_feat_id), None)
        if feat_match:
            _mse, _area = znorm_curve_metrics(
                feat_match.get("median_probs", []),
                feat_match.get("overall_performance", tdata.get("overall_performance", [])),
            )
            cross_task_rows.append({
                "Partial Spearman": feat_match.get("partial_spearman_corr", 0),
                "Partial Pearson": feat_match.get("partial_pearson_corr", 0),
                "Spearman": feat_match.get("spearman_corr", 0),
                "Pearson": feat_match.get("pearson_corr", 0),
                "Z-norm MSE": _mse,
                "Z-norm Area": _area,
                "Task": tname,
                "Trend": feat_match.get("trend_median_probs", ""),
            })
    if cross_task_rows:
        cross_df_t = pd.DataFrame(cross_task_rows)
        def highlight_current(row):
            if row["Task"] == task_name:
                return ["background-color: rgba(33, 150, 243, 0.15)"] * len(row)
            return [""] * len(row)
        styled = cross_df_t.style.apply(highlight_current, axis=1).format(
            {c: "{:+.4f}" for c in cross_df_t.columns if c not in ("Task", "Trend")}
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=min(400, 35 * len(cross_task_rows) + 40))
    else:
        st.info("Feature not found in other tasks for this dataset.")

    # ── Features sharing >50% samples ──────────────────────────────────
    features_by_id = {f["feature_id"]: f for f in features}
    word_to_feats = build_sample_overlap_index(task_path)
    sel_samples = sel_feat.get("samples", [])
    overlap_rows = find_overlapping_features(sel_feat_id, sel_samples, word_to_feats, features_by_id)
    if overlap_rows:
        st.markdown("---")
        st.markdown("#### Similar features (>50% shared samples)")
        overlap_df = pd.DataFrame(overlap_rows)
        st.dataframe(
            overlap_df.style.format({c: "{:+.4f}" for c in ["Partial Spearman", "Partial Pearson", "Spearman", "Pearson"]}),
            use_container_width=True, hide_index=True,
            height=min(400, 35 * len(overlap_rows) + 40),
        )


def render_feature_list(df, corr_type, key_prefix):
    """Render a feature table and return selected feature_id."""
    if df.empty:
        st.info("No features in this category.")
        return None
    cols = [corr_type, "spread", "description", "trend", "feature_id"]
    cols = [c for c in cols if c in df.columns]
    display = df[cols].copy()
    rename = {corr_type: "Correlation", "spread": "Spread",
              "description": "Description", "trend": "Trend", "feature_id": "ID"}
    display = display.rename(columns=rename)
    display["Correlation"] = display["Correlation"].apply(lambda v: f"{'+' if v >= 0 else ''}{v:.4f}")
    if "Spread" in display.columns:
        display["Spread"] = display["Spread"].apply(
            lambda v: "—" if pd.isna(v) else f"{v:.3f}"
        )
    event = st.dataframe(display, use_container_width=True, hide_index=True,
                         on_select="rerun", selection_mode="single-row", height=300,
                         key=f"table_{key_prefix}")
    sel_rows = event.selection.rows if event.selection.rows else [0]
    return df.iloc[sel_rows[0]]["feature_id"]


with tabs[0]:
    fe_pos_tab, fe_neg_tab = st.tabs(["Positively Correlated", "Negatively Correlated"])

    for direction_df, dir_tab, dir_key in [(pos_df, fe_pos_tab, "pos"), (neg_df, fe_neg_tab, "neg")]:
        with dir_tab:
            if direction_df.empty:
                st.info("No features in this category.")
            else:
                sel_feat_id = render_feature_list(direction_df, corr_type, dir_key)
                if sel_feat_id is not None:
                    sel_feat = next((f for f in features if f["feature_id"] == sel_feat_id), None)
                    if sel_feat:
                        st.markdown("---")
                        render_feature_detail(sel_feat, sel_feat_id, data, checkpoints, n_ckpts,
                                              features, dataset_name, task_name,
                                              task_path=tasks[task_name][dataset_name], tab_key=dir_key)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Correlation Distributions
# ══════════════════════════════════════════════════════════════════════════════

DIST_EXCLUDE = {("squad", "OLMo3")}

@st.cache_data
def load_all_distributions(dataset_name, corr_key):
    """Load correlation values for all features across all tasks for a dataset."""
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    result = {}
    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        if any(tname == t and dataset_name.startswith(m) for t, m in DIST_EXCLUDE):
            continue
        fpath = os.path.join(d, f"{dataset_name}.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                tdata = json.load(f)
            corrs = [f.get(corr_key, 0) for f in tdata["features"] if f.get(corr_key) is not None]
            perf = tdata.get("overall_performance", [])
            valid_perf = [p for p in perf if p is not None]
            avg_perf = np.mean(valid_perf) if valid_perf else None
            perf_range = (max(valid_perf) - min(valid_perf)) if len(valid_perf) >= 2 else 0
            result[tname] = {"corrs": corrs, "avg_perf": avg_perf, "perf_range": perf_range}
    return result


with tabs[1]:
    st.subheader("Correlation distributions across tasks")
    st.caption("Compare how feature correlations are distributed per task. "
               "Tasks where the model is already strong tend to have smaller performance range "
               "across checkpoints, leading to noisier/weaker correlations.")

    # Find all datasets available across tasks
    all_datasets = set()
    for tname, tpaths in tasks.items():
        all_datasets.update(tpaths.keys())
    all_datasets = sorted(all_datasets)
    print([d for d in all_datasets if "3000" in d])

    dist_datasets = st.multiselect("Datasets to compare", [d for d in all_datasets if "3000" in d],
                                   default=[d for d in all_datasets if "3000" in d],
                                   key="dist_datasets")
    dist_corr = st.selectbox("Correlation metric", [
        "partial_spearman_corr", "partial_pearson_corr", "spearman_corr", "pearson_corr",
        "residual_spearman_corr", "residual_pearson_corr",
    ], index=0, key="dist_corr")

    DS_COLORS = [
        "rgba(33, 150, 243, 0.6)",   # blue
        "rgba(255, 87, 34, 0.6)",    # orange
        "rgba(76, 175, 80, 0.6)",    # green
        "rgba(156, 39, 176, 0.6)",   # purple
        "rgba(255, 193, 7, 0.6)",    # amber
    ]

    # Load all data
    all_dist = {}
    for ds_name in dist_datasets:
        dd = load_all_distributions(ds_name, dist_corr)
        if dd:
            all_dist[ds_name] = dd

    if not all_dist:
        st.info("No data found for selected datasets.")
    else:
        # Union of all tasks across selected datasets
        all_task_names = sorted(set().union(*(d.keys() for d in all_dist.values())))

        # ── Overlaid histograms per task ──────────────────────────────────
        n_tasks = len(all_task_names)
        n_cols = 4
        n_rows = (n_tasks + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=all_task_names + [""] * (n_rows * n_cols - n_tasks),
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
        )

        for i, tname in enumerate(all_task_names):
            row = i // n_cols + 1
            col = i % n_cols + 1
            for j, ds_name in enumerate(dist_datasets):
                dd = all_dist.get(ds_name, {})
                if tname not in dd or not dd[tname]["corrs"]:
                    continue
                color = DS_COLORS[j % len(DS_COLORS)]
                fig.add_trace(
                    go.Histogram(
                        x=dd[tname]["corrs"],
                        nbinsx=30,
                        marker_color=color,
                        opacity=0.6,
                        name=ds_name,
                        showlegend=(i == 0),
                        legendgroup=ds_name,
                    ),
                    row=row, col=col,
                )
            fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1,
                          row=row, col=col)

        fig.update_layout(
            height=250 * n_rows,
            margin=dict(l=10, r=10, t=40, b=10),
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        fig.update_xaxes(title_text=dist_corr, row=n_rows)
        fig.update_yaxes(title_text="Count", col=1)

        st.plotly_chart(fig, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────
        st.markdown("#### Summary")
        summary_rows = []
        for ds_name in dist_datasets:
            dd = all_dist.get(ds_name, {})
            for tname in all_task_names:
                if tname not in dd or not dd[tname]["corrs"]:
                    continue
                d = dd[tname]
                corrs = d["corrs"]
                summary_rows.append({
                    "Dataset": ds_name,
                    "Task": tname,
                    "Avg Perf": d["avg_perf"],
                    "Perf Range": d["perf_range"],
                    "Median |corr|": float(np.median(np.abs(corrs))),
                    "Mean corr": float(np.mean(corrs)),
                    "Std corr": float(np.std(corrs)),
                    "N features": len(corrs),
                })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(
                summary_df.style.format({
                    "Avg Perf": "{:.3f}", "Perf Range": "{:.3f}",
                    "Median |corr|": "{:.4f}", "Mean corr": "{:+.4f}", "Std corr": "{:.4f}",
                }),
                use_container_width=True, hide_index=True,
            )

        # ── Overlaid scatter: avg perf vs median |corr| ───────────────────
        st.markdown("#### Avg performance vs median |correlation|")
        scatter_fig = go.Figure()
        for j, ds_name in enumerate(dist_datasets):
            dd = all_dist.get(ds_name, {})
            pts = []
            for tname in all_task_names:
                if tname not in dd or not dd[tname]["corrs"]:
                    continue
                d = dd[tname]
                pts.append({
                    "task": tname,
                    "avg_perf": d["avg_perf"],
                    "med_corr": float(np.median(np.abs(d["corrs"]))),
                })
            if pts:
                color = DS_COLORS[j % len(DS_COLORS)].replace("0.6", "0.9")
                scatter_fig.add_trace(go.Scatter(
                    x=[p["avg_perf"] for p in pts],
                    y=[p["med_corr"] for p in pts],
                    text=[p["task"] for p in pts],
                    mode="markers+text",
                    textposition="top center",
                    name=ds_name,
                    marker=dict(size=10, color=color),
                ))
        scatter_fig.update_layout(
            height=400,
            xaxis_title="Average performance across checkpoints",
            yaxis_title=f"Median |{dist_corr}|",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab: Task Grouping (within one model, group tasks by shared feature correlations)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def build_feature_task_matrix(dataset_name, corr_key):
    """Build a feature × task correlation matrix for a single dataset across all tasks.

    Returns (matrix DataFrame with features as rows and tasks as columns,
             feature_info dict mapping feature_id -> description).
    """
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    task_feature_corrs = {}  # {task_name: {feature_id: corr}}
    feature_info = {}

    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        fpath = os.path.join(d, f"{dataset_name}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            tdata = json.load(f)
        task_corrs = {}
        for feat in tdata["features"]:
            fid = feat["feature_id"]
            c = feat.get(corr_key)
            if c is not None:
                task_corrs[fid] = c
                if fid not in feature_info:
                    feature_info[fid] = feat.get("description", "")
        task_feature_corrs[tname] = task_corrs

    if not task_feature_corrs:
        return None, {}

    all_tasks = sorted(task_feature_corrs.keys())
    all_features = sorted(set().union(*(tc.keys() for tc in task_feature_corrs.values())))

    # Build matrix: features × tasks, NaN where missing
    matrix = pd.DataFrame(index=all_features, columns=all_tasks, dtype=float)
    for tname in all_tasks:
        for fid, c in task_feature_corrs[tname].items():
            matrix.loc[fid, tname] = c

    return matrix, feature_info


@st.cache_data
def compute_task_grouping(dataset_name, corr_key, min_abs_corr):
    """Compute task-task similarity based on shared feature correlations.

    For each pair of tasks, find features present in both with |corr| >= min_abs_corr
    in at least one task. Compute:
    - Agreement score: fraction of shared strong features with same sign
    - Opposition score: fraction with opposite sign
    - Cosine similarity of correlation vectors
    """
    matrix, feature_info = build_feature_task_matrix(dataset_name, corr_key)
    if matrix is None:
        return None, None, None, {}

    task_names = list(matrix.columns)
    n = len(task_names)

    # Cosine similarity matrix
    cos_sim = pd.DataFrame(np.zeros((n, n)), index=task_names, columns=task_names)
    # Shared feature details for each pair
    pair_details = {}

    for i in range(n):
        for j in range(i, n):
            t1, t2 = task_names[i], task_names[j]
            v1 = matrix[t1].values
            v2 = matrix[t2].values
            # Only use features present in both tasks and strong in at least one
            both_valid = ~np.isnan(v1) & ~np.isnan(v2)
            strong = (np.abs(v1) >= min_abs_corr) | (np.abs(v2) >= min_abs_corr)
            mask = both_valid & strong

            if mask.sum() == 0:
                cos_sim.loc[t1, t2] = 0
                cos_sim.loc[t2, t1] = 0
                continue

            a = v1[mask]
            b = v2[mask]
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            sim = dot / norm if norm > 0 else 0
            cos_sim.loc[t1, t2] = sim
            cos_sim.loc[t2, t1] = sim

            # Track features
            feat_ids = np.array(matrix.index)[mask]
            same_sign = []
            opp_sign = []
            for fi, ca, cb in zip(feat_ids, a, b):
                entry = {"feature_id": fi, "desc": feature_info.get(fi, ""),
                         f"{t1}_corr": ca, f"{t2}_corr": cb}
                if ca * cb > 0:
                    same_sign.append(entry)
                elif ca * cb < 0:
                    opp_sign.append(entry)

            same_sign.sort(key=lambda x: -min(abs(x[f"{t1}_corr"]), abs(x[f"{t2}_corr"])))
            opp_sign.sort(key=lambda x: -min(abs(x[f"{t1}_corr"]), abs(x[f"{t2}_corr"])))
            pair_details[(t1, t2)] = {"same": same_sign, "opposite": opp_sign, "n_shared": int(mask.sum())}
            if i != j:
                pair_details[(t2, t1)] = pair_details[(t1, t2)]

    return cos_sim, pair_details, task_names, feature_info


with tabs[2]:
    st.subheader("Task Grouping by Shared Features")
    st.caption("For a single model (dataset), find tasks that share features with correlated "
               "behavior. Tasks are grouped by cosine similarity of their feature correlation vectors.")

    # Dataset selector
    all_datasets_tg = set()
    for tname, tpaths in tasks.items():
        all_datasets_tg.update(tpaths.keys())
    all_datasets_tg = sorted(all_datasets_tg)

    tg_dataset = st.selectbox("Dataset (model)", all_datasets_tg,
                              index=all_datasets_tg.index(dataset_name) if dataset_name in all_datasets_tg else 0,
                              key="tg_dataset")
    tg_col1, tg_col2 = st.columns(2)
    with tg_col1:
        tg_corr = st.selectbox("Correlation metric", [
            "partial_pearson_corr", "partial_spearman_corr", "pearson_corr", "spearman_corr",
        ], index=0, key="tg_corr")
    with tg_col2:
        tg_min_corr = st.slider("Min |correlation| threshold", 0.0, 0.8, 0.3, 0.05,
                                help="Features must have |corr| >= this in at least one task to be included",
                                key="tg_min_corr")

    cos_sim, pair_details, tg_task_names, tg_feat_info = compute_task_grouping(
        tg_dataset, tg_corr, tg_min_corr)

    if cos_sim is None:
        st.info("No data found for selected dataset.")
    else:
        # ── Heatmap ──────────────────────────────────────────────────────
        st.markdown("#### Task-Task Similarity Heatmap")
        st.caption("Cosine similarity of feature correlation vectors between task pairs. "
                   "Blue = similar feature profiles, Red = opposite.")

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=cos_sim.values,
            x=cos_sim.columns.tolist(),
            y=cos_sim.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
            text=np.round(cos_sim.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        heatmap_fig.update_layout(
            height=50 + 35 * len(tg_task_names),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # ── Clustering ────────────────────────────────────────────────────
        _cluster_labels = None
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            sim_vals = cos_sim.values.astype(float)
            dist_matrix = np.clip(1.0 - sim_vals, 0, 2)
            np.fill_diagonal(dist_matrix, 0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2

            condensed = squareform(dist_matrix, checks=False)
            Z = linkage(condensed, method="ward")

            # Find best k via silhouette score
            best_k, best_score = 2, -1
            max_k = min(10, len(tg_task_names) - 1)
            silhouette_scores = {}
            try:
                from sklearn.metrics import silhouette_score
                for k in range(2, max_k + 1):
                    labels_k = fcluster(Z, t=k, criterion="maxclust")
                    if len(set(labels_k)) < 2:
                        continue
                    sc = silhouette_score(dist_matrix, labels_k, metric="precomputed")
                    silhouette_scores[k] = sc
                    if sc > best_score:
                        best_score = sc
                        best_k = k
            except ImportError:
                pass

            if silhouette_scores:
                st.caption(f"Suggested k={best_k} (silhouette={best_score:.3f})")

            n_clusters = st.slider("Number of clusters", 2, max_k,
                                   best_k, key="tg_n_clusters")
            _cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

            # Show silhouette for chosen k
            if n_clusters in silhouette_scores:
                st.caption(f"Silhouette score for k={n_clusters}: {silhouette_scores[n_clusters]:.3f}")

            cluster_df = pd.DataFrame({"Task": tg_task_names, "Cluster": _cluster_labels})
            cluster_df = cluster_df.sort_values("Cluster").reset_index(drop=True)

            cluster_cols = st.columns(n_clusters)
            for ci in range(1, n_clusters + 1):
                with cluster_cols[ci - 1]:
                    members = cluster_df[cluster_df["Cluster"] == ci]["Task"].tolist()
                    st.markdown(f"**Cluster {ci}** ({len(members)})")
                    for m in members:
                        st.markdown(f"- {m}")

        except ImportError:
            st.warning("Install `scipy` for clustering: `pip install scipy`")

        # ── 2D projection (PCA) ──────────────────────────────────────────
        st.markdown("#### 2D Projection of Task Correlation Vectors")
        st.caption("Each task is a point in feature-correlation space, projected to 2D via PCA.")

        try:
            from sklearn.decomposition import PCA

            # Build task vectors: for each task, its correlation across all features (NaN → 0)
            task_vectors = []
            for t in tg_task_names:
                v = cos_sim.loc[t].values.astype(float)
                task_vectors.append(v)
            task_vectors = np.array(task_vectors)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(task_vectors)

            # Color by cluster if available
            if _cluster_labels is not None:
                colors = _cluster_labels.tolist()
            else:
                colors = [1] * len(tg_task_names)

            COLOR_PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
                             "#FFC107", "#00BCD4", "#E91E63", "#8BC34A",
                             "#FF9800", "#607D8B"]

            pca_fig = go.Figure()
            unique_clusters = sorted(set(colors))
            for ci in unique_clusters:
                mask = [c == ci for c in colors]
                pca_fig.add_trace(go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1],
                    text=[t for t, m in zip(tg_task_names, mask) if m],
                    mode="markers+text",
                    textposition="top center",
                    marker=dict(size=12, color=COLOR_PALETTE[(ci - 1) % len(COLOR_PALETTE)]),
                    name=f"Cluster {ci}",
                ))

            pca_fig.update_layout(
                height=500,
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(pca_fig, use_container_width=True)
        except ImportError:
            st.warning("Install `scikit-learn` for PCA projection: `pip install scikit-learn`")

        # ── Most similar and most opposite pairs ─────────────────────────
        pairs_list = []
        for i in range(len(tg_task_names)):
            for j in range(i + 1, len(tg_task_names)):
                t1, t2 = tg_task_names[i], tg_task_names[j]
                sim = cos_sim.loc[t1, t2]
                det = pair_details.get((t1, t2), {})
                pairs_list.append({
                    "Task 1": t1, "Task 2": t2,
                    "Cosine Sim": sim,
                    "Shared Features": det.get("n_shared", 0),
                    "Same Sign": len(det.get("same", [])),
                    "Opposite Sign": len(det.get("opposite", [])),
                })

        pairs_df = pd.DataFrame(pairs_list)

        st.markdown("#### Task Pairs — Same Feature Profiles")
        st.caption("Task pairs where features tend to correlate in the same direction.")
        top_similar = pairs_df.sort_values("Cosine Sim", ascending=False).head(15).reset_index(drop=True)
        st.dataframe(top_similar.style.format({"Cosine Sim": "{:+.3f}"}),
                      use_container_width=True, hide_index=True)

        st.markdown("#### Task Pairs — Opposite Feature Profiles")
        st.caption("Task pairs where features correlate in opposite directions.")
        top_opposite = pairs_df.sort_values("Cosine Sim", ascending=True).head(15).reset_index(drop=True)
        st.dataframe(top_opposite.style.format({"Cosine Sim": "{:+.3f}"}),
                      use_container_width=True, hide_index=True)

        # ── Drill into a pair ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Explore a Task Pair")
        pair_options = [f"{r['Task 1']}  ↔  {r['Task 2']}  (sim={r['Cosine Sim']:+.3f})"
                        for _, r in pairs_df.sort_values("Cosine Sim", key=abs, ascending=False).iterrows()]
        pair_keys = [(r["Task 1"], r["Task 2"])
                     for _, r in pairs_df.sort_values("Cosine Sim", key=abs, ascending=False).iterrows()]

        sel_pair_idx = st.selectbox("Select pair", range(len(pair_options)),
                                    format_func=lambda i: pair_options[i], key="tg_pair")
        sel_t1, sel_t2 = pair_keys[sel_pair_idx]
        det = pair_details.get((sel_t1, sel_t2), {})

        same_feats = det.get("same", [])
        opp_feats = det.get("opposite", [])

        feat_same_tab, feat_opp_tab = st.tabs([
            f"Same Direction ({len(same_feats)} features)",
            f"Opposite Direction ({len(opp_feats)} features)",
        ])

        with feat_same_tab:
            if same_feats:
                same_df = pd.DataFrame(same_feats)
                same_df = same_df.rename(columns={
                    "feature_id": "Feature ID", "desc": "Description",
                    f"{sel_t1}_corr": f"{sel_t1} corr", f"{sel_t2}_corr": f"{sel_t2} corr",
                })
                st.dataframe(
                    same_df.style.format({f"{sel_t1} corr": "{:+.4f}", f"{sel_t2} corr": "{:+.4f}"}),
                    use_container_width=True, hide_index=True,
                    height=min(600, 35 * len(same_feats) + 40),
                )
            else:
                st.info("No features with same-direction correlations.")

        with feat_opp_tab:
            if opp_feats:
                opp_df = pd.DataFrame(opp_feats)
                opp_df = opp_df.rename(columns={
                    "feature_id": "Feature ID", "desc": "Description",
                    f"{sel_t1}_corr": f"{sel_t1} corr", f"{sel_t2}_corr": f"{sel_t2} corr",
                })
                st.dataframe(
                    opp_df.style.format({f"{sel_t1} corr": "{:+.4f}", f"{sel_t2} corr": "{:+.4f}"}),
                    use_container_width=True, hide_index=True,
                    height=min(600, 35 * len(opp_feats) + 40),
                )
            else:
                st.info("No features with opposite-direction correlations.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab: Shared Task Features
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def find_common_correlated_features(dataset_name, corr_key, selected_tasks, min_pos_corr, max_neg_corr):
    """Find features correlated in the same direction across ALL selected tasks.

    Returns (pos_df, neg_df) — features positively correlated in all, and negatively in all.
    min_pos_corr: minimum correlation for positive features (e.g. 0.1 means corr >= 0.1)
    max_neg_corr: maximum |correlation| for negative features (e.g. 0.8 means corr >= -0.8)
    """
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    task_corrs = {}  # {task: {feature_id: corr}}
    feature_descs = {}
    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        if tname not in selected_tasks:
            continue
        fpath = os.path.join(d, f"{dataset_name}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            tdata = json.load(f)
        corrs = {}
        for feat in tdata["features"]:
            fid = feat["feature_id"]
            c = feat.get(corr_key)
            if c is not None:
                corrs[fid] = c
                if fid not in feature_descs:
                    feature_descs[fid] = feat.get("description", "")
        task_corrs[tname] = corrs

    if len(task_corrs) < len(selected_tasks):
        return pd.DataFrame(), pd.DataFrame()

    # Find features present in all selected tasks
    common_fids = None
    for tname in selected_tasks:
        tc = task_corrs.get(tname, {})
        common_fids = set(tc.keys()) if common_fids is None else common_fids & set(tc.keys())

    if not common_fids:
        return pd.DataFrame(), pd.DataFrame()

    pos_rows, neg_rows = [], []
    for fid in sorted(common_fids):
        vals = [task_corrs[t][fid] for t in selected_tasks]
        all_pos = all(c >= min_pos_corr for c in vals)
        all_neg = all(-max_neg_corr >= c for c in vals)
        if not all_pos and not all_neg:
            continue
        row = {"Feature ID": fid, "Description": feature_descs.get(fid, "")}
        for tname, c in zip(selected_tasks, vals):
            row[tname] = c
        row["Min |Corr|"] = min(abs(c) for c in vals)
        if all_pos:
            pos_rows.append(row)
        else:
            neg_rows.append(row)

    pos_df = pd.DataFrame(pos_rows)
    neg_df = pd.DataFrame(neg_rows)
    if not pos_df.empty:
        pos_df = pos_df.sort_values("Min |Corr|", ascending=False).reset_index(drop=True)
    if not neg_df.empty:
        neg_df = neg_df.sort_values("Min |Corr|", ascending=False).reset_index(drop=True)
    return pos_df, neg_df


with tabs[3]:
    st.subheader("Shared Task Features")
    st.caption("Find features consistently correlated (positive or negative) with performance "
               "across ALL selected benchmark tasks.")

    # Dataset selector
    all_datasets_stf = set()
    for tname, tpaths in tasks.items():
        all_datasets_stf.update(tpaths.keys())
    all_datasets_stf = sorted(all_datasets_stf)

    stf_dataset = st.selectbox("Dataset (model)", all_datasets_stf,
                               index=all_datasets_stf.index(dataset_name) if dataset_name in all_datasets_stf else 0,
                               key="stf_dataset")

    stf_col1, stf_col2 = st.columns(2)
    with stf_col1:
        stf_corr = st.selectbox("Correlation metric", [
            "partial_pearson_corr", "partial_spearman_corr", "pearson_corr", "spearman_corr",
        ], index=0, key="stf_corr")
    with stf_col2:
        stf_min_pos = st.slider("Min correlation (positive)", 0.0, 1.0, 0.1, 0.05,
                                help="Only include positively correlated features with corr >= this value in all tasks",
                                key="stf_min_pos")
        stf_max_neg = st.slider("Max |correlation| (negative)", 0.1, 1.0, 1.0, 0.05,
                                help="Exclude negatively correlated features with |corr| above this value",
                                key="stf_max_neg")

    # Task multi-select
    all_task_names_stf = sorted(tasks.keys())
    stf_selected_tasks = st.multiselect(
        "Select benchmark tasks",
        all_task_names_stf,
        default=all_task_names_stf[:3],
        key="stf_tasks",
    )

    if len(stf_selected_tasks) < 2:
        st.warning("Select at least 2 tasks to find shared features.")
    else:
        stf_pos_df, stf_neg_df = find_common_correlated_features(
            stf_dataset, stf_corr, stf_selected_tasks, stf_min_pos, stf_max_neg)

        stf_pos_tab, stf_neg_tab = st.tabs([
            f"Positively Correlated ({len(stf_pos_df)})",
            f"Negatively Correlated ({len(stf_neg_df)})",
        ])

        for stf_dir_df, stf_dir_tab, stf_cmap in [
            (stf_pos_df, stf_pos_tab, "Greens"),
            (stf_neg_df, stf_neg_tab, "Reds"),
        ]:
            with stf_dir_tab:
                if stf_dir_df.empty:
                    st.info("No features found that are consistently correlated across all "
                            "selected tasks with the current threshold.")
                else:
                    st.success(f"**{len(stf_dir_df)}** features across all "
                               f"{len(stf_selected_tasks)} selected tasks")

                    format_dict = {t: "{:+.4f}" for t in stf_selected_tasks}
                    format_dict["Min |Corr|"] = "{:.4f}"

                    st.dataframe(
                        stf_dir_df.style.format(format_dict).background_gradient(
                            subset=stf_selected_tasks, cmap=stf_cmap,
                            vmin=0, vmax=1.0,
                        ),
                        use_container_width=True,
                        hide_index=True,
                        height=min(800, 35 * len(stf_dir_df) + 40),
                    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab: Suppressor Features
# (features whose partial Pearson is strongly negative while raw Pearson is
# negative or weakly positive — the partial correlation reveals a hidden effect)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("Suppressor Features")
    st.caption("Features with strongly negative partial Pearson correlation whose raw Pearson "
               "is negative or weakly positive (< 0.5). These are cases where controlling for "
               "other features reveals a negative relationship that the raw correlation masks.")

    supp_col1, supp_col2 = st.columns(2)
    with supp_col1:
        supp_max_partial = st.slider(
            "Max partial Pearson (must be <= this)",
            -1.0, 0.0, -0.3, 0.05,
            help="Only features with partial_pearson_corr at or below this value",
            key="supp_max_partial",
        )
    with supp_col2:
        supp_max_pearson = st.slider(
            "Max raw Pearson (must be < this)",
            -1.0, 1.0, 0.5, 0.05,
            help="Only features whose raw pearson_corr is below this value",
            key="supp_max_pearson",
        )

    supp_rows = []
    for feat in features:
        pp = feat.get("partial_pearson_corr")
        p = feat.get("pearson_corr")
        if pp is None or p is None:
            continue
        if pp <= supp_max_partial and p < supp_max_pearson:
            median_probs = feat.get("median_probs", [])
            max_median = max(median_probs) if median_probs else 0
            supp_rows.append({
                "feature_id": feat["feature_id"],
                "description": feat["description"],
                "partial_pearson_corr": pp,
                "pearson_corr": p,
                "partial_spearman_corr": feat.get("partial_spearman_corr", 0),
                "spearman_corr": feat.get("spearman_corr", 0),
                "trend": feat.get("trend_median_probs", ""),
                "gap": p - pp,  # how much the partial is below raw
                "acquired": max_median >= 0.5,
            })

    if not supp_rows:
        st.info("No features match the current thresholds.")
    else:
        supp_df = pd.DataFrame(supp_rows).sort_values("partial_pearson_corr", ascending=True).reset_index(drop=True)
        st.markdown(f"**{len(supp_df)}** features match (sorted by most negative partial Pearson)")

        display = supp_df[["partial_pearson_corr", "pearson_corr", "gap", "description", "trend", "feature_id"]].copy()
        display.columns = ["Partial Pearson", "Pearson", "Gap (P−PP)", "Description", "Trend", "ID"]
        event = st.dataframe(
            display.style.format({
                "Partial Pearson": "{:+.4f}", "Pearson": "{:+.4f}", "Gap (P−PP)": "{:+.4f}",
            }),
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", height=400,
            key="table_supp",
        )
        sel_rows = event.selection.rows if event.selection.rows else [0]
        sel_feat_id = supp_df.iloc[sel_rows[0]]["feature_id"]
        sel_feat = next((f for f in features if f["feature_id"] == sel_feat_id), None)
        if sel_feat:
            st.markdown("---")
            render_feature_detail(sel_feat, sel_feat_id, data, checkpoints, n_ckpts,
                                  features, dataset_name, task_name,
                                  task_path=tasks[task_name][dataset_name], tab_key="supp")


# ══════════════════════════════════════════════════════════════════════════════
# Tab: Difficult Features
# (features with strongly positive raw Pearson but mildly negative partial
# Pearson — the raw trend is explained away when controlling for other features)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("Difficult Features")
    st.caption("Features with high positive raw Pearson correlation but negative partial Pearson. "
               "These look informative in isolation but their apparent effect is explained away "
               "(or reversed) when controlling for other features.")

    diff_col1, diff_col2, diff_col3 = st.columns(3)
    with diff_col1:
        diff_min_pearson = st.slider(
            "Min raw Pearson (must be >= this)",
            0.0, 1.0, 0.5, 0.05,
            help="Only features whose raw pearson_corr is at or above this value",
            key="diff_min_pearson",
        )
    with diff_col2:
        diff_max_partial = st.slider(
            "Max partial Pearson (must be < this)",
            -1.0, 0.5, 0.0, 0.05,
            help="Only features with partial_pearson_corr below this value",
            key="diff_max_partial",
        )
    with diff_col3:
        diff_min_partial = st.slider(
            "Min partial Pearson (must be >= this)",
            -1.0, 0.5, -0.5, 0.05,
            help="Lower bound on partial_pearson_corr (e.g. −0.3 keeps only 'low negative' values)",
            key="diff_min_partial",
        )

    diff_rows = []
    for feat in features:
        pp = feat.get("partial_pearson_corr")
        p = feat.get("pearson_corr")
        if pp is None or p is None:
            continue
        if p >= diff_min_pearson and diff_min_partial <= pp < diff_max_partial:
            median_probs = feat.get("median_probs", [])
            max_median = max(median_probs) if median_probs else 0
            diff_rows.append({
                "feature_id": feat["feature_id"],
                "description": feat["description"],
                "partial_pearson_corr": pp,
                "pearson_corr": p,
                "partial_spearman_corr": feat.get("partial_spearman_corr", 0),
                "spearman_corr": feat.get("spearman_corr", 0),
                "trend": feat.get("trend_median_probs", ""),
                "gap": p - pp,
                "acquired": max_median >= 0.5,
            })

    if not diff_rows:
        st.info("No features match the current thresholds.")
    else:
        diff_df = pd.DataFrame(diff_rows).sort_values("gap", ascending=False).reset_index(drop=True)
        st.markdown(f"**{len(diff_df)}** features match (sorted by largest gap: Pearson − Partial Pearson)")

        display = diff_df[["pearson_corr", "partial_pearson_corr", "gap", "description", "trend", "feature_id"]].copy()
        display.columns = ["Pearson", "Partial Pearson", "Gap (P−PP)", "Description", "Trend", "ID"]
        event = st.dataframe(
            display.style.format({
                "Pearson": "{:+.4f}", "Partial Pearson": "{:+.4f}", "Gap (P−PP)": "{:+.4f}",
            }),
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", height=400,
            key="table_diff",
        )
        sel_rows = event.selection.rows if event.selection.rows else [0]
        sel_feat_id = diff_df.iloc[sel_rows[0]]["feature_id"]
        sel_feat = next((f for f in features if f["feature_id"] == sel_feat_id), None)
        if sel_feat:
            st.markdown("---")
            render_feature_detail(sel_feat, sel_feat_id, data, checkpoints, n_ckpts,
                                  features, dataset_name, task_name,
                                  task_path=tasks[task_name][dataset_name], tab_key="diff")


# ══════════════════════════════════════════════════════════════════════════════
# Tab: Feature Specificity
# (features grouped by how many tasks they correlate strongly with —
#  task-specific at one end, common/general at the other)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.subheader("Feature Specificity")
    st.caption("For each feature, compute the standard deviation of its correlation across all "
               "tasks. Low spread + strong current correlation = generic (feature behaves the same "
               "everywhere). High spread = at least one task stands out. Requires |corr| on the "
               "reference task to be above a threshold so 'always near zero' features are filtered out.")

    all_datasets_fs = set()
    for tname, tpaths in tasks.items():
        all_datasets_fs.update(tpaths.keys())
    all_datasets_fs = sorted(all_datasets_fs)

    fs_dataset = st.selectbox("Dataset (model)", all_datasets_fs,
                              index=all_datasets_fs.index(dataset_name) if dataset_name in all_datasets_fs else 0,
                              key="fs_dataset")

    fs_task_list = sorted(tasks.keys())
    fs_current_task = st.selectbox("Reference task",
                                   fs_task_list,
                                   index=fs_task_list.index(task_name) if task_name in fs_task_list else 0,
                                   key="fs_current_task")

    fs_col1, fs_col2 = st.columns(2)
    with fs_col1:
        fs_corr = st.selectbox("Correlation metric", [
            "partial_spearman_corr", "partial_pearson_corr", "spearman_corr", "pearson_corr",
        ], index=0, key="fs_corr")
    with fs_col2:
        fs_min_current = st.slider("Min |corr| on reference task", 0.0, 1.0, 0.3, 0.05,
                                   help="Features must correlate at least this much with the reference task",
                                   key="fs_min_current")

    fs_spread = compute_feature_spread(fs_dataset, fs_corr, fs_current_task)

    fs_feat_desc = {}
    fs_current_path = tasks.get(fs_current_task, {}).get(fs_dataset)
    if fs_current_path:
        with open(fs_current_path) as _f:
            for feat in json.load(_f)["features"]:
                fs_feat_desc[feat["feature_id"]] = feat.get("description", "")

    fs_rows = []
    for fid, info in fs_spread.items():
        if abs(info["current"]) < fs_min_current:
            continue
        fs_rows.append({
            "Feature ID": fid,
            "Description": fs_feat_desc.get(fid, ""),
            "Current Corr": info["current"],
            "Spread (std)": info["std"],
            "Range": info["range"],
            "Mean Corr": info["mean"],
            "Min": f"{info['min_val']:+.2f} ({info['min_task']})",
            "Max": f"{info['max_val']:+.2f} ({info['max_task']})",
            "N Tasks": info["n_tasks"],
        })

    if not fs_rows:
        st.info(f"No features have |corr| >= {fs_min_current} on {fs_current_task}.")
    else:
        fs_full_df = pd.DataFrame(fs_rows)

        # ── Scatter: current corr vs spread ───────────────────────────────────
        scatter_fig = go.Figure()
        scatter_fig.add_trace(go.Scatter(
            x=fs_full_df["Current Corr"],
            y=fs_full_df["Spread (std)"],
            mode="markers",
            marker=dict(
                size=7,
                color=fs_full_df["Spread (std)"],
                colorscale="Viridis_r",
                showscale=True,
                colorbar=dict(title="Spread (std)"),
            ),
            text=[f"{r['Feature ID']}<br>{r['Description'][:80]}"
                  f"<br>spread={r['Spread (std)']:.3f}, range={r['Range']:.3f}"
                  f"<br>min={r['Min']}, max={r['Max']}"
                  for _, r in fs_full_df.iterrows()],
            hoverinfo="text",
            showlegend=False,
        ))
        scatter_fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1)
        scatter_fig.update_layout(
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=f"Correlation on {fs_current_task}",
            yaxis_title="Std of correlation across all tasks",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        gen_tab, spec_tab = st.tabs(["Generic Features (low spread)", "Task-Specific Features (high spread)"])
        fs_sidebar_match = (fs_dataset == dataset_name and fs_current_task == task_name)
        fs_format = {
            "Current Corr": "{:+.4f}", "Spread (std)": "{:.4f}",
            "Range": "{:.4f}", "Mean Corr": "{:+.4f}",
        }

        # ── Generic: low spread ───────────────────────────────────────────────
        with gen_tab:
            spread_vals = fs_full_df["Spread (std)"].values
            gen_max = st.slider(
                "Max spread", 0.0, float(max(spread_vals.max(), 0.01)),
                float(min(0.1, spread_vals.max())), 0.01,
                help="Lower spread = more consistent across tasks = more generic",
                key="fs_gen_max",
            )
            gen_df = fs_full_df[fs_full_df["Spread (std)"] <= gen_max].copy()
            gen_df["_abs_curr"] = gen_df["Current Corr"].abs()
            gen_df = gen_df.sort_values(
                ["Spread (std)", "_abs_curr"], ascending=[True, False]
            ).drop(columns="_abs_curr").reset_index(drop=True)

            st.markdown(f"**{len(gen_df)}** generic features (spread ≤ {gen_max:.3f})")
            if gen_df.empty:
                st.info("No features match.")
            else:
                gen_event = st.dataframe(
                    gen_df.style.format(fs_format),
                    use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row",
                    height=min(500, 35 * len(gen_df) + 40),
                    key="table_fs_gen",
                )
                if fs_sidebar_match:
                    sel = gen_event.selection.rows if gen_event.selection.rows else [0]
                    sel_fid = gen_df.iloc[sel[0]]["Feature ID"]
                    sel_feat = next((f for f in features if f["feature_id"] == sel_fid), None)
                    if sel_feat:
                        st.markdown("---")
                        render_feature_detail(sel_feat, sel_fid, data, checkpoints, n_ckpts,
                                              features, dataset_name, task_name,
                                              task_path=tasks[task_name][dataset_name], tab_key="fs_gen")
                else:
                    st.info("Switch sidebar dataset & task to match for the checkpoint-level detail view.")

        # ── Task-specific: high spread ────────────────────────────────────────
        with spec_tab:
            spread_vals = fs_full_df["Spread (std)"].values
            spec_min = st.slider(
                "Min spread", 0.0, float(max(spread_vals.max(), 0.01)),
                float(min(0.2, spread_vals.max() * 0.5)), 0.01,
                help="Higher spread = more variable across tasks = more specific somewhere",
                key="fs_spec_min",
            )
            spec_df = fs_full_df[fs_full_df["Spread (std)"] >= spec_min].copy()
            spec_df["_abs_curr"] = spec_df["Current Corr"].abs()
            spec_df = spec_df.sort_values(
                ["Spread (std)", "_abs_curr"], ascending=[False, False]
            ).drop(columns="_abs_curr").reset_index(drop=True)

            st.markdown(f"**{len(spec_df)}** task-specific features (spread ≥ {spec_min:.3f})")
            if spec_df.empty:
                st.info("No features match.")
            else:
                spec_event = st.dataframe(
                    spec_df.style.format(fs_format),
                    use_container_width=True, hide_index=True,
                    on_select="rerun", selection_mode="single-row",
                    height=min(500, 35 * len(spec_df) + 40),
                    key="table_fs_spec",
                )
                if fs_sidebar_match:
                    sel = spec_event.selection.rows if spec_event.selection.rows else [0]
                    sel_fid = spec_df.iloc[sel[0]]["Feature ID"]
                    sel_feat = next((f for f in features if f["feature_id"] == sel_fid), None)
                    if sel_feat:
                        st.markdown("---")
                        render_feature_detail(sel_feat, sel_fid, data, checkpoints, n_ckpts,
                                              features, dataset_name, task_name,
                                              task_path=tasks[task_name][dataset_name], tab_key="fs_spec")
                else:
                    st.info("Switch sidebar dataset & task to match for the checkpoint-level detail view.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab: Cross-Model Topics
# ══════════════════════════════════════════════════════════════════════════════

if has_topics:
    _topics_tab_idx = tab_names.index("Cross-Model Topics")
    with tabs[_topics_tab_idx]:
        task_topics = topic_data[task_name]

        pos_tab, neg_tab = st.tabs(["Positively Correlated", "Negatively Correlated"])

        for direction, dir_tab in [("positive", pos_tab), ("negative", neg_tab)]:
            with dir_tab:
                r = task_topics[direction]

                # ── Side-by-side top features ─────────────────────────────────
                tcol_amber, tcol_olmo = st.columns(2)

                with tcol_amber:
                    st.markdown("**Amber**")
                    amber_df = pd.DataFrame(r["amber_features"])
                    if not amber_df.empty:
                        show_cols = ["corr", "desc", "id"]
                        amber_show = amber_df[[c for c in show_cols if c in amber_df.columns]].copy()
                        amber_show.columns = ["Correlation", "Description", "Feature ID"]
                        amber_show["Correlation"] = amber_show["Correlation"].apply(lambda v: f"{'+' if v >= 0 else ''}{v:.3f}")
                        st.dataframe(amber_show, use_container_width=True, hide_index=True, height=350)

                with tcol_olmo:
                    st.markdown("**OLMo3**")
                    olmo_df = pd.DataFrame(r["olmo3_features"])
                    if not olmo_df.empty:
                        show_cols = ["corr", "desc", "id"]
                        olmo_show = olmo_df[[c for c in show_cols if c in olmo_df.columns]].copy()
                        olmo_show.columns = ["Correlation", "Description", "Feature ID"]
                        olmo_show["Correlation"] = olmo_show["Correlation"].apply(lambda v: f"{'+' if v >= 0 else ''}{v:.3f}")
                        st.dataframe(olmo_show, use_container_width=True, hide_index=True, height=350)

                # ── Combined topics with features and samples ─────────────────
                st.markdown("#### Combined Topics (across both models)")
                for t in r["combined_topics"]:
                    words = ", ".join(t["words"][:8])
                    topic_label = t.get("topic_name", f"Topic {t['topic_id']}")
                    topic_feats = t.get("features", [])
                    n_feats = len(topic_feats)
                    with st.expander(f"{topic_label}: {words} ({n_feats} features)"):
                        for feat in topic_feats:
                            model_tag = feat.get("model", "")
                            corr_val = feat.get("corr", 0)
                            st.markdown(f"**[{model_tag}]** `{corr_val:+.3f}` — {feat.get('desc', '')}")
                            feat_samples = feat.get("samples", [])
                            if feat_samples:
                                s_rows = []
                                for s in feat_samples[:5]:
                                    s_rows.append({
                                        "Before": s.get("before", ""),
                                        "Word": s.get("word", ""),
                                        "After": s.get("after", ""),
                                    })
                                st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True, height=min(200, 35 * len(s_rows) + 40))

                # ── Per-model topics ──────────────────────────────────────────
                mcol_a, mcol_o = st.columns(2)
                with mcol_a:
                    st.markdown("**Amber-only topics**")
                    for t in r["amber_topics"]:
                        words = ", ".join(t["words"][:6])
                        topic_label = t.get("topic_name", f"Topic {t['topic_id']}")
                        with st.expander(f"{topic_label}: {words}"):
                            for feat in t.get("features", []):
                                st.markdown(f"`{feat.get('corr', 0):+.3f}` — {feat.get('desc', '')}")
                with mcol_o:
                    st.markdown("**OLMo3-only topics**")
                    for t in r["olmo3_topics"]:
                        words = ", ".join(t["words"][:6])
                        topic_label = t.get("topic_name", f"Topic {t['topic_id']}")
                        with st.expander(f"{topic_label}: {words}"):
                            for feat in t.get("features", []):
                                st.markdown(f"`{feat.get('corr', 0):+.3f}` — {feat.get('desc', '')}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Cross-Model Matches
# ══════════════════════════════════════════════════════════════════════════════

if has_matches:
    match_tab_idx = tab_names.index("Cross-Model Matches")
    with tabs[match_tab_idx]:
        cfg = match_data.get("config", {})
        st.subheader("Cross-Model Feature Matches")
        st.caption(f"Combined TF-IDF + embedding similarity "
                   f"(model: {cfg.get('embedding_model', 'unknown')}, "
                   f"tfidf weight: {cfg.get('tfidf_weight', '?')}). "
                   f"Top {cfg.get('top_n', '?')} features per model by |{cfg.get('corr_key', '?')}|.")

        a2o = match_data["amber_to_olmo"]
        o2a = match_data["olmo_to_amber"]

        direction = st.radio("Compare", ["Amber → OLMo3", "OLMo3 → Amber"],
                             horizontal=True, key="match_dir")

        if direction == "Amber → OLMo3":
            match_list = a2o
            src_label, tgt_label = "amber", "olmo"
        else:
            match_list = o2a
            src_label, tgt_label = "olmo", "amber"

        # Filter to only show features whose best match has similarity >= 0.8
        match_list_filtered = [m for m in match_list
                               if m["matches"] and m["matches"][0]["similarity"] >= 0.6]
        match_list_sorted = sorted(match_list_filtered,
                                   key=lambda m: m["matches"][0]["similarity"],
                                   reverse=True)

        st.markdown(f"**{len(match_list_sorted)}** common features (best match similarity >= 0.8)")

        # Scrollable list of all features, each showing source on left and best match on right
        for entry in match_list_sorted:
            src_id = entry[f"{src_label}_id"]
            src_desc = entry[f"{src_label}_desc"]
            matches = entry.get("matches", [])
            best_sim = matches[0]["similarity"] if matches else 0

            with st.expander(f"`{src_id}` — {src_desc}  (best match: {best_sim:.3f})", expanded=False):
                col_src, col_tgt = st.columns(2)

                with col_src:
                    st.markdown(f"**{src_label.capitalize()} `{src_id}`**")
                    st.markdown(src_desc)

                    # Task correlations
                    src_corrs = entry.get(f"{src_label}_task_corrs", {})
                    if src_corrs:
                        corr_items = sorted(src_corrs.items(), key=lambda x: -abs(x[1]))
                        corr_df = pd.DataFrame(corr_items, columns=["Task", "Corr"])
                        corr_df["Corr"] = corr_df["Corr"].apply(lambda v: f"{'+' if v >= 0 else ''}{v:.4f}")
                        st.dataframe(corr_df, use_container_width=True, hide_index=True,
                                     height=min(200, 35 * len(corr_items) + 40))

                    # Samples
                    src_samples = entry.get(f"{src_label}_samples", [])
                    if src_samples:
                        s_rows = [{"Before": s.get("before", ""), "Word": s.get("word", ""), "After": s.get("after", "")}
                                  for s in src_samples]
                        st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True,
                                     height=min(180, 35 * len(s_rows) + 40))

                with col_tgt:
                    if not matches:
                        st.info("No matches found.")
                    else:
                        # Show all matches as sub-tabs
                        match_tabs = st.tabs([f"#{i+1} sim={m['similarity']:.3f}" for i, m in enumerate(matches)])
                        for mi, (mt, mt_tab) in enumerate(zip(matches, match_tabs)):
                            with mt_tab:
                                tid = mt[f"{tgt_label}_id"]
                                tdesc = mt[f"{tgt_label}_desc"]
                                e_sim = mt.get("embed_sim", mt["similarity"])
                                t_sim = mt.get("tfidf_sim", 0)
                                st.markdown(f"**{tgt_label.capitalize()} `{tid}`**")
                                st.markdown(tdesc)
                                st.caption(f"combined={mt['similarity']:.3f} | embed={e_sim:.3f} | tfidf={t_sim:.3f}")

                                tgt_corrs = mt.get(f"{tgt_label}_task_corrs", {})
                                if tgt_corrs:
                                    tc_items = sorted(tgt_corrs.items(), key=lambda x: -abs(x[1]))
                                    tc_df = pd.DataFrame(tc_items, columns=["Task", "Corr"])
                                    tc_df["Corr"] = tc_df["Corr"].apply(lambda v: f"{'+' if v >= 0 else ''}{v:.4f}")
                                    st.dataframe(tc_df, use_container_width=True, hide_index=True,
                                                 height=min(200, 35 * len(tc_items) + 40))

                                tgt_samples = mt.get(f"{tgt_label}_samples", [])
                                if tgt_samples:
                                    ts_rows = [{"Before": s.get("before", ""), "Word": s.get("word", ""), "After": s.get("after", "")}
                                               for s in tgt_samples]
                                    st.dataframe(pd.DataFrame(ts_rows), use_container_width=True, hide_index=True,
                                                 height=min(180, 35 * len(ts_rows) + 40))


# ══════════════════════════════════════════════════════════════════════════════
# Tab: Jump Groups
# ══════════════════════════════════════════════════════════════════════════════

if has_jumps:
    jump_tab_idx = tab_names.index("Jump Groups")
    with tabs[jump_tab_idx]:
        st.subheader("Features grouped by max-Δ checkpoint transition")
        st.caption(
            "For each feature: median prob across samples per checkpoint, then "
            "the largest single-checkpoint jump (max |Δ|). Same for each task's "
            "overall_performance. Bucketed by transition index."
        )

        jd_ckpts = jump_data["checkpoints"]
        jd_trans = jump_data["transitions"]
        jd_by_trans = jump_data["by_transition"]
        feat_jumps_all = jump_data["feature_jumps"]
        task_jumps_all = jump_data["task_jumps"]

        fid_to_desc = {str(f["feature_id"]): f.get("description", "")
                       for f in features}

        summary_rows = [{
            "Transition": label,
            "# Tasks": jd_by_trans[label]["n_tasks"],
            "# Features": jd_by_trans[label]["n_features"],
        } for label in jd_trans]
        sum_df = pd.DataFrame(summary_rows)
        st.markdown("**Co-occurrence summary**")
        st.dataframe(sum_df, use_container_width=True, hide_index=True,
                     height=min(400, 35 * len(summary_rows) + 40))

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name="# Tasks", x=jd_trans,
                                  y=[r["# Tasks"] for r in summary_rows],
                                  marker_color="#2196F3"))
        bar_fig.add_trace(go.Bar(name="# Features", x=jd_trans,
                                  y=[r["# Features"] for r in summary_rows],
                                  marker_color="#FF5722", yaxis="y2"))
        bar_fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            barmode="group",
            yaxis=dict(title="Tasks", color="#2196F3"),
            yaxis2=dict(title="Features", color="#FF5722",
                        overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # ── Feature efficiency curve ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### Feature efficiency curve")
        st.caption(
            "x = cumulative fraction of total task improvement by checkpoint t. "
            "y = cumulative fraction of features whose max-Δ jump has occurred by t. "
            "Above diagonal → features acquired faster than capabilities (representational "
            "pieces accumulate before they can be used). Below → capabilities outpace features."
        )

        n_ck_jump = len(jd_ckpts)
        # Cumulative feature fraction at each checkpoint (after t transitions).
        feat_counts_per_trans = np.array(
            [jd_by_trans[label]["n_features"] for label in jd_trans], dtype=float
        )
        total_feats_jumped = float(feat_counts_per_trans.sum())
        if total_feats_jumped > 0:
            cum_feat_frac = np.concatenate(
                [[0.0], np.cumsum(feat_counts_per_trans) / total_feats_jumped]
            )
        else:
            cum_feat_frac = np.zeros(n_ck_jump)

        # Per-task cumulative perf fraction. Only include tasks where total
        # change is positive; clip to [0,1] so non-monotone trajectories don't
        # produce negative values.
        per_task_xy = {}
        agg_perf_frac = []
        for tname, info in task_jumps_all.items():
            deltas = np.array(info.get("all_deltas", []), dtype=float)
            if deltas.size != n_ck_jump - 1:
                continue
            perf_rel = np.concatenate([[0.0], np.cumsum(deltas)])
            total_change = perf_rel[-1] - perf_rel[0]
            if total_change <= 1e-12:
                continue
            cum_perf_frac = np.clip((perf_rel - perf_rel[0]) / total_change, 0.0, 1.0)
            per_task_xy[tname] = cum_perf_frac
            agg_perf_frac.append(cum_perf_frac)

        # 24-color qualitative palette so 16+ tasks each get a distinct color.
        try:
            import plotly.colors as _pc
            palette = _pc.qualitative.Light24 + _pc.qualitative.Dark24
        except Exception:
            palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        eff_fig = go.Figure()
        eff_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="diagonal (x=y)",
            line=dict(dash="dash", color="#888", width=1),
            hoverinfo="skip",
        ))
        for i, (tname, x) in enumerate(per_task_xy.items()):
            color = palette[i % len(palette)]
            eff_fig.add_trace(go.Scatter(
                x=x, y=cum_feat_frac, mode="lines+markers",
                name=tname,
                line=dict(width=1.5, color=color),
                marker=dict(size=5, color=color),
                opacity=0.7,
                hovertemplate="ckpt %{text}: x=%{x:.3f}, y=%{y:.3f}"
                              "<extra>%{fullData.name}</extra>",
                text=jd_ckpts,
            ))
        if agg_perf_frac:
            agg_x = np.mean(np.stack(agg_perf_frac, axis=0), axis=0)
            eff_fig.add_trace(go.Scatter(
                x=agg_x, y=cum_feat_frac, mode="lines+markers",
                name="MEAN across tasks",
                line=dict(color="#000", width=3),
                marker=dict(size=8, color="#000"),
                hovertemplate="ckpt %{text}: x=%{x:.3f}, y=%{y:.3f}"
                              "<extra>mean</extra>",
                text=jd_ckpts,
            ))
        eff_fig.update_layout(
            height=500, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(title="Cumulative task perf fraction",
                       range=[-0.05, 1.05], scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Cumulative feature fraction jumped",
                       range=[-0.05, 1.05]),
            legend=dict(font=dict(size=9)),
            hovermode="closest",
        )
        st.plotly_chart(eff_fig, use_container_width=True)

        # Summary metric: signed area between mean curve and diagonal.
        if agg_perf_frac:
            order = np.argsort(agg_x)
            x_sorted = agg_x[order]
            y_sorted = cum_feat_frac[order]
            area_curve = float(np.trapz(y_sorted, x_sorted))
            area_above = area_curve - 0.5
            verdict = ("features ahead of capabilities" if area_above > 0
                       else "capabilities ahead of features")
            st.metric(
                "Mean curve area − 0.5 (signed)",
                f"{area_above:+.3f}",
                help=("Positive → mean curve sits above the diagonal "
                      "(features acquired earlier than perf gains). "
                      "Magnitude scales with how much."),
            )
            st.caption(f"Verdict: {verdict}.")

        st.markdown("---")
        sel_label = st.selectbox(
            "Select transition (jump checkpoint)",
            jd_trans,
            format_func=lambda l: (
                f"{l}   "
                f"[{jd_by_trans[l]['n_tasks']} tasks, "
                f"{jd_by_trans[l]['n_features']} features]"
            ),
            key="jump_trans_sel",
        )
        bucket = jd_by_trans[sel_label]
        sel_idx = bucket["transition_idx"]

        st.markdown(f"### Tasks jumping at {sel_label} ({bucket['n_tasks']})")
        if bucket["tasks"]:
            tdf = pd.DataFrame([{
                "Task": t["task"],
                "Δ": f"{t['delta']:+.4f}",
                "Before": f"{t['perf_before']:.4f}",
                "After": f"{t['perf_after']:.4f}",
            } for t in bucket["tasks"]])
            st.dataframe(tdf, use_container_width=True, hide_index=True,
                         height=min(300, 35 * len(tdf) + 40))
        else:
            st.info("No tasks jump at this transition.")

        st.markdown("---")
        st.markdown(f"### Features jumping at {sel_label} ({bucket['n_features']})")

        features_by_id_jump = {f["feature_id"]: f for f in features}
        bucket_feats = [f for f in bucket["features"]
                        if f["feature_id"] in features_by_id_jump]
        missing_n = bucket["n_features"] - len(bucket_feats)
        if missing_n:
            st.caption(f"({missing_n} features in this group not present in the "
                       f"current task's JSON — selecting a different task may surface them.)")

        n_show = bucket["n_features"]
        if n_show > 0:
            max_show = st.slider(
                "Max features to render", 1, max(1, min(100, n_show)),
                min(20, n_show), key="jump_max_render",
            )
        else:
            max_show = 0

        task_path_jump = tasks[task_name][dataset_name]

        # Reconstruct absolute perf curve for each task in this cluster.
        cluster_task_curves = []
        for t in bucket["tasks"]:
            info_full = task_jumps_all.get(t["task"], {})
            deltas = info_full.get("all_deltas", [])
            if deltas:
                cur = 0.0
                absolute = [cur]
                for d in deltas:
                    cur += d
                    absolute.append(cur)
                shift = t["perf_before"] - absolute[sel_idx]
                perf = [a + shift for a in absolute]
            else:
                perf = ([t["perf_before"]] * (sel_idx + 1)
                        + [t["perf_after"]] * (len(jd_ckpts) - sel_idx - 1))
            cluster_task_curves.append((t["task"], t["delta"], perf))

        all_task_data_jump = load_all_tasks_for_dataset(dataset_name)

        for f in bucket_feats[:max_show]:
            fid = f["feature_id"]
            sel_feat = features_by_id_jump[fid]
            desc = fid_to_desc.get(str(fid), "")
            header = (f"f{fid}  Δmed={f['delta']:+.4f}  "
                      f"({f['median_before']:.3f} → {f['median_after']:.3f})  "
                      f"— {desc[:80]}")
            with st.expander(header, expanded=False):
                st.markdown(f"**Feature {fid}: {desc}**")

                # ── Dual-axis: every cluster task's perf + this feature's median prob ──
                medians = sel_feat.get("median_probs", [])
                std_probs = sel_feat.get("std_probs", [0] * len(medians))
                plot_ckpts = jd_ckpts[:len(medians)]
                mp = np.array(medians)
                sp = np.array(std_probs[:len(medians)])

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for tname, tdelta, perf in cluster_task_curves:
                    fig.add_trace(
                        go.Scatter(x=jd_ckpts, y=perf, mode="lines+markers",
                                   name=f"{tname} ({tdelta:+.3f})",
                                   line=dict(width=1.5),
                                   hovertemplate="%{x}: %{y:.4f}<extra>%{fullData.name}</extra>"),
                        secondary_y=False,
                    )
                fig.add_trace(
                    go.Scatter(x=plot_ckpts, y=mp, name="Median Prob (feature)",
                               mode="lines+markers",
                               line=dict(color="#FF5722", width=3),
                               marker=dict(size=9),
                               error_y=dict(type="data", array=sp, arrayminus=sp,
                                            visible=True, color="rgba(255,87,34,0.3)",
                                            thickness=1.5, width=4)),
                    secondary_y=True,
                )
                fig.add_vrect(
                    x0=jd_ckpts[sel_idx], x1=jd_ckpts[sel_idx + 1],
                    fillcolor="yellow", opacity=0.15, line_width=0,
                )
                fig.update_layout(
                    height=420, margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="center", x=0.5, font=dict(size=9)),
                    hovermode="x unified",
                )
                fig.update_yaxes(title_text="Task Performance", secondary_y=False)
                fig.update_yaxes(title_text="Median Probability",
                                 secondary_y=True, color="#FF5722",
                                 range=([max(0, float(mp.min()) - float(sp.max()) * 1.2),
                                         min(1, float(mp.max()) + float(sp.max()) * 1.2)]
                                        if len(mp) else None))
                st.plotly_chart(fig, use_container_width=True,
                                key=f"jump_plot_{sel_idx}_{fid}")

                # ── Correlation stats (against currently-selected task) ─────
                stat_cols = st.columns(4)
                stat_cols[0].metric("Pearson", f"{sel_feat.get('pearson_corr', 0):.4f}")
                stat_cols[1].metric("Spearman", f"{sel_feat.get('spearman_corr', 0):.4f}")
                stat_cols[2].metric("Partial Pearson", f"{sel_feat.get('partial_pearson_corr', 0):.4f}")
                stat_cols[3].metric("Partial Spearman", f"{sel_feat.get('partial_spearman_corr', 0):.4f}")
                st.caption(f"(corrs computed against `{task_name}`)")

                # ── Samples table ────────────────────────────────────────────
                samples = sel_feat.get("samples", [])
                if samples:
                    st.markdown("**Samples**")
                    sort_by = st.radio(
                        "Sort by", ["cos_sim", "activation"],
                        horizontal=True, index=0,
                        key=f"jump_sort_{sel_idx}_{fid}",
                    )
                    sample_df = pd.DataFrame(samples).sort_values(
                        sort_by, ascending=False).reset_index(drop=True)
                    html_rows = []
                    for _, r in sample_df.iterrows():
                        before = html_mod.escape(str(r.get("before", "")))
                        word = html_mod.escape(str(r.get("word", "")))
                        after = html_mod.escape(str(r.get("after", "")))
                        cos = r.get("cos_sim", 0)
                        act = r.get("activation", 0)
                        html_rows.append(
                            f"<tr>"
                            f"<td style='text-align:right;color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{before}</td>"
                            f"<td style='font-weight:bold;padding:4px 6px;white-space:nowrap;'>{word}</td>"
                            f"<td style='color:#888;font-size:0.85em;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;'>{after}</td>"
                            f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{cos:.4f}</td>"
                            f"<td style='text-align:right;font-family:monospace;padding:4px 6px;'>{act:.4f}</td>"
                            f"</tr>"
                        )
                    table_html = (
                        "<div style='max-height:300px;overflow-y:auto;border:1px solid #ddd;border-radius:4px;'>"
                        "<table style='width:100%;border-collapse:collapse;'>"
                        "<thead style='position:sticky;top:0;background:#f0f0f0;z-index:1;'><tr>"
                        "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Before</th>"
                        "<th style='border-bottom:2px solid #ddd;padding:6px;'>Word</th>"
                        "<th style='border-bottom:2px solid #ddd;padding:6px;'>After</th>"
                        "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Cos</th>"
                        "<th style='text-align:right;border-bottom:2px solid #ddd;padding:6px;'>Act</th>"
                        "</tr></thead><tbody>" + "\n".join(html_rows) + "</tbody></table></div>"
                    )
                    st.markdown(table_html, unsafe_allow_html=True)

                # ── Cross-task corr summary for this feature ────────────────
                cross_rows = []
                for tname in sorted(all_task_data_jump.keys()):
                    tdata = all_task_data_jump[tname]
                    fm = next((ff for ff in tdata["features"]
                               if ff["feature_id"] == fid), None)
                    if fm:
                        cross_rows.append({
                            "Task": tname,
                            "In cluster": "✓" if tname in {tt[0] for tt in cluster_task_curves} else "",
                            "Partial Pearson": fm.get("partial_pearson_corr", 0),
                            "Pearson": fm.get("pearson_corr", 0),
                            "Spearman": fm.get("spearman_corr", 0),
                        })
                if cross_rows:
                    st.markdown("**Cross-task correlations**")
                    cdf = pd.DataFrame(cross_rows)
                    st.dataframe(
                        cdf.style.format({c: "{:+.4f}" for c in cdf.columns
                                          if c not in ("Task", "In cluster")}),
                        use_container_width=True, hide_index=True,
                        height=min(300, 35 * len(cdf) + 40),
                    )

