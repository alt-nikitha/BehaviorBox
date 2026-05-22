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

st.set_page_config(page_title="Feature Explorer — Jumps", layout="wide")

# ── Data loading ──────────────────────────────────────────────────────────────

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
JUMP_FILE = os.path.join(ANALYSIS_DIR, "jump_alignment.json")

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
def load_all_tasks_for_dataset(dataset_name):
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
def load_jump_data(_mtime):
    if os.path.exists(JUMP_FILE):
        with open(JUMP_FILE) as f:
            return json.load(f)
    return None


CONSENSUS_DIR = os.path.join(ANALYSIS_DIR, "feature_trajectory_consensus")


@st.cache_data
def load_consensus(task_name, _mtime):
    """Look for {CONSENSUS_DIR}/{task_name}.json. Returns dict or None."""
    path = os.path.join(CONSENSUS_DIR, f"{task_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

tasks = discover_tasks()
if not tasks:
    st.error(f"No precomputed_data_* directories found in {ANALYSIS_DIR}")
    st.stop()

st.sidebar.title("Feature Explorer — Jumps")
task_name = st.sidebar.selectbox("Task", list(tasks.keys()))
dataset_options = [d for d in tasks[task_name].keys()
                   if "delta" in d and "3000" in d and "auto" in d]
if not dataset_options:
    dataset_options = list(tasks[task_name].keys())
dataset_name = st.sidebar.selectbox("Dataset", dataset_options)

data = load_task_data(tasks[task_name][dataset_name])
features = data["features"]
checkpoints = data["checkpoints"]
n_ckpts = len(checkpoints)

_jump_mtime = os.path.getmtime(JUMP_FILE) if os.path.exists(JUMP_FILE) else 0
jump_data = load_jump_data(_jump_mtime)
if jump_data is None:
    st.error(f"jump_alignment.json not found at {JUMP_FILE}. "
             f"Run find_jump_alignment.py first.")
    st.stop()

# ── Trajectory consensus filter ───────────────────────────────────────────────
_cons_path = os.path.join(CONSENSUS_DIR, f"{task_name}.json")
_cons_mtime = os.path.getmtime(_cons_path) if os.path.exists(_cons_path) else 0
consensus_data = load_consensus(task_name, _cons_mtime)
consensus_map = (consensus_data or {}).get("features", {})

st.sidebar.markdown("---")
st.sidebar.markdown("**Trajectory consensus filter**")
if consensus_data is None:
    st.sidebar.caption(
        f"No consensus file at `{_cons_path}`. "
        f"Run compute_feature_trajectory_consensus.py to enable filtering."
    )
    consensus_threshold = 0.0
    require_consensus = False
    allowed_categories = None
else:
    require_consensus = st.sidebar.checkbox(
        "Only keep features where top tokens agree on trajectory",
        value=True,
    )
    consensus_threshold = st.sidebar.slider(
        "Min consensus fraction", 0.0, 1.0, 0.0, 0.05,
        help="Fraction of top-K samples sharing the dominant trajectory category.",
    )
    cats_present = sorted({v.get("dominant_category") for v in consensus_map.values()
                           if v.get("dominant_category")})
    allowed_categories = st.sidebar.multiselect(
        "Allowed dominant categories (empty = all)",
        cats_present,
        default=[],
    )
    st.sidebar.caption(
        f"Consensus computed with top_k={consensus_data.get('top_k')}, "
        f"flat_thresh={consensus_data.get('flat_thresh'):.3g}"
    )


def passes_consensus(fid):
    if not require_consensus:
        return True
    info = consensus_map.get(str(fid))
    if info is None:
        return False
    if info.get("consensus_frac", 0.0) < consensus_threshold:
        return False
    if allowed_categories and info.get("dominant_category") not in allowed_categories:
        return False
    return True

# ── Header ────────────────────────────────────────────────────────────────────

task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
st.markdown(f"### {task_name}")
st.caption(task_desc)

# ══════════════════════════════════════════════════════════════════════════════
# Jump Groups
# ══════════════════════════════════════════════════════════════════════════════

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

fid_to_desc = {str(f["feature_id"]): f.get("description", "") for f in features}

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

# ── Feature efficiency curve ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Feature efficiency curve")
st.caption(
    "x = cumulative fraction of total task improvement by checkpoint t. "
    "y = cumulative fraction of features whose max-Δ jump has occurred by t. "
    "Above diagonal → features acquired faster than capabilities. "
    "Below → capabilities outpace features."
)

n_ck_jump = len(jd_ckpts)
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
bucket_feats_all = [f for f in bucket["features"]
                    if f["feature_id"] in features_by_id_jump]
missing_n = bucket["n_features"] - len(bucket_feats_all)
if missing_n:
    st.caption(f"({missing_n} features in this group not present in the "
               f"current task's JSON — selecting a different task may surface them.)")

bucket_feats = [f for f in bucket_feats_all if passes_consensus(f["feature_id"])]
n_filtered = len(bucket_feats_all) - len(bucket_feats)
if require_consensus and consensus_data is not None:
    st.caption(
        f"Trajectory-consensus filter: {len(bucket_feats)} of "
        f"{len(bucket_feats_all)} kept "
        f"(min consensus {consensus_threshold:.2f}"
        + (f", categories: {allowed_categories}" if allowed_categories else "")
        + f"). {n_filtered} dropped."
    )

n_show = len(bucket_feats)

# ── Sort by correlation with a task ──────────────────────────────────────────
bucket_task_names = sorted({t["task"] for t in bucket["tasks"]})
sort_options = [
    "Default (Δ desc)",
    "Correlation with task (positive, desc)",
    "Correlation with task (negative, asc)",
    "Specificity to task (positive, desc)",
    "Specificity to task (negative, asc)",
]
sort_mode = st.radio(
    "Sort features by",
    sort_options,
    horizontal=True,
    key=f"jump_feat_sort_{sel_idx}",
)

corr_task = None
corr_metric = "pearson"
feat_corrs = {}
if sort_mode != "Default (Δ desc)":
    if not bucket_task_names:
        st.warning(
            "No tasks jump at this transition — pick a different transition "
            "to enable correlation sorting."
        )
    else:
        cs1, cs2 = st.columns([2, 1])
        with cs1:
            corr_task = st.selectbox(
                "Correlate with task (jumps at this transition)",
                bucket_task_names,
                key=f"jump_corr_task_{sel_idx}",
            )
        with cs2:
            corr_metric = st.selectbox(
                "Metric",
                ["pearson", "spearman", "kendall",
                 "partial_pearson", "partial_spearman"],
                help=("partial_* controls for checkpoint index "
                      "(removes monotonic time trend)."),
                key=f"jump_corr_metric_{sel_idx}",
            )

if corr_task is not None:
    task_info = task_jumps_all.get(corr_task, {})
    _deltas = task_info.get("all_deltas", [])
    if _deltas:
        _cur = 0.0
        _absolute = [_cur]
        for _d in _deltas:
            _cur += _d
            _absolute.append(_cur)
        task_perf_arr = np.array(_absolute, dtype=float)
    else:
        task_perf_arr = None

    def _pearson(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _partial_pearson(x, y, z):
        rxy = _pearson(x, y)
        rxz = _pearson(x, z)
        ryz = _pearson(y, z)
        denom = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
        if denom == 0:
            return 0.0
        return float((rxy - rxz * ryz) / denom)

    def _kendall(x, y):
        try:
            from scipy.stats import kendalltau
            tau, _ = kendalltau(x, y)
            return float(tau) if not np.isnan(tau) else 0.0
        except Exception:
            n = len(x)
            if n < 2:
                return 0.0
            conc = disc = 0
            for i in range(n):
                for j in range(i + 1, n):
                    s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
                    if s > 0:
                        conc += 1
                    elif s < 0:
                        disc += 1
            tot = conc + disc
            return (conc - disc) / tot if tot else 0.0

    def _compute_corr(fid):
        if task_perf_arr is None:
            return 0.0
        feat = features_by_id_jump.get(fid)
        if feat is None:
            return 0.0
        medians = feat.get("median_probs", [])
        n = min(len(medians), len(task_perf_arr))
        if n < 2:
            return 0.0
        x = np.array(medians[:n], dtype=float)
        y = task_perf_arr[:n]
        z = np.arange(n, dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        if corr_metric == "pearson":
            return _pearson(x, y)
        if corr_metric == "spearman":
            xr = pd.Series(x).rank().values
            yr = pd.Series(y).rank().values
            return _pearson(xr, yr)
        if corr_metric == "kendall":
            return _kendall(x, y)
        if corr_metric == "partial_pearson":
            return _partial_pearson(x, y, z)
        if corr_metric == "partial_spearman":
            xr = pd.Series(x).rank().values
            yr = pd.Series(y).rank().values
            zr = pd.Series(z).rank().values
            return _partial_pearson(xr, yr, zr)
        return 0.0

    is_specificity = sort_mode.startswith("Specificity")

    if is_specificity:
        all_task_perfs = {}
        for _tname in task_jumps_all.keys():
            _ti = task_jumps_all.get(_tname, {})
            _ds = _ti.get("all_deltas", [])
            if not _ds:
                continue
            _c = 0.0
            _ab = [_c]
            for _d in _ds:
                _c += _d
                _ab.append(_c)
            all_task_perfs[_tname] = np.array(_ab, dtype=float)

        def _safe(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if np.isnan(v) or np.isinf(v) else v

        def _corr_with(x, y_arr):
            n = min(len(x), len(y_arr))
            if n < 2:
                return 0.0
            xx = np.asarray(x[:n], dtype=float)
            yy = np.asarray(y_arr[:n], dtype=float)
            mask = np.isfinite(xx) & np.isfinite(yy)
            if mask.sum() < 2:
                return 0.0
            xx = xx[mask]
            yy = yy[mask]
            z = np.arange(len(xx), dtype=float)
            if np.std(xx) == 0 or np.std(yy) == 0:
                return 0.0
            if corr_metric == "pearson":
                return _safe(_pearson(xx, yy))
            if corr_metric == "spearman":
                return _safe(_pearson(pd.Series(xx).rank().values,
                                      pd.Series(yy).rank().values))
            if corr_metric == "kendall":
                return _safe(_kendall(xx, yy))
            if corr_metric == "partial_pearson":
                return _safe(_partial_pearson(xx, yy, z))
            if corr_metric == "partial_spearman":
                return _safe(_partial_pearson(pd.Series(xx).rank().values,
                                              pd.Series(yy).rank().values,
                                              pd.Series(z).rank().values))
            return 0.0

        spec_sign = +1 if "(positive" in sort_mode else -1

        def _compute_specificity(fid):
            feat = features_by_id_jump.get(fid)
            if feat is None or corr_task not in all_task_perfs:
                return 0.0
            x = np.array(feat.get("median_probs", []), dtype=float)
            if len(x) < 2:
                return 0.0
            corrs = [_corr_with(x, yp) for yp in all_task_perfs.values()]
            target = _corr_with(x, all_task_perfs[corr_task])
            if not np.isfinite(target):
                return 0.0
            if spec_sign > 0:
                if target <= 0:
                    return 0.0
                denom = sum(c for c in corrs if np.isfinite(c) and c > 0)
                return float(target / denom) if denom > 0 else 0.0
            else:
                if target >= 0:
                    return 0.0
                denom = sum(-c for c in corrs if np.isfinite(c) and c < 0)
                return float(target / denom) if denom > 0 else 0.0

        pairs = [(f, _compute_specificity(f["feature_id"])) for f in bucket_feats]
    else:
        pairs = [(f, _compute_corr(f["feature_id"])) for f in bucket_feats]

    reverse = "(positive" in sort_mode
    pairs.sort(key=lambda p: (p[1] if not np.isnan(p[1]) else 0.0),
               reverse=reverse)
    bucket_feats = [p[0] for p in pairs]
    feat_corrs = {p[0]["feature_id"]: p[1] for p in pairs}

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

PAGE_SIZE = 20
total = len(bucket_feats)
n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
pc1, pc2 = st.columns([1, 3])
with pc1:
    page = st.number_input(
        f"Page (1–{n_pages})", min_value=1, max_value=n_pages,
        value=1, step=1, key=f"jump_page_{sel_idx}",
    )
with pc2:
    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    st.caption(f"Showing features {start + 1}–{end} of {total}")

page_feats = bucket_feats[start:end]

for f in page_feats:
    fid = f["feature_id"]
    sel_feat = features_by_id_jump[fid]
    desc = fid_to_desc.get(str(fid), "")
    cinfo = consensus_map.get(str(fid)) or {}
    cons_tag = ""
    if cinfo:
        cons_tag = (f"  [{cinfo.get('dominant_category', '?')} "
                    f"{cinfo.get('consensus_frac', 0.0):.0%} of "
                    f"{cinfo.get('n_used', 0)}]")
    corr_tag = ""
    if fid in feat_corrs:
        _label = "spec" if sort_mode.startswith("Specificity") else "corr"
        corr_tag = f"  {_label}({corr_metric[:4]},{corr_task})={feat_corrs[fid]:+.3f}"
    header = (f"f{fid}  Δmed={f['delta']:+.4f}  "
              f"({f['median_before']:.3f} → {f['median_after']:.3f}){cons_tag}{corr_tag}  "
              f"— {desc[:80]}")
    with st.expander(header, expanded=False):
        st.markdown(f"**Feature {fid}: {desc}**")

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

        cross_rows = []
        for tname in sorted(all_task_data_jump.keys()):
            tdata = all_task_data_jump[tname]
            fm = next((ff for ff in tdata["features"]
                       if ff["feature_id"] == fid), None)
            if fm:
                cross_rows.append({
                    "Task": tname,
                    "In cluster": "✓" if tname in {tt[0] for tt in cluster_task_curves} else "",
                    "Δ med (jump)": next(
                        (jf["delta"] for jf in feat_jumps_all.get(str(fid), {}).get("per_task", [])
                         if jf.get("task") == tname),
                        None,
                    ) if isinstance(feat_jumps_all.get(str(fid)), dict) else None,
                })
        if cross_rows:
            st.markdown("**Cross-task presence**")
            cdf = pd.DataFrame(cross_rows)
            cdf = cdf.dropna(axis=1, how="all")
            st.dataframe(
                cdf, use_container_width=True, hide_index=True,
                height=min(300, 35 * len(cdf) + 40),
            )
