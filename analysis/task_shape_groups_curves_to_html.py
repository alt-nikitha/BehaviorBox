"""Per-cluster and per-task feature listing with per-feature curve plots.

Merges the feature/sample tables of task_shape_groups_to_html.py with the
per-feature dual-axis curve plot style of group_features_by_task_olmo.py.

Tab 1: task clusters (hierarchical by curve shape). For each cluster, two
       columns of features (positive / negative). Each feature row has, when
       expanded, a dual-axis plot (cluster-mean task perf + cluster-mean
       feature median) and its top samples by cos_sim.
Tab 2: per-task centroid view. For each task, two columns of assigned
       features (from the precomputed pickle). Each feature row has, when
       expanded, a dual-axis plot (task perf + feature median for that task)
       and its top samples by z-distance to the task curve.
"""

import argparse
import glob
import html
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy import stats

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
    "winogrande":           {"type": "reasoning",  "domain": "coreference",  "format": "MCQ"},
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

METRIC_SLUGS = {"Z-norm Area between": "area", "Z-norm MSE": "mse",
                "1 - Spearman": "spearman",
                "1 - Spearman combo (level+diff)": "spearman_combo"}
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
    diff = np.array([da[k] for k in shared]) - np.array([db[k] for k in shared])
    return shared, diff


def _ckpt_x(shared, ckpts):
    """Numeric x-coordinates from ckpts, or None to signal unit-spacing."""
    if not ckpts:
        return None
    try:
        x = np.array([float(ckpts[i]) for i in shared], dtype=float)
    except (TypeError, ValueError, IndexError):
        return None
    if not np.all(np.diff(x) > 0):
        return None
    return x


def area_distance(a, b, ckpts=None):
    res = _shared_diff(a, b)
    if res is None:
        return float("nan")
    shared, diff = res
    x = _ckpt_x(shared, ckpts)
    if x is not None:
        return float(np.trapezoid(np.abs(diff), x=x))
    return float(np.trapezoid(np.abs(diff)))


def mse_distance(a, b, ckpts=None):
    res = _shared_diff(a, b)
    return float("nan") if res is None else float(np.mean(res[1] ** 2))


def _spearman_rho(a, b):
    """Spearman rank correlation over shared checkpoints, or None if degenerate."""
    ia, va = a
    ib, vb = b
    shared = sorted(set(ia) & set(ib))
    if len(shared) < 3:
        return None
    da, db = dict(zip(ia, va)), dict(zip(ib, vb))
    xa = np.array([da[k] for k in shared])
    xb = np.array([db[k] for k in shared])
    if np.std(xa) < 1e-12 or np.std(xb) < 1e-12:
        return None
    rho, _ = stats.spearmanr(xa, xb)
    return None if rho != rho else float(rho)


def spearman_distance(a, b, ckpts=None):
    """Distance = 1 - Spearman rank correlation over shared checkpoints (lower =
    better match), matching the distance semantics used elsewhere. Invariant to
    checkpoint spacing, so ckpts is unused."""
    rho = _spearman_rho(a, b)
    return float("nan") if rho is None else float(1.0 - rho)


# Blend weight for spearman_combo (see precompute_task_shape_groups.py).
# Set from --combo-weight in main(). Smaller leans on jump-alignment (diff).
COMBO_WEIGHT = 0.25


def _difference_curve(curve):
    idxs, vals = curve
    if len(vals) < 2:
        return None
    return idxs[1:], np.diff(vals)


def spearman_combo_distance(a, b, ckpts=None):
    """Distance = 1 - [w*rho_level + (1-w)*rho_diff], combining Spearman of the
    raw curves (ordering agrees) with Spearman of consecutive-checkpoint diffs
    (jumps align). w = COMBO_WEIGHT. Mirrors the precompute metric."""
    rho_level = _spearman_rho(a, b)
    da, db = _difference_curve(a), _difference_curve(b)
    rho_diff = None if (da is None or db is None) else _spearman_rho(da, db)
    if rho_level is None or rho_diff is None:
        return float("nan")
    w = COMBO_WEIGHT
    return float(1.0 - (w * rho_level + (1.0 - w) * rho_diff))


DISTANCE_FNS = {"area": area_distance, "mse": mse_distance,
                "spearman": spearman_distance,
                "spearman_combo": spearman_combo_distance}


# ── Data loading ──────────────────────────────────────────────────────────


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


def compute_task_clusters(tsg_data, distance_fn, linkage_method="average"):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    task_curves, task_ckpts = {}, None
    for tname, td in tsg_data.items():
        norm = normalize_curve(td.get("overall_performance", []))
        if norm is not None:
            task_curves[tname] = norm
            if task_ckpts is None:
                task_ckpts = td.get("checkpoints", [])
    names = sorted(task_curves.keys())
    n = len(names)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(task_curves[names[i]], task_curves[names[j]],
                            ckpts=task_ckpts)
            if np.isnan(d):
                d = 0.0
            dist[i, j] = dist[j, i] = d
    condensed = squareform((dist + dist.T) / 2, checks=False)
    Z = linkage(condensed, method=linkage_method)
    heights = Z[:, 2]
    max_k = min(8, n - 1)
    auto_k, best_gap = 2, -np.inf
    start = 3 if max_k >= 3 else 2
    for k in range(start, max_k + 1):
        gap = float(heights[n - k] - heights[n - k - 1])
        if gap > best_gap:
            best_gap, auto_k = gap, k
    labels = fcluster(Z, t=auto_k, criterion="maxclust")
    clusters = defaultdict(list)
    for tname, lbl in zip(names, labels):
        clusters[int(lbl)].append(tname)
    return clusters, task_curves, task_ckpts, auto_k


def compute_cluster_feature_scores(clusters, tsg_data, distance_fn, ckpts=None):
    out = {}
    for ci, members in clusters.items():
        feat_scores = {}
        for tname in members:
            td = tsg_data.get(tname, {})
            op_norm = normalize_curve(td.get("overall_performance", []))
            if op_norm is None:
                continue
            t_ckpts = ckpts or td.get("checkpoints") or None
            for feat in td.get("features", []):
                mp_norm = normalize_curve(feat.get("median_probs", []))
                if mp_norm is None:
                    continue
                d_pos = distance_fn(op_norm, mp_norm, ckpts=t_ckpts)
                d_neg = distance_fn(op_norm, (mp_norm[0], -mp_norm[1]),
                                    ckpts=t_ckpts)
                if np.isnan(d_pos) and np.isnan(d_neg):
                    continue
                fid = feat["feature_id"]
                if fid not in feat_scores:
                    feat_scores[fid] = {
                        "pos_scores": [],
                        "neg_scores": [],
                        "stds": [],
                        "desc": feat.get("description", ""),
                        "samples": feat.get("samples", []) or [],
                        "src_task": tname,
                    }
                if not np.isnan(d_pos):
                    feat_scores[fid]["pos_scores"].append(d_pos)
                if not np.isnan(d_neg):
                    feat_scores[fid]["neg_scores"].append(d_neg)
                sps = [s for s in (feat.get("std_probs") or []) if s is not None]
                if sps:
                    feat_scores[fid]["stds"].append(float(np.median(sps)))
                if not feat_scores[fid]["samples"] and feat.get("samples"):
                    feat_scores[fid]["samples"] = feat["samples"]
                    feat_scores[fid]["src_task"] = tname
        out[ci] = feat_scores
    return out


def select_top_features(feat_scores, members, direction, top_n,
                        max_dist=None, max_std=None):
    key = "pos_scores" if direction == "positive" else "neg_scores"
    min_presence = max(1, (len(members) + 1) // 2)
    rows = []
    for fid, info in feat_scores.items():
        scores = info[key]
        if len(scores) < min_presence:
            continue
        avg = float(np.mean(scores))
        if max_dist is not None and avg > max_dist:
            continue
        med_std = float(np.median(info["stds"])) if info["stds"] else float("nan")
        if max_std is not None and not np.isnan(med_std) and med_std > max_std:
            continue
        rows.append({
            "fid": fid,
            "avg": avg,
            "max": float(np.max(scores)),
            "median_std": med_std,
            "n_tasks": len(scores),
            "desc": info["desc"],
            "samples": info["samples"],
            "src_task": info["src_task"],
        })
    rows.sort(key=lambda r: r["avg"])
    return rows[:top_n]


def trim_samples(samples, key, reverse=True, top_k=None):
    def get(s):
        return s.get(key) if s.get(key) is not None else (-float("inf") if reverse else float("inf"))
    ordered = sorted(samples, key=get, reverse=reverse)
    if top_k:
        ordered = ordered[:top_k]
    return ordered


# ── Raw curves (for olmo-style plots) ─────────────────────────────────────


def build_cluster_raw_curves(clusters, tsg_data):
    """Per cluster: cluster-averaged raw task perf, plus per-feature
    cluster-averaged raw median_probs.

    Returns {ci: {"ckpts": [...], "task_perf": [...], "feat_med": {fid: [...]}}}.
    """
    out = {}
    for ci, members in clusters.items():
        ckpts = None
        sums, counts = defaultdict(float), defaultdict(int)
        feat_sums = defaultdict(lambda: defaultdict(float))
        feat_counts = defaultdict(lambda: defaultdict(int))
        for tname in members:
            td = tsg_data.get(tname, {})
            if ckpts is None and td.get("checkpoints"):
                ckpts = td["checkpoints"]
            for i, p in enumerate(td.get("overall_performance", []) or []):
                if p is not None and np.isfinite(p):
                    sums[i] += float(p)
                    counts[i] += 1
            for feat in td.get("features", []):
                fid = feat["feature_id"]
                for i, m in enumerate(feat.get("median_probs", []) or []):
                    if m is not None and np.isfinite(m):
                        feat_sums[fid][i] += float(m)
                        feat_counts[fid][i] += 1
        if not ckpts:
            continue
        n = len(ckpts)
        task_perf = [(sums[i] / counts[i] if counts[i] else None) for i in range(n)]
        feat_med = {}
        for fid, sm in feat_sums.items():
            cm = feat_counts[fid]
            feat_med[fid] = [(sm[i] / cm[i] if cm[i] else None) for i in range(n)]
        out[ci] = {"ckpts": ckpts, "task_perf": task_perf, "feat_med": feat_med}
    return out


def build_task_raw_curves(tsg_data):
    """Per task: raw task perf and per-feature raw median_probs.

    Returns {tname: {"ckpts": [...], "task_perf": [...], "feat_med": {fid: [...]}}}.
    """
    out = {}
    for tname, td in tsg_data.items():
        ckpts = td.get("checkpoints", [])
        if not ckpts:
            continue
        task_perf = [
            (float(x) if x is not None and np.isfinite(x) else None)
            for x in (td.get("overall_performance", []) or [])
        ]
        feat_med = {}
        for feat in td.get("features", []):
            fid = feat["feature_id"]
            feat_med[fid] = [
                (float(x) if x is not None and np.isfinite(x) else None)
                for x in (feat.get("median_probs", []) or [])
            ]
        out[tname] = {"ckpts": ckpts, "task_perf": task_perf, "feat_med": feat_med}
    return out


# ── HTML rendering ───────────────────────────────────────────────────────


def esc(s):
    return html.escape(str(s) if s is not None else "")


def make_plot_spec(ckpts, feat_med, task_perf, fid_label, task_label,
                   feat_color="#1976d2", task_color="#444"):
    """Dual y-axis plot: task perf (y) + feature median (y2)."""
    traces = []
    if task_perf is not None:
        traces.append({
            "x": list(ckpts), "y": list(task_perf), "type": "scatter",
            "mode": "lines+markers", "name": task_label,
            "line": {"color": task_color, "width": 1.5},
            "marker": {"size": 4}, "yaxis": "y",
        })
    if feat_med is not None:
        traces.append({
            "x": list(ckpts), "y": list(feat_med), "type": "scatter",
            "mode": "lines+markers", "name": fid_label,
            "line": {"color": feat_color, "width": 2.5},
            "marker": {"size": 6}, "yaxis": "y2",
        })
    layout = {
        "height": 240,
        "margin": {"l": 45, "r": 55, "t": 10, "b": 60},
        "xaxis": {"tickangle": -45, "tickfont": {"size": 9}},
        "yaxis": {"title": "task perf", "titlefont": {"size": 10}},
        "yaxis2": {"title": "feat med", "titlefont": {"size": 10, "color": feat_color},
                   "tickfont": {"color": feat_color},
                   "overlaying": "y", "side": "right"},
        "legend": {"font": {"size": 9}, "orientation": "h", "y": -0.35},
        "hovermode": "x unified",
    }
    return {"data": traces, "layout": layout}


def render_samples_table(samples, dist_key=None, dist_label=None):
    rows = []
    for s in samples:
        before = esc(s.get("before", ""))
        word = esc(s.get("word", ""))
        after = esc(s.get("after", ""))
        cos = s.get("cos_sim", 0) or 0
        act = s.get("activation", 0) or 0
        cells = (
            f"<td class='before'>{before}</td>"
            f"<td class='word'>{word}</td>"
            f"<td class='after'>{after}</td>"
        )
        if dist_key is not None:
            d = (s.get("_task_dists") or {}).get(dist_key)
            d_cell = f"{d:.4f}" if d is not None else "—"
            cells += f"<td class='num'>{d_cell}</td>"
        cells += f"<td class='num'>{cos:.4f}</td><td class='num'>{act:.4f}</td>"
        rows.append(f"<tr>{cells}</tr>")
    headers = "<th>Before</th><th>Word</th><th>After</th>"
    if dist_label:
        headers += f"<th>{esc(dist_label)}</th>"
    headers += "<th>Cos&nbsp;Sim</th><th>Activation</th>"
    return (
        "<div class='samples-wrap'><table class='samples'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_feature_block(row, plot_id, samples_html, summary_metrics):
    """row: dict with fid, desc; summary_metrics: pre-formatted spans."""
    head = (
        f"<summary>"
        f"<span class='fid'>#{esc(row['fid'])}</span>"
        f"{summary_metrics}"
        f"<span class='desc'>{esc(row['desc'])}</span>"
        f"</summary>"
    )
    body = (
        f"<div class='feat-body'>"
        f"<div class='feat-plot' id='{esc(plot_id)}'></div>"
        f"<div class='feat-samples'>{samples_html}</div>"
        f"</div>"
    )
    return f"<details class='feature'>{head}{body}</details>"


def render_tab1(clusters, cluster_feat_scores, cluster_raw, top_n,
                max_samples, plot_specs):
    parts = []
    plot_idx = [0]

    def make_pid():
        plot_idx[0] += 1
        return f"plot_t1_{plot_idx[0]}"

    for ci in sorted(clusters.keys()):
        members = clusters[ci]
        scores = cluster_feat_scores.get(ci, {})
        pos_rows = select_top_features(scores, members, "positive", top_n)
        neg_rows = select_top_features(scores, members, "negative", top_n)

        raw = cluster_raw.get(ci, {})
        ckpts = raw.get("ckpts", [])
        task_perf = raw.get("task_perf")
        feat_med_by_fid = raw.get("feat_med", {})
        task_label = f"cluster {ci} mean task perf"

        member_rows = []
        for m in members:
            md = TASK_METADATA.get(m, {})
            member_rows.append(
                f"<li><b>{esc(m)}</b> "
                f"<span class='tag'>{esc(md.get('type','?'))}</span>"
                f"<span class='tag'>{esc(md.get('format','?'))}</span>"
                f"<span class='tag'>{esc(md.get('domain','?'))}</span></li>"
            )
        cluster_card = (
            f"<div class='cluster-meta'>"
            f"<h3>Cluster {ci} <small>({len(members)} tasks)</small></h3>"
            f"<ul class='task-list'>{''.join(member_rows)}</ul></div>"
        )

        def col(rows, label, color):
            blocks = []
            for r in rows:
                samples = trim_samples(r["samples"], "cos_sim",
                                       reverse=True, top_k=max_samples)
                samples_html = render_samples_table(samples)
                pid = make_pid()
                feat_med = feat_med_by_fid.get(r["fid"])
                plot_specs[pid] = make_plot_spec(
                    ckpts, feat_med, task_perf,
                    f"f{r['fid']}", task_label, feat_color=color)
                metrics = (
                    f"<span class='metric'>avg={r['avg']:.4f}</span>"
                    f"<span class='metric'>max={r['max']:.4f}</span>"
                    f"<span class='metric'>n={r['n_tasks']}</span>"
                )
                blocks.append(render_feature_block(r, pid, samples_html, metrics))
            if not blocks:
                blocks.append("<div class='empty'>No features below threshold.</div>")
            return (
                f"<div class='col'>"
                f"<h4>{label} <small>({len(rows)})</small></h4>"
                f"{''.join(blocks)}</div>"
            )

        pos_col = col(pos_rows, "Positive features (curve-aligned)", "#1976d2")
        neg_col = col(neg_rows, "Negative features (anti-aligned)", "#c62828")
        parts.append(
            f"<section class='cluster-row'>"
            f"{cluster_card}<div class='feat-cols'>{pos_col}{neg_col}</div></section>"
        )
    return "<div class='tab1'>" + "".join(parts) + "</div>"


def render_tab2(pre, task_raw, top_n_per_task, max_samples, plot_specs):
    task_names = sorted(pre["task_curves"].keys())
    feature_meta = pre["feature_meta"]
    feature_samples = pre["feature_samples"]

    plot_idx = [0]

    def make_pid():
        plot_idx[0] += 1
        return f"plot_t2_{plot_idx[0]}"

    parts = []
    for t in task_names:
        md = TASK_METADATA.get(t, {})
        desc = TASK_DESCRIPTIONS.get(t, "")
        pos = sorted(pre["assignments_pos"].get(t, []), key=lambda x: x[1])[:top_n_per_task]
        neg = sorted(pre["assignments_neg"].get(t, []), key=lambda x: x[1])[:top_n_per_task]

        raw = task_raw.get(t, {})
        ckpts = raw.get("ckpts", [])
        task_perf = raw.get("task_perf")
        feat_med_by_fid = raw.get("feat_med", {})

        def feat_rows(assignments, color):
            blocks = []
            for fid, d in assignments:
                meta = feature_meta.get(fid, {})
                samples = feature_samples.get(fid, []) or []

                def k(s):
                    v = (s.get("_task_dists") or {}).get(t)
                    return (v is None, v if v is not None else float("inf"))

                ordered = sorted(samples, key=k)[:max_samples]
                samples_html = render_samples_table(
                    ordered, dist_key=t,
                    dist_label=f"z-{pre['metric']} to {t}")
                pid = make_pid()
                feat_med = feat_med_by_fid.get(fid)
                plot_specs[pid] = make_plot_spec(
                    ckpts, feat_med, task_perf,
                    f"f{fid}", f"{t} perf", feat_color=color)
                med_std = meta.get("median_std", float("nan"))
                med_std_str = f"{med_std:.4f}" if med_std == med_std else "nan"
                metrics = (
                    f"<span class='metric'>d={d:.4f}</span>"
                    f"<span class='metric'>med_std={med_std_str}</span>"
                    f"<span class='metric'>n_tasks={meta.get('n_tasks','?')}</span>"
                )
                row = {"fid": fid, "desc": meta.get("desc", "")}
                blocks.append(render_feature_block(row, pid, samples_html, metrics))
            if not blocks:
                blocks.append("<div class='empty'>No features assigned.</div>")
            return "".join(blocks)

        header = (
            f"<div class='cluster-meta'>"
            f"<h3>{esc(t)} "
            f"<span class='tag'>{esc(md.get('type','?'))}</span>"
            f"<span class='tag'>{esc(md.get('format','?'))}</span>"
            f"<span class='tag'>{esc(md.get('domain','?'))}</span></h3>"
            f"<p class='task-desc'>{esc(desc)}</p>"
            f"<p class='counts'>positive: {len(pre['assignments_pos'].get(t, []))} · "
            f"negative: {len(pre['assignments_neg'].get(t, []))} "
            f"(showing top {top_n_per_task} each)</p>"
            f"</div>"
        )
        parts.append(
            f"<section class='cluster-row'>{header}"
            f"<div class='feat-cols'>"
            f"<div class='col'><h4>Positive features "
            f"<small>({len(pos)})</small></h4>{feat_rows(pos, '#1976d2')}</div>"
            f"<div class='col'><h4>Negative (flipped) features "
            f"<small>({len(neg)})</small></h4>{feat_rows(neg, '#c62828')}</div>"
            f"</div></section>"
        )
    return "<div class='tab2'>" + "".join(parts) + "</div>"


def render_tab3(pre, task_raw, top_n_per_task, max_samples, plot_specs):
    """Per task, rank features by specificity = median(d_others) - d_task.
    Positive (raw curve) and negative (flipped curve) columns are shown
    separately."""
    spec_pos = pre.get("specificity_pos", {})
    spec_neg = pre.get("specificity_neg", {})
    feature_meta = pre["feature_meta"]
    feature_samples = pre["feature_samples"]
    task_names = sorted(pre["task_curves"].keys())

    plot_idx = [0]

    def make_pid():
        plot_idx[0] += 1
        return f"plot_t3_{plot_idx[0]}"

    parts = [
        "<div class='cluster-row'><div class='cluster-meta'>"
        "<p class='task-desc'>Per task, features ranked by "
        "<code>specificity = median(d to other tasks) − d to this task</code> "
        "(median, not mean, so a single similar task can't tank the score). "
        "Higher = the feature's training curve fits this task notably better "
        "than the rest, implicating that pretraining content for this task.</p>"
        "</div></div>"
    ]

    for t in task_names:
        md = TASK_METADATA.get(t, {})
        desc = TASK_DESCRIPTIONS.get(t, "")
        pos_rows = spec_pos.get(t, [])[:top_n_per_task]
        neg_rows = spec_neg.get(t, [])[:top_n_per_task]

        raw = task_raw.get(t, {})
        ckpts = raw.get("ckpts", [])
        task_perf = raw.get("task_perf")
        feat_med_by_fid = raw.get("feat_med", {})

        def feat_rows(rows, color):
            blocks = []
            for entry in rows:
                fid, specificity, d_task, med_others = entry
                meta = feature_meta.get(fid, {})
                samples = feature_samples.get(fid, []) or []

                def k(s):
                    v = (s.get("_task_dists") or {}).get(t)
                    return (v is None, v if v is not None else float("inf"))

                ordered = sorted(samples, key=k)[:max_samples]
                samples_html = render_samples_table(
                    ordered, dist_key=t,
                    dist_label=f"z-{pre['metric']} to {t}")
                pid = make_pid()
                feat_med = feat_med_by_fid.get(fid)
                plot_specs[pid] = make_plot_spec(
                    ckpts, feat_med, task_perf,
                    f"f{fid}", f"{t} perf", feat_color=color)
                med_std = meta.get("median_std", float("nan"))
                med_std_str = f"{med_std:.4f}" if med_std == med_std else "nan"
                metrics = (
                    f"<span class='metric'>spec={specificity:+.4f}</span>"
                    f"<span class='metric'>d_task={d_task:.4f}</span>"
                    f"<span class='metric'>med_others={med_others:.4f}</span>"
                    f"<span class='metric'>med_std={med_std_str}</span>"
                    f"<span class='metric'>n_tasks={meta.get('n_tasks','?')}</span>"
                )
                row = {"fid": fid, "desc": meta.get("desc", "")}
                blocks.append(render_feature_block(row, pid, samples_html, metrics))
            if not blocks:
                blocks.append("<div class='empty'>No features ranked.</div>")
            return "".join(blocks)

        header = (
            f"<div class='cluster-meta'>"
            f"<h3>{esc(t)} "
            f"<span class='tag'>{esc(md.get('type','?'))}</span>"
            f"<span class='tag'>{esc(md.get('format','?'))}</span>"
            f"<span class='tag'>{esc(md.get('domain','?'))}</span></h3>"
            f"<p class='task-desc'>{esc(desc)}</p>"
            f"<p class='counts'>positive ranked: {len(spec_pos.get(t, []))} · "
            f"negative ranked: {len(spec_neg.get(t, []))} "
            f"(showing top {top_n_per_task} each)</p>"
            f"</div>"
        )
        parts.append(
            f"<section class='cluster-row'>{header}"
            f"<div class='feat-cols'>"
            f"<div class='col'><h4>Top-specific positive features "
            f"<small>({len(pos_rows)})</small></h4>{feat_rows(pos_rows, '#1976d2')}</div>"
            f"<div class='col'><h4>Top-specific negative (flipped) features "
            f"<small>({len(neg_rows)})</small></h4>{feat_rows(neg_rows, '#c62828')}</div>"
            f"</div></section>"
        )
    return "<div class='tab3'>" + "".join(parts) + "</div>"


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 0; color: #222; background: #fafafa; }
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
.cluster-row { background: #fff; border: 1px solid #ddd; border-radius: 6px;
               padding: 16px; margin-bottom: 20px; }
.cluster-meta h3 { margin: 0 0 6px; font-size: 15px; }
.cluster-meta small { color: #888; font-weight: normal; }
.task-list { margin: 0 0 12px; padding-left: 18px; }
.task-list li { font-size: 13px; margin: 2px 0; }
.tag { display: inline-block; padding: 1px 6px; margin-left: 4px; font-size: 11px;
       background: #eef; border-radius: 3px; color: #339; }
.task-desc { color: #666; font-size: 12px; margin: 4px 0 8px; }
.counts { color: #888; font-size: 11px; margin: 0 0 8px; }
.feat-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
             margin-top: 8px; }
.col h4 { margin: 0 0 8px; font-size: 13px; color: #333;
          padding-bottom: 6px; border-bottom: 1px solid #eee; }
.col h4 small { color: #888; font-weight: normal; }
.feature { background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px;
           margin-bottom: 4px; padding: 0; }
.feature[open] { background: #fff; border-color: #b8c8e0; }
.feature summary { padding: 6px 10px; cursor: pointer; font-size: 12px;
                   display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.feature summary::-webkit-details-marker { color: #888; }
.feature summary .fid { font-family: monospace; font-weight: 600; color: #335; }
.feature summary .metric { font-family: monospace; font-size: 11px; color: #666; }
.feature summary .desc { color: #555; font-size: 11.5px; flex: 1 1 100%;
                         margin-top: 2px; }
.feat-body { display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
             padding: 8px; align-items: start; }
.feat-plot { width: 100%; height: 240px; background: #fff;
             border: 1px solid #e0e0e0; border-radius: 3px; }
.feat-samples { min-width: 0; }
.empty { color: #999; font-size: 12px; padding: 8px; font-style: italic; }
.samples-wrap { max-height: 280px; overflow-y: auto;
                border: 1px solid #e0e0e0; border-radius: 3px; background: #fff; }
table.samples { width: 100%; border-collapse: collapse; font-size: 11.5px; }
table.samples thead th { position: sticky; top: 0; background: #f0f0f0;
                         padding: 5px 6px; border-bottom: 1px solid #ddd;
                         text-align: left; }
table.samples td { padding: 3px 6px; border-bottom: 1px solid #f0f0f0;
                   vertical-align: top; }
table.samples td.before { text-align: right; color: #888; max-width: 220px;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
table.samples td.word { font-weight: 700; white-space: nowrap; }
table.samples td.after { color: #888; max-width: 220px; overflow: hidden;
                         text-overflow: ellipsis; white-space: nowrap; }
table.samples td.num { font-family: monospace; text-align: right;
                       white-space: nowrap; }
@media (max-width: 1100px) {
  .feat-body { grid-template-columns: 1fr; }
}
"""


JS_TEMPLATE = """
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('btn-' + name).classList.add('active');
  window.dispatchEvent(new Event('resize'));
}
document.addEventListener('toggle', function(ev){
  if (!(ev.target instanceof HTMLDetailsElement) || !ev.target.open) return;
  ev.target.querySelectorAll('.feat-plot').forEach(function(div){
    if (PLOTS[div.id] && !div.dataset.plotted) {
      Plotly.newPlot(div.id, PLOTS[div.id].data, PLOTS[div.id].layout,
                     {responsive: true, displaylogo: false});
      div.dataset.plotted = '1';
    }
  });
}, true);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm-sampled_trajectory_centroids")
    ap.add_argument("--metric", default="area", choices=list(DISTANCE_FNS))
    ap.add_argument("--combo-weight", type=float, default=0.25,
                    help="spearman_combo blend weight w (level vs diff); must "
                         "match the value used to build the pickle.")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Top features per cluster column (tab 1).")
    ap.add_argument("--top-n-tab2", type=int, default=30,
                    help="Top features per task column (tab 2).")
    ap.add_argument("--max-samples", type=int, default=50,
                    help="Max samples shown per feature.")
    args = ap.parse_args()

    global COMBO_WEIGHT
    COMBO_WEIGHT = args.combo_weight

    metric_name = {"area": "Z-norm Area between", "mse": "Z-norm MSE",
                   "spearman": "1 - Spearman",
                   "spearman_combo": "1 - Spearman combo (level+diff)"}[args.metric]
    distance_fn = DISTANCE_FNS[args.metric]

    print(f"loading task json for dataset={args.dataset!r}…")
    tsg_data = load_all_tasks_for_dataset(args.dataset)
    print(f"  {len(tsg_data)} tasks")

    print("clustering…")
    clusters, task_curves, task_ckpts, auto_k = compute_task_clusters(
        tsg_data, distance_fn)
    print(f"  k = {auto_k}")

    print("scoring features per cluster…")
    cluster_feat_scores = compute_cluster_feature_scores(
        clusters, tsg_data, distance_fn, ckpts=task_ckpts)

    print("building raw cluster curves…")
    cluster_raw = build_cluster_raw_curves(clusters, tsg_data)

    plot_specs = {}
    print("rendering tab 1…")
    tab1 = render_tab1(clusters, cluster_feat_scores, cluster_raw,
                       args.top_n, args.max_samples, plot_specs)

    pkl = precompute_path(args.dataset, metric_name)
    if not os.path.exists(pkl):
        raise SystemExit(f"missing pickle: {pkl}")
    print(f"loading {pkl}…")
    with open(pkl, "rb") as f:
        pre = pickle.load(f)

    print("building raw task curves…")
    task_raw = build_task_raw_curves(tsg_data)

    print("rendering tab 2…")
    tab2 = render_tab2(pre, task_raw, args.top_n_tab2,
                       args.max_samples, plot_specs)

    print("rendering tab 3…")
    tab3 = render_tab3(pre, task_raw, args.top_n_tab2,
                       args.max_samples, plot_specs)

    summary = (
        f"{len(tsg_data)} tasks · k={auto_k} clusters · "
        f"{len(pre['feature_curves'])} features · "
        f"tab1 top-{args.top_n} per side, tab2/3 top-{args.top_n_tab2} per side"
    )

    plots_json = json.dumps(plot_specs)
    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>Task Shape Groups — Curves + Features — {esc(args.dataset)}</title>
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>{CSS}</style></head><body>
<header>
  <h1>Task Shape Groups — Curves + Features</h1>
  <div class='meta'>Dataset: <code>{esc(args.dataset)}</code> · metric: <code>{esc(metric_name)}</code> · {summary}</div>
  <div class='tabs'>
    <button id='btn-tab1' class='active' onclick="showTab('tab1')">Task clusters → matched features</button>
    <button id='btn-tab2' onclick="showTab('tab2')">Features assigned to each task</button>
    <button id='btn-tab3' onclick="showTab('tab3')">Task-specific features (contrastive)</button>
  </div>
</header>
<div id='panel-tab1' class='tab-panel active'>{tab1}</div>
<div id='panel-tab2' class='tab-panel'>{tab2}</div>
<div id='panel-tab3' class='tab-panel'>{tab3}</div>
<script>const PLOTS = {plots_json};{JS_TEMPLATE}</script>
</body></html>"""

    with open(args.dataset+"_viz.html", "w") as f:
        f.write(html_doc)
    size_mb = os.path.getsize(args.dataset+"_viz.html") / (1024 * 1024)
    print(f"wrote {args.dataset+"_viz.html"} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
