"""Render task_shape_groups.py's two views as a static, self-contained HTML file.

Tab 1: task clusters (hierarchical clustering by curve shape) with two columns of
       matched features (positive / negative) per cluster, each expandable to its
       samples sorted by cos_sim.
Tab 2: features assigned to each task centroid (from the precomputed pickle), with
       positive / negative columns and samples sorted by z-area distance to the
       chosen task centroid.

Usage:
    python task_shape_groups_to_html.py \
        --dataset OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm \
        --metric area \
        --out task_shape_groups.html
"""

import argparse
import glob
import html
import json
import os
import pickle
from collections import defaultdict

import numpy as np

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
    diff = _shared_diff(curve_a, curve_b)
    if diff is None:
        return float("nan")
    return float(np.mean(diff ** 2))


def area_distance(curve_a, curve_b):
    diff = _shared_diff(curve_a, curve_b)
    if diff is None:
        return float("nan")
    return float(np.trapezoid(np.abs(diff)))


DISTANCE_FNS = {"area": area_distance, "mse": mse_distance}


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
            d = distance_fn(task_curves[names[i]], task_curves[names[j]])
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


def compute_cluster_feature_scores(clusters, tsg_data, distance_fn):
    """Per cluster: feature_id -> {pos_scores, neg_scores, stds, desc, src_task_with_samples}."""
    out = {}
    for ci, members in clusters.items():
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
                mp_flip = (mp_norm[0], -mp_norm[1])
                d_neg = distance_fn(op_norm, mp_flip)
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
                # prefer a copy that has samples
                if not feat_scores[fid]["samples"] and feat.get("samples"):
                    feat_scores[fid]["samples"] = feat["samples"]
                    feat_scores[fid]["src_task"] = tname
        out[ci] = feat_scores
    return out


def select_top_features(feat_scores, members, direction, top_n, max_dist=None, max_std=None):
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


# ── HTML rendering ─────────────────────────────────────────────────────────


def esc(s):
    return html.escape(str(s) if s is not None else "")


def render_samples_table(samples, dist_key=None, dist_label=None):
    """Render a samples sub-table. If dist_key given (e.g. cluster task name), pull
    distance from `_task_dists[dist_key]`."""
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


def render_feature_block(row, samples_html, idx_key):
    """Collapsible feature block — uses <details> for native dropdown behaviour."""
    head = (
        f"<summary>"
        f"<span class='fid'>#{esc(row['fid'])}</span>"
        f"<span class='metric'>avg={row['avg']:.4f}</span>"
        f"<span class='metric'>max={row['max']:.4f}</span>"
        f"<span class='metric'>n={row['n_tasks']}</span>"
        f"<span class='desc'>{esc(row['desc'])}</span>"
        f"</summary>"
    )
    return f"<details class='feature' id='{esc(idx_key)}'>{head}{samples_html}</details>"


def render_tab1(clusters, task_curves, cluster_feat_scores, metric_name, top_n=20, max_samples=50):
    parts = []
    for ci in sorted(clusters.keys()):
        members = clusters[ci]
        scores = cluster_feat_scores.get(ci, {})
        pos_rows = select_top_features(scores, members, "positive", top_n)
        neg_rows = select_top_features(scores, members, "negative", top_n)

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
            f"<div class='cluster-meta'><h3>Cluster {ci} <small>({len(members)} tasks)</small></h3>"
            f"<ul class='task-list'>{''.join(member_rows)}</ul></div>"
        )

        def col(rows, label, idx_prefix):
            blocks = []
            for r in rows:
                samples = trim_samples(r["samples"], "cos_sim", reverse=True, top_k=max_samples)
                samples_html = render_samples_table(samples)
                blocks.append(render_feature_block(r, samples_html, f"{idx_prefix}-{r['fid']}"))
            if not blocks:
                blocks.append("<div class='empty'>No features below threshold.</div>")
            return (
                f"<div class='col'>"
                f"<h4>{label} <small>({len(rows)})</small></h4>"
                f"{''.join(blocks)}</div>"
            )

        pos_col = col(pos_rows, "Positive features (curve-aligned)", f"c{ci}-pos")
        neg_col = col(neg_rows, "Negative features (anti-aligned)", f"c{ci}-neg")
        parts.append(
            f"<section class='cluster-row'>"
            f"{cluster_card}<div class='feat-cols'>{pos_col}{neg_col}</div></section>"
        )
    return "<div class='tab1'>" + "".join(parts) + "</div>"


def render_tab2(pre, metric_name, top_n_per_task=50, max_samples=50):
    task_names = sorted(pre["task_curves"].keys())
    feature_meta = pre["feature_meta"]
    feature_samples = pre["feature_samples"]

    parts = []
    for t in task_names:
        md = TASK_METADATA.get(t, {})
        desc = TASK_DESCRIPTIONS.get(t, "")
        pos = sorted(pre["assignments_pos"].get(t, []), key=lambda x: x[1])[:top_n_per_task]
        neg = sorted(pre["assignments_neg"].get(t, []), key=lambda x: x[1])[:top_n_per_task]

        def feat_rows(assignments, idx_prefix):
            blocks = []
            for fid, d in assignments:
                meta = feature_meta.get(fid, {})
                samples = feature_samples.get(fid, []) or []
                # sort samples by _task_dists[t] ascending (closer to task curve first)
                def k(s):
                    v = (s.get("_task_dists") or {}).get(t)
                    return (v is None, v if v is not None else float("inf"))
                ordered = sorted(samples, key=k)[:max_samples]
                samples_html = render_samples_table(ordered, dist_key=t,
                                                    dist_label=f"z-{pre['metric']} to {t}")
                head = (
                    f"<summary>"
                    f"<span class='fid'>#{esc(fid)}</span>"
                    f"<span class='metric'>d={d:.4f}</span>"
                    f"<span class='metric'>med_std={meta.get('median_std', float('nan')):.4f}</span>"
                    f"<span class='metric'>n_tasks={meta.get('n_tasks', '?')}</span>"
                    f"<span class='desc'>{esc(meta.get('desc',''))}</span>"
                    f"</summary>"
                )
                blocks.append(
                    f"<details class='feature' id='{esc(idx_prefix)}-{esc(fid)}'>{head}{samples_html}</details>"
                )
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
            f"<div class='col'><h4>Positive features <small>({len(pos)})</small></h4>{feat_rows(pos, f't2-{t}-pos')}</div>"
            f"<div class='col'><h4>Negative (flipped) features <small>({len(neg)})</small></h4>{feat_rows(neg, f't2-{t}-neg')}</div>"
            f"</div></section>"
        )
    return "<div class='tab2'>" + "".join(parts) + "</div>"


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
.empty { color: #999; font-size: 12px; padding: 8px; font-style: italic; }
.samples-wrap { max-height: 360px; overflow-y: auto; margin: 8px;
                border: 1px solid #e0e0e0; border-radius: 3px; background: #fff; }
table.samples { width: 100%; border-collapse: collapse; font-size: 11.5px; }
table.samples thead th { position: sticky; top: 0; background: #f0f0f0;
                         padding: 5px 6px; border-bottom: 1px solid #ddd;
                         text-align: left; }
table.samples td { padding: 3px 6px; border-bottom: 1px solid #f0f0f0;
                   vertical-align: top; }
table.samples td.before { text-align: right; color: #888; max-width: 240px;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
table.samples td.word { font-weight: 700; white-space: nowrap; }
table.samples td.after { color: #888; max-width: 240px; overflow: hidden;
                         text-overflow: ellipsis; white-space: nowrap; }
table.samples td.num { font-family: monospace; text-align: right;
                       white-space: nowrap; }
"""


JS = """
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('btn-' + name).classList.add('active');
}
"""


def render_html(tab1_html, tab2_html, dataset_name, metric_name, summary):
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Task Shape Groups — {esc(dataset_name)}</title>
<style>{CSS}</style></head><body>
<header>
  <h1>Task Shape Groups</h1>
  <div class='meta'>Dataset: <code>{esc(dataset_name)}</code> · metric: <code>{esc(metric_name)}</code> · {summary}</div>
  <div class='tabs'>
    <button id='btn-tab1' class='active' onclick="showTab('tab1')">Task clusters → matched features</button>
    <button id='btn-tab2' onclick="showTab('tab2')">Features assigned to each task</button>
  </div>
</header>
<div id='panel-tab1' class='tab-panel active'>{tab1_html}</div>
<div id='panel-tab2' class='tab-panel'>{tab2_html}</div>
<script>{JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm")
    ap.add_argument("--metric", default="area", choices=list(DISTANCE_FNS))
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR, "task_shape_groups.html"))
    ap.add_argument("--top-n", type=int, default=20, help="Top features per cluster column (tab 1).")
    ap.add_argument("--top-n-tab2", type=int, default=50, help="Top features per task column (tab 2).")
    ap.add_argument("--max-samples", type=int, default=50,
                    help="Max samples shown per feature.")
    args = ap.parse_args()

    metric_name = "Z-norm Area between" if args.metric == "area" else "Z-norm MSE"

    print(f"loading task json for dataset={args.dataset!r}…")
    tsg_data = load_all_tasks_for_dataset(args.dataset)
    print(f"  {len(tsg_data)} tasks")

    print("clustering tasks…")
    distance_fn = DISTANCE_FNS[args.metric]
    clusters, task_curves, task_ckpts, auto_k = compute_task_clusters(tsg_data, distance_fn)
    print(f"  k = {auto_k} clusters")

    print("scoring features against each cluster (positive + negative)…")
    cluster_feat_scores = compute_cluster_feature_scores(clusters, tsg_data, distance_fn)

    print("rendering tab 1…")
    tab1 = render_tab1(clusters, task_curves, cluster_feat_scores, metric_name,
                       top_n=args.top_n, max_samples=args.max_samples)

    pkl = precompute_path(args.dataset, metric_name)
    if not os.path.exists(pkl):
        raise SystemExit(f"missing precomputed pickle: {pkl}")
    print(f"loading {pkl}…")
    with open(pkl, "rb") as f:
        pre = pickle.load(f)

    print("rendering tab 2…")
    tab2 = render_tab2(pre, metric_name,
                       top_n_per_task=args.top_n_tab2,
                       max_samples=args.max_samples)

    summary = (
        f"{len(tsg_data)} tasks → {auto_k} clusters · "
        f"{len(pre['feature_curves'])} features assigned to "
        f"{len(pre['task_curves'])} task centroids"
    )
    out_html = render_html(tab1, tab2, args.dataset, metric_name, summary)

    with open(args.out, "w") as f:
        f.write(out_html)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
