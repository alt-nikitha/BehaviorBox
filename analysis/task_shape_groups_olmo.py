"""Per-cluster feature listing for a single model (OLMO), grouped by task shape.

Tasks are clustered by the shape of their performance trajectory over training
(z-normalized curves; distance = pointwise MSE or integrated area between curves).
For each cluster:
  - Lists every OLMO feature whose median_probs curve is, on average, close to
    the cluster's task performance curves (avg distance below threshold), and
    whose within-feature sample spread is small enough.
  - For each feature, annotates which OTHER clusters it also matches
    (cross-cluster chips).
"""

import argparse
import glob
import html as html_mod
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sample_curves import sample_distance_to

ROOT = Path("/home/nsrikant/BehaviorBoxNew/analysis")

VALID_TASKS = [
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
]

DEFAULT_MODEL_ID = "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm"

# Cluster colors (cycled if more clusters than colors).
CLUSTER_COLORS = ["#1976d2", "#e64a19", "#388e3c", "#7b1fa2", "#f9a825",
                  "#00838f", "#c2185b", "#5d4037"]


# ── Data loading ────────────────────────────────────────────────────────────

def load_native_data(model_id: str):
    """Returns (ckpts, task_perf, feat_meta) where feat_meta[fid] holds
    description, samples, median_probs, std_probs.

    Feature curves (median_probs, std_probs, samples) are task-independent —
    the same SAE feature has the same trajectory regardless of which task
    file it's read from. So we deduplicate across tasks.
    """
    ckpts = None
    task_perf = {}
    feat_meta = {}
    sample_pool = defaultdict(list)

    for d in sorted(glob.glob(str(ROOT / "precomputed_data_*"))):
        tname = Path(d).name.replace("precomputed_data_", "")
        if tname not in VALID_TASKS:
            continue
        fpath = Path(d) / f"{model_id}.json"
        if not fpath.exists():
            continue
        data = json.load(open(fpath))
        if ckpts is None:
            ckpts = data.get("checkpoints", [])
        perf = data.get("overall_performance") or []
        if perf and len(perf) >= 2:
            task_perf[tname] = [float(x) if x is not None and np.isfinite(x) else None
                                for x in perf]
        for feat in data.get("features", []) or []:
            fid = str(feat.get("feature_id"))
            desc = feat.get("description", "")
            entry = feat_meta.setdefault(fid, {"description": "", "median_probs": None, "std_probs": None})
            if desc and not entry.get("description"):
                entry["description"] = desc
            med = feat.get("median_probs") or feat.get("avg_probs")
            if med is not None and entry.get("median_probs") is None:
                entry["median_probs"] = [float(x) if x is not None and np.isfinite(x) else None
                                         for x in med]
            std = feat.get("std_probs")
            if std is not None and entry.get("std_probs") is None:
                entry["std_probs"] = [float(x) if x is not None and np.isfinite(x) else None
                                      for x in std]
            for s in feat.get("samples", []) or []:
                sample_pool[fid].append(s)

    for fid, samples in sample_pool.items():
        seen, unique = set(), []
        for s in sorted(samples, key=lambda r: -float(r.get("activation", 0) or 0)):
            key = (str(s.get("before", "")), str(s.get("word", "")), str(s.get("after", "")))
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
            if len(unique) >= 50:
                break
        feat_meta.setdefault(fid, {"description": ""})["samples"] = unique

    for fid in feat_meta:
        feat_meta[fid].setdefault("samples", [])
    return ckpts or [], task_perf, feat_meta


# ── Z-norm curve utilities (mirror task_shape_groups.py) ────────────────────

def z_normalize(values):
    """Return (valid_idxs, z_values) or None if degenerate."""
    valid = [(i, p) for i, p in enumerate(values or []) if p is not None and np.isfinite(p)]
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


METRIC_FNS = {"mse": mse_distance, "area": area_distance}


# ── Task clustering ─────────────────────────────────────────────────────────

def cluster_tasks(task_perf, distance_fn, linkage_method, k_override=None,
                  max_k_cap=8):
    """Returns (cluster_map: ci -> [tasks], dist_mat, task_order, chosen_k)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    curves = {}
    for t, p in task_perf.items():
        nz = z_normalize(p)
        if nz is not None:
            curves[t] = nz
    if len(curves) < 2:
        raise SystemExit("Need at least 2 tasks with valid performance curves.")

    order = sorted(curves)
    n = len(order)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(curves[order[i]], curves[order[j]])
            if np.isnan(d):
                d = 0.0
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    sym = (dist_mat + dist_mat.T) / 2
    np.fill_diagonal(sym, 0.0)
    condensed = squareform(sym, checks=False)
    Z = linkage(condensed, method=linkage_method)

    max_k = min(max_k_cap, n - 1)
    if k_override is not None:
        chosen_k = int(max(2, min(max_k, k_override)))
    else:
        # Dendrogram-gap auto-pick. Start at k=3: the k=2→1 merge is almost always
        # the biggest absolute jump in any tree (it's the trivial first split),
        # so it dominates without telling you anything interesting. k>=3 finds
        # the real structural gap. Falls back to k=2 only if data is tiny.
        heights = Z[:, 2]
        chosen_k, best_gap = 2, -np.inf
        start = 3 if max_k >= 3 else 2
        for k in range(start, max_k + 1):
            gap = float(heights[n - k] - heights[n - k - 1])
            if gap > best_gap:
                best_gap = gap
                chosen_k = k

    labels = fcluster(Z, t=chosen_k, criterion="maxclust")
    cluster_map = defaultdict(list)
    for tname, lbl in zip(order, labels):
        cluster_map[int(lbl)].append(tname)
    return dict(cluster_map), dist_mat, order, chosen_k, curves


# ── Feature scoring against clusters ────────────────────────────────────────

def typical_std(std_probs):
    """Median of std_probs — robust to a few outlier checkpoints."""
    arr = [s for s in (std_probs or []) if s is not None and np.isfinite(s)]
    if not arr:
        return None
    return float(np.median(arr))


def score_features_per_cluster(cluster_map, task_perf, feat_meta, distance_fn):
    """For each (cluster, fid): distance from the feature's single z-normalized
    curve to each task curve in the cluster, plus the within-feature spread.

    Returns {ci: {fid: {"avg_dist": float, "max_dist": float, "n_tasks": int,
                        "med_std": float, "per_task": {tname: dist}}}}.
    """
    task_curves = {}
    for t, p in task_perf.items():
        nz = z_normalize(p)
        if nz is not None:
            task_curves[t] = nz

    out = {}
    for ci, members in cluster_map.items():
        per_feat = {}
        for fid, meta in feat_meta.items():
            med = meta.get("median_probs")
            if med is None:
                continue
            fz = z_normalize(med)
            if fz is None:
                continue
            dists = []
            for tname in members:
                if tname not in task_curves:
                    continue
                d = distance_fn(task_curves[tname], fz)
                if np.isnan(d):
                    continue
                dists.append((tname, d))
            if not dists:
                continue
            dvals = [d for _, d in dists]
            per_feat[fid] = {
                "avg_dist": float(np.mean(dvals)),
                "max_dist": float(np.max(dvals)),
                "n_tasks": len(dvals),
                "med_std": typical_std(meta.get("std_probs")),
                "per_task": dict(dists),
            }
        out[ci] = per_feat
    return out


# ── HTML rendering ──────────────────────────────────────────────────────────

def truncate(s, n=140):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def render_samples(samples, target_z_curve=None, sample_metric="area",
                   metric_label="dist"):
    """If target_z_curve is given, compute each sample's z-norm distance to
    the target curve and sort ascending (closest first). Samples whose curve
    can't be resolved go to the bottom."""
    if not samples:
        return "<div style='color:#888;font-style:italic;padding:4px;'>no samples</div>"

    enriched = []
    for s in samples:
        d = None
        if target_z_curve is not None:
            d = sample_distance_to(target_z_curve, s.get("word_id"),
                                   mode=sample_metric)
        enriched.append((s, d))
    if target_z_curve is not None:
        enriched.sort(key=lambda sd: (sd[1] is None, sd[1] if sd[1] is not None else float("inf")))

    rows = []
    for s, d in enriched:
        before = html_mod.escape(str(s.get("before", "")))
        word = html_mod.escape(str(s.get("word", "")))
        after = html_mod.escape(str(s.get("after", "")))
        cos = s.get("cos_sim", 0) or 0
        act = s.get("activation", 0) or 0
        d_cell = (f"<td style='text-align:right;font-family:monospace;padding:2px 6px;'>"
                  f"{(f'{d:.3f}' if d is not None else '—')}</td>")
        rows.append(
            "<tr>"
            f"<td style='text-align:right;color:#888;font-size:0.85em;max-width:220px;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:2px 6px;'>{before}</td>"
            f"<td style='font-weight:bold;padding:2px 6px;white-space:nowrap;'>{word}</td>"
            f"<td style='color:#888;font-size:0.85em;max-width:220px;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;padding:2px 6px;'>{after}</td>"
            f"{d_cell}"
            f"<td style='text-align:right;font-family:monospace;padding:2px 6px;'>{cos:.3f}</td>"
            f"<td style='text-align:right;font-family:monospace;padding:2px 6px;'>{act:.3f}</td>"
            "</tr>"
        )
    return ("<div style='max-height:260px;overflow-y:auto;border:1px solid #ddd;margin-top:4px;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.78em;background:#fafafa;'>"
            "<thead style='position:sticky;top:0;background:#f0f0f0;'><tr>"
            "<th style='text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'>before</th>"
            "<th style='padding:2px 6px;border-bottom:1px solid #ccc;'>word</th>"
            "<th style='padding:2px 6px;border-bottom:1px solid #ccc;'>after</th>"
            f"<th style='text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'>{metric_label}</th>"
            "<th style='text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'>cos</th>"
            "<th style='text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'>act</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def make_plot_spec(ckpts, task_z_curves, feat_z_curve, feat_label, feat_color,
                   task_colors):
    """Overlay z-normalized task curves and the feature's z-normalized curve.
    Single shared y-axis (z-score)."""
    traces = []
    for tname, (idxs, vals) in task_z_curves.items():
        x = [ckpts[i] if i < len(ckpts) else str(i) for i in idxs]
        traces.append({
            "x": x, "y": list(vals), "type": "scatter", "mode": "lines+markers",
            "name": f"task: {tname}",
            "line": {"color": task_colors.get(tname, "#888"), "width": 1.5},
            "marker": {"size": 4}, "opacity": 0.7,
        })
    if feat_z_curve is not None:
        idxs, vals = feat_z_curve
        x = [ckpts[i] if i < len(ckpts) else str(i) for i in idxs]
        traces.append({
            "x": x, "y": list(vals), "type": "scatter", "mode": "lines+markers",
            "name": feat_label,
            "line": {"color": feat_color, "width": 3, "dash": "dash"},
            "marker": {"size": 7, "color": feat_color},
        })
    layout = {
        "height": 260,
        "margin": {"l": 50, "r": 20, "t": 10, "b": 60},
        "xaxis": {"tickangle": -45, "tickfont": {"size": 9}},
        "yaxis": {"title": "z-score", "titlefont": {"size": 10}},
        "legend": {"font": {"size": 8}, "orientation": "h", "y": -0.4},
        "hovermode": "x unified",
    }
    return {"data": traces, "layout": layout}


def other_clusters_sorted(fid, cluster_scores, exclude_ci):
    """All other clusters where this feature has a valid avg distance, sorted
    by distance ascending (closest first)."""
    rows = []
    for ci, fmap in cluster_scores.items():
        if ci == exclude_ci:
            continue
        info = fmap.get(fid)
        if not info:
            continue
        rows.append((ci, info["avg_dist"]))
    rows.sort(key=lambda kv: kv[1])
    return rows


def render_cluster_block(ci, members, ci_color, fmap,
                         feat_meta, task_perf, ckpts, task_colors,
                         cluster_scores,
                         top_k, require_desc, metric_label,
                         feat_metric_name,
                         plot_specs, plot_idx_ref):
    # Mean of cluster task z-curves restricted to shared checkpoints — used as
    # the target curve for per-sample sorting.
    member_z = []
    shared_idxs = None
    for tname in members:
        tz = z_normalize(task_perf.get(tname, []))
        if tz is None:
            continue
        idxs, vals = tz
        if shared_idxs is None:
            shared_idxs = set(idxs)
        else:
            shared_idxs = shared_idxs & set(idxs)
        member_z.append((idxs, vals))
    cluster_target = None
    if member_z and shared_idxs:
        ordered = sorted(shared_idxs)
        stacked = []
        for idxs, vals in member_z:
            idx_to_v = dict(zip(idxs, vals))
            stacked.append([idx_to_v[i] for i in ordered])
        cluster_target = (ordered, np.mean(stacked, axis=0))

    def render_other_cluster_chips(fid):
        rows = other_clusters_sorted(fid, cluster_scores, ci)
        if not rows:
            return ("<span style='color:#aaa;font-style:italic;font-size:0.78em;'>"
                    "(only this cluster)</span>")
        chips = []
        for other_ci, d in rows:
            chips.append(
                f"<span class='taskchip'>cluster {other_ci}: "
                f"<span class='pos'>{d:.3f}</span></span>"
            )
        return " ".join(chips)

    def render_one(fid, info):
        meta = feat_meta.get(fid) or {}
        desc = html_mod.escape(truncate(meta.get("description", "")))
        avg_d = info["avg_dist"]
        max_d = info["max_dist"]
        med_std = info["med_std"]
        spread_str = (f" <span class='meta'>med std={med_std:.3f}</span>"
                      if med_std is not None else "")
        # Build task z-curves for this cluster
        task_z = {}
        for tname in members:
            tz = z_normalize(task_perf.get(tname, []))
            if tz is not None:
                task_z[tname] = tz
        # Feature curve is task-independent — just z-normalize the one median_probs.
        feat_z = z_normalize(meta.get("median_probs"))

        pid = f"plot_olmo_{plot_idx_ref[0]}"
        plot_idx_ref[0] += 1
        plot_specs.append((pid, make_plot_spec(
            ckpts, task_z, feat_z, f"OLMO f{fid}", "#222", task_colors)))

        chips = render_other_cluster_chips(fid)
        per_t = info.get("per_task", {})
        per_task_chips = " ".join(
            f"<span class='taskchip' style='background:#f6f6f6'>"
            f"{html_mod.escape(t)}: <span class='neg'>{d:.3f}</span></span>"
            for t, d in sorted(per_t.items(), key=lambda kv: kv[1])
        )
        return (
            "<div class='fcell'>"
            f"<div class='fhdr'><code>OLMO f{fid}</code> "
            f"<span class='pos'>avg {metric_label}={avg_d:.3f}</span> "
            f"<span class='meta'>max={max_d:.3f}, n={info['n_tasks']}</span>"
            f"{spread_str}</div>"
            f"<div class='fdesc'>{desc}</div>"
            f"<div class='otherrow'><b>per task in cluster:</b> {per_task_chips}</div>"
            f"<div class='otherrow'><b>also matches:</b> {chips}</div>"
            "<details><summary style='cursor:pointer;color:#06c;font-size:0.85em;'>"
            f"▸ z-norm trajectory + samples (sorted by z-{feat_metric_name} "
            "to cluster mean)</summary>"
            f"<div id='{pid}' style='width:100%;height:260px;'></div>"
            f"{render_samples(meta.get('samples', []), cluster_target, feat_metric_name, f'z-{feat_metric_name}')}"
            "</details></div>"
        )

    # Rank features by avg distance — no thresholds, just top-N.
    min_presence = max(1, (len(members) + 1) // 2)
    rows = []
    for fid, info in fmap.items():
        if info["n_tasks"] < min_presence:
            continue
        meta = feat_meta.get(fid) or {}
        if require_desc and not meta.get("description"):
            continue
        rows.append((fid, info))

    rows.sort(key=lambda fi: fi[1]["avg_dist"])
    total = len(rows)
    if top_k > 0:
        rows = rows[:top_k]

    parts = [
        f"<h2 id='cluster-{ci}' style='border-bottom-color:{ci_color};'>"
        f"<span style='color:{ci_color}'>Cluster {ci}</span> "
        f"<span class='meta'>({len(members)} tasks: {', '.join(members)})</span></h2>",
        f"<h3><span class='sectlabel' style='background:{ci_color}22;border-color:{ci_color}66;'>"
        f"top {len(rows)} features by avg {metric_label}</span>"
        f"<span class='meta'>of {total} total (min presence {min_presence}/{len(members)} tasks)</span></h3>",
        "<div class='panel'>",
    ]
    if not rows:
        parts.append("<div class='meta'>no features available</div>")
    for fid, info in rows:
        parts.append(render_one(fid, info))
    parts.append("</div>")
    return "".join(parts)


def render(out_path, model_id, ckpts, task_perf, feat_meta,
           cluster_map, cluster_scores, chosen_k,
           metric_name, feat_metric_name,
           linkage_method, top_k, require_desc):

    metric_label = {"mse": "MSE", "area": "area"}[metric_name]
    feat_metric_label = {"mse": "MSE", "area": "area"}[feat_metric_name]

    # Cluster header color (used only for the cluster heading and section chip).
    cluster_color = {ci: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                     for i, ci in enumerate(sorted(cluster_map))}
    # Give every task a distinct color via plotly's qualitative palette so they
    # are visually distinguishable inside a single cluster's plot.
    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    task_colors = {t: PALETTE[i % len(PALETTE)]
                   for i, t in enumerate(sorted(task_perf))}

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Task shape clusters (OLMO)</title>",
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,sans-serif;margin:20px;background:#fafafa;}",
        "h1{margin-top:0;}h2{margin:30px 0 8px;border-bottom:2px solid #333;padding-bottom:4px;}",
        "h3{margin:14px 0 6px;color:#444;font-size:1.05em;}",
        ".banner{background:#fffbe6;border:1px solid #f0d871;padding:8px 12px;border-radius:6px;margin:8px 0;}",
        ".panel{background:white;border:1px solid #ddd;border-radius:6px;padding:8px 10px;}",
        ".paneltitle{font-weight:bold;margin-bottom:6px;font-size:1.05em;}",
        ".fcell{border:1px solid #eee;border-radius:5px;padding:6px 8px;margin:5px 0;background:#fdfdfd;}",
        ".fhdr{font-size:0.9em;margin-bottom:2px;}",
        ".fdesc{font-size:0.85em;color:#555;margin-bottom:3px;}",
        ".otherrow{font-size:0.78em;margin:3px 0;color:#333;}",
        ".pos{color:#0a0;font-family:monospace;}.neg{color:#a00;font-family:monospace;}",
        ".meta{color:#888;font-size:0.85em;}",
        ".sectlabel{display:inline-block;background:#eef;border:1px solid #ccd;border-radius:10px;"
        "padding:2px 10px;font-size:0.85em;margin-right:6px;}",
        ".taskchip{display:inline-block;background:#eef;border:1px solid #ccd;border-radius:10px;"
        "padding:1px 8px;margin:1px 3px;font-size:0.78em;font-family:monospace;}",
        ".toc{background:white;border:1px solid #ddd;border-radius:6px;padding:8px 12px;margin:8px 0;}",
        ".toc a{margin-right:10px;font-size:0.9em;}",
        "</style></head><body>",
        f"<h1>Task shape clusters — OLMO ({html_mod.escape(model_id)})</h1>",
        "<div class='banner'>"
        f"Tasks clustered by shape of their performance trajectory "
        f"(z-normalized; clustering distance = <b>{metric_label}</b>, "
        f"<code>{linkage_method}</code> linkage, k={chosen_k}).<br>"
        f"&nbsp;&nbsp;• For each cluster: top <b>{top_k}</b> OLMO features by avg "
        f"<b>{feat_metric_label}</b> distance between the feature's "
        f"(task-independent) median_probs curve and the cluster's task curves.<br>"
        "&nbsp;&nbsp;• <b>also matches</b> chips list ALL other clusters with "
        "their avg distance, sorted closest first."
        "</div>",
    ]

    parts.append("<div class='toc'><b>Jump to:</b> ")
    for ci in sorted(cluster_map):
        parts.append(
            f"<a href='#cluster-{ci}' style='color:{cluster_color[ci]}'>"
            f"cluster {ci} ({len(cluster_map[ci])})</a>"
        )
    parts.append("</div>")

    plot_specs = []
    plot_idx_ref = [0]

    for ci in sorted(cluster_map):
        members = cluster_map[ci]
        parts.append(render_cluster_block(
            ci, members, cluster_color[ci], cluster_scores.get(ci, {}),
            feat_meta, task_perf, ckpts, task_colors,
            cluster_scores,
            top_k, require_desc, feat_metric_label,
            feat_metric_name,
            plot_specs, plot_idx_ref))

    parts.append("<script>")
    parts.append("const PLOTS = " + json.dumps({d: f for d, f in plot_specs}) + ";")
    parts.append("""
document.addEventListener('toggle', function(ev){
  if (!(ev.target instanceof HTMLDetailsElement) || !ev.target.open) return;
  ev.target.querySelectorAll('div[id]').forEach(function(div){
    if (PLOTS[div.id] && !div.dataset.plotted) {
      Plotly.newPlot(div.id, PLOTS[div.id].data, PLOTS[div.id].layout, {responsive:true, displaylogo:false});
      div.dataset.plotted = '1';
    }
  });
}, true);
""")
    parts.append("</script></body></html>")
    out_path.write_text("".join(parts))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["mse", "area"], default="area",
                    help="Distance metric for clustering tasks by curve shape.")
    ap.add_argument("--feat-metric", choices=["mse", "area"], default=None,
                    help="Distance metric for ranking features against each cluster's "
                         "task curves. Defaults to whatever --metric is set to.")
    ap.add_argument("--linkage", default="average",
                    choices=["average", "complete", "single", "ward"],
                    help="Hierarchical clustering linkage method.")
    ap.add_argument("--k", type=int, default=None,
                    help="Number of clusters (override auto dendrogram-gap pick).")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Top-N features per cluster by avg distance.")
    ap.add_argument("--require-desc", action="store_true", default=True)
    ap.add_argument("--no-require-desc", dest="require_desc", action="store_false")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="Model JSON filename stem under precomputed_data_*/.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "task_shape_groups_olmo.html")
    args = ap.parse_args()

    print(f"loading {args.model_id}...")
    ckpts, task_perf, feat_meta = load_native_data(args.model_id)
    print(f"  {len(feat_meta)} features / {len(task_perf)} tasks")

    cluster_fn = METRIC_FNS[args.metric]
    feat_metric_name = args.feat_metric or args.metric
    feat_fn = METRIC_FNS[feat_metric_name]

    cluster_map, _, _, chosen_k, _ = cluster_tasks(
        task_perf, cluster_fn, args.linkage, k_override=args.k)
    print(f"  clustered into k={chosen_k} via {args.metric}:")
    for ci, members in sorted(cluster_map.items()):
        print(f"    {ci}: {members}")

    cluster_scores = score_features_per_cluster(
        cluster_map, task_perf, feat_meta, feat_fn)

    render(args.out, args.model_id, ckpts, task_perf, feat_meta,
           cluster_map, cluster_scores, chosen_k,
           args.metric, feat_metric_name,
           args.linkage, args.top_k, args.require_desc)


if __name__ == "__main__":
    main()
