"""Per-task feature listing for a single model (OLMO), ranked by curve shape.

Like `group_features_by_task_olmo.py` but uses z-normalized shape distance
(MSE or area between curves) instead of correlation as the ranking metric.

For each task:
  - Lists every OLMO feature whose (task-independent) median_probs curve is
    close to the task's overall_performance curve under the z-norm distance.
  - For each feature, annotates which OTHER tasks it also matches under
    threshold (cross-task chips).
"""

import argparse
import glob
import html as html_mod
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse loaders + math from the cluster-wise script.
from task_shape_groups_olmo import (
    ROOT, VALID_TASKS, DEFAULT_MODEL_ID,
    load_native_data, z_normalize, METRIC_FNS,
    typical_std, truncate, render_samples,
)


def score_features_per_task(task_perf, feat_meta, distance_fn):
    """Returns {task: {fid: dist}} — distance from each feature's z-norm curve
    to each task's z-norm performance curve."""
    task_z = {}
    for t, p in task_perf.items():
        nz = z_normalize(p)
        if nz is not None:
            task_z[t] = nz

    out = {t: {} for t in task_z}
    for fid, meta in feat_meta.items():
        fz = z_normalize(meta.get("median_probs"))
        if fz is None:
            continue
        for t, tz in task_z.items():
            d = distance_fn(tz, fz)
            if not np.isnan(d):
                out[t][fid] = float(d)
    return out


def make_plot_spec(ckpts, task_z_curve, feat_z_curve, task_name, feat_label,
                   task_color, feat_color):
    traces = []
    if task_z_curve is not None:
        idxs, vals = task_z_curve
        x = [ckpts[i] if i < len(ckpts) else str(i) for i in idxs]
        traces.append({
            "x": x, "y": list(vals), "type": "scatter", "mode": "lines+markers",
            "name": f"task: {task_name}",
            "line": {"color": task_color, "width": 2}, "marker": {"size": 5},
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
        "height": 240,
        "margin": {"l": 50, "r": 20, "t": 10, "b": 60},
        "xaxis": {"tickangle": -45, "tickfont": {"size": 9}},
        "yaxis": {"title": "z-score", "titlefont": {"size": 10}},
        "legend": {"font": {"size": 8}, "orientation": "h", "y": -0.4},
        "hovermode": "x unified",
    }
    return {"data": traces, "layout": layout}


def other_tasks_sorted(fid, task_scores, exclude_task):
    """All other tasks where this feature has a valid distance, sorted ascending."""
    rows = []
    for t, fmap in task_scores.items():
        if t == exclude_task:
            continue
        d = fmap.get(fid)
        if d is not None:
            rows.append((t, d))
    rows.sort(key=lambda kv: kv[1])
    return rows


def render_task_block(task, task_color, fmap, feat_meta, task_perf, ckpts,
                      task_scores, top_k, require_desc,
                      metric_label, plot_specs, plot_idx_ref):

    task_z = z_normalize(task_perf.get(task, []))

    def render_other_task_chips(fid):
        rows = other_tasks_sorted(fid, task_scores, task)
        if not rows:
            return ("<span style='color:#aaa;font-style:italic;font-size:0.78em;'>"
                    "(only this task)</span>")
        chips = []
        for t, d in rows:
            chips.append(
                f"<span class='taskchip'>{html_mod.escape(t)}: "
                f"<span class='pos'>{d:.3f}</span></span>"
            )
        return " ".join(chips)

    def render_one(fid, dist):
        meta = feat_meta.get(fid) or {}
        desc = html_mod.escape(truncate(meta.get("description", "")))
        med_std = typical_std(meta.get("std_probs"))
        spread_str = (f" <span class='meta'>med std={med_std:.3f}</span>"
                      if med_std is not None else "")
        feat_z = z_normalize(meta.get("median_probs"))

        pid = f"plot_olmo_{plot_idx_ref[0]}"
        plot_idx_ref[0] += 1
        plot_specs.append((pid, make_plot_spec(
            ckpts, task_z, feat_z, task, f"OLMO f{fid}", task_color, "#222")))

        chips = render_other_task_chips(fid)
        return (
            "<div class='fcell'>"
            f"<div class='fhdr'><code>OLMO f{fid}</code> "
            f"<span class='pos'>{metric_label}={dist:.3f}</span>"
            f"{spread_str}</div>"
            f"<div class='fdesc'>{desc}</div>"
            f"<div class='otherrow'><b>also matches:</b> {chips}</div>"
            "<details><summary style='cursor:pointer;color:#06c;font-size:0.85em;'>"
            f"▸ z-norm trajectory + samples (sorted by z-{metric_label} to task curve)</summary>"
            f"<div id='{pid}' style='width:100%;height:240px;'></div>"
            f"{render_samples(meta.get('samples', []), task_z, metric_label.lower() if metric_label.lower() in ('mse','area') else 'area', f'z-{metric_label}')}"
            "</details></div>"
        )

    rows = []
    for fid, dist in fmap.items():
        meta = feat_meta.get(fid) or {}
        if require_desc and not meta.get("description"):
            continue
        rows.append((fid, dist))
    rows.sort(key=lambda fi: fi[1])
    total = len(rows)
    if top_k > 0:
        rows = rows[:top_k]

    parts = [
        f"<h2 id='task-{html_mod.escape(task)}' "
        f"style='border-bottom-color:{task_color};'>"
        f"<span style='color:{task_color}'>{html_mod.escape(task)}</span> "
        f"<span class='meta'>({len(rows)}/{total} features shown)</span></h2>",
        "<div class='panel'>",
    ]
    if not rows:
        parts.append("<div class='meta'>no features available</div>")
    for fid, dist in rows:
        parts.append(render_one(fid, dist))
    parts.append("</div>")
    return "".join(parts)


def render(out_path, model_id, ckpts, task_perf, feat_meta,
           task_scores, metric_name,
           top_k, require_desc, tasks_to_render):

    metric_label = {"mse": "MSE", "area": "area"}[metric_name]

    PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
    task_colors = {t: PALETTE[i % len(PALETTE)]
                   for i, t in enumerate(sorted(task_perf))}

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Per-task features by shape (OLMO)</title>",
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,sans-serif;margin:20px;background:#fafafa;}",
        "h1{margin-top:0;}h2{margin:30px 0 8px;border-bottom:2px solid #333;padding-bottom:4px;}",
        "h3{margin:14px 0 6px;color:#444;font-size:1.05em;}",
        ".banner{background:#fffbe6;border:1px solid #f0d871;padding:8px 12px;border-radius:6px;margin:8px 0;}",
        ".panel{background:white;border:1px solid #ddd;border-radius:6px;padding:8px 10px;}",
        ".fcell{border:1px solid #eee;border-radius:5px;padding:6px 8px;margin:5px 0;background:#fdfdfd;}",
        ".fhdr{font-size:0.9em;margin-bottom:2px;}",
        ".fdesc{font-size:0.85em;color:#555;margin-bottom:3px;}",
        ".otherrow{font-size:0.78em;margin:3px 0;color:#333;}",
        ".pos{color:#0a0;font-family:monospace;}.neg{color:#a00;font-family:monospace;}",
        ".meta{color:#888;font-size:0.85em;}",
        ".taskchip{display:inline-block;background:#eef;border:1px solid #ccd;border-radius:10px;"
        "padding:1px 8px;margin:1px 3px;font-size:0.78em;font-family:monospace;}",
        ".toc{background:white;border:1px solid #ddd;border-radius:6px;padding:8px 12px;margin:8px 0;}",
        ".toc a{margin-right:10px;font-size:0.9em;}",
        "</style></head><body>",
        f"<h1>Per-task features by curve shape — OLMO ({html_mod.escape(model_id)})</h1>",
        "<div class='banner'>"
        f"For each task: top <b>{top_k}</b> OLMO features by z-norm "
        f"<b>{metric_label}</b> distance between their (task-independent) "
        f"median_probs curve and the task's performance curve.<br>"
        "&nbsp;&nbsp;• <b>also matches</b> chips list ALL other tasks for "
        "this feature, sorted by distance ascending."
        "</div>",
    ]

    parts.append("<div class='toc'><b>Jump to:</b> ")
    for t in tasks_to_render:
        col = task_colors.get(t, "#333")
        parts.append(
            f"<a href='#task-{html_mod.escape(t)}' style='color:{col}'>"
            f"{html_mod.escape(t)}</a>"
        )
    parts.append("</div>")

    plot_specs = []
    plot_idx_ref = [0]

    for task in tasks_to_render:
        parts.append(render_task_block(
            task, task_colors.get(task, "#333"), task_scores.get(task, {}),
            feat_meta, task_perf, ckpts,
            task_scores, top_k, require_desc,
            metric_label, plot_specs, plot_idx_ref))

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
                    help="Distance metric between z-normalized curves.")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Top-N features per task by distance.")
    ap.add_argument("--require-desc", action="store_true", default=True)
    ap.add_argument("--no-require-desc", dest="require_desc", action="store_false")
    ap.add_argument("--task", type=str, default=None,
                    help="If set, only render this single task. Otherwise renders all.")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help="Model JSON filename stem under precomputed_data_*/.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "group_features_by_task_shape_olmo.html")
    args = ap.parse_args()

    print(f"loading {args.model_id}...")
    ckpts, task_perf, feat_meta = load_native_data(args.model_id)
    print(f"  {len(feat_meta)} features / {len(task_perf)} tasks")

    distance_fn = METRIC_FNS[args.metric]
    task_scores = score_features_per_task(task_perf, feat_meta, distance_fn)
    print(f"  scored against {len(task_scores)} tasks with metric={args.metric}")

    tasks_available = sorted(task_scores)
    if args.task:
        if args.task not in tasks_available:
            raise SystemExit(f"--task {args.task!r} not in available tasks: {tasks_available}")
        tasks_to_render = [args.task]
    else:
        tasks_to_render = tasks_available

    render(args.out, args.model_id, ckpts, task_perf, feat_meta,
           task_scores, args.metric,
           args.top_k, args.require_desc, tasks_to_render)


if __name__ == "__main__":
    main()
