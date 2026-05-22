"""Per-task feature listing for a single model (OLMO).

For a chosen task (or all tasks):
  - Lists every OLMO feature that passes the sign-specific correlation
    threshold for this task.
  - For each feature, annotates which OTHER tasks (besides this one) it
    also passes the threshold for (cross-task chips).
"""

import argparse
import glob
import html as html_mod
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/nsrikant/BehaviorBoxNew/analysis")

VALID_TASKS = [
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
]

NATIVE = {
    "OLMO": {
        "model_id": "OLMo3-7b-256k-3000-k25-0.8-early-checkpoints",
        "sae_folder": Path(
            "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early/n_only_early_olmo3_seed=42_ofw=0.8_N=3000_k=25_lp=None_znorm_odlw=auto"
        ),
        "column_prefix": "olmo3-",
    },
}

POS_CORR_KEY = "partial_diff_spearman_corr"
NEG_CORR_KEY = "diff_spearman_corr"


def load_native_corrs_and_meta(model_id: str):
    """Returns (ckpts, task_perf, pos_corrs, neg_corrs, feat_meta,
    median_by_task, std_by_task)."""
    ckpts = None
    task_perf = {}
    pos_corrs = defaultdict(dict)
    neg_corrs = defaultdict(dict)
    feat_meta = {}
    median_by_task = defaultdict(dict)
    std_by_task = defaultdict(dict)
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
            c_pos = feat.get(POS_CORR_KEY)
            if c_pos is not None and np.isfinite(c_pos):
                pos_corrs[fid][tname] = float(c_pos)
            c_neg = feat.get(NEG_CORR_KEY)
            if c_neg is not None and np.isfinite(c_neg):
                neg_corrs[fid][tname] = float(c_neg)
            desc = feat.get("description", "")
            if fid not in feat_meta:
                feat_meta[fid] = {"description": desc}
            elif desc and not feat_meta[fid].get("description"):
                feat_meta[fid]["description"] = desc
            med = feat.get("median_probs") or feat.get("avg_probs")
            if med is not None:
                median_by_task[fid][tname] = [float(x) for x in med]
            std = feat.get("std_probs")
            if std is not None:
                std_by_task[fid][tname] = [
                    float(x) if x is not None and np.isfinite(x) else None
                    for x in std
                ]
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
    return (ckpts or [], task_perf, dict(pos_corrs), dict(neg_corrs),
            feat_meta, dict(median_by_task), dict(std_by_task))


def truncate(s, n=140):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def render_samples(samples):
    if not samples:
        return "<div style='color:#888;font-style:italic;padding:4px;'>no samples</div>"
    rows = []
    for s in samples:
        before = html_mod.escape(str(s.get("before", "")))
        word = html_mod.escape(str(s.get("word", "")))
        after = html_mod.escape(str(s.get("after", "")))
        cos = s.get("cos_sim", 0) or 0
        act = s.get("activation", 0) or 0
        rows.append(
            "<tr>"
            f"<td style='text-align:right;color:#888;font-size:0.85em;max-width:220px;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:2px 6px;'>{before}</td>"
            f"<td style='font-weight:bold;padding:2px 6px;white-space:nowrap;'>{word}</td>"
            f"<td style='color:#888;font-size:0.85em;max-width:220px;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;padding:2px 6px;'>{after}</td>"
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
            "<th style='text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'>cos</th>"
            "<th style='text-align:right;padding:2px 6px;border-bottom:1px solid #ccc;'>act</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def make_plot_spec(ckpts, feat_med, task_perf, fid_label, color):
    traces = []
    if task_perf is not None:
        traces.append({
            "x": list(ckpts), "y": list(task_perf), "type": "scatter",
            "mode": "lines+markers", "name": "task perf",
            "line": {"color": "#444", "width": 1.5}, "marker": {"size": 4},
            "yaxis": "y",
        })
    if feat_med is not None:
        traces.append({
            "x": list(ckpts), "y": list(feat_med), "type": "scatter",
            "mode": "lines+markers", "name": fid_label,
            "line": {"color": color, "width": 2.5}, "marker": {"size": 6},
            "yaxis": "y2",
        })
    layout = {
        "height": 220,
        "margin": {"l": 40, "r": 50, "t": 10, "b": 60},
        "xaxis": {"tickangle": -45, "tickfont": {"size": 9}},
        "yaxis": {"title": "task perf", "titlefont": {"size": 10}},
        "yaxis2": {"title": "feat med", "titlefont": {"size": 10, "color": color},
                   "tickfont": {"color": color}, "overlaying": "y", "side": "right"},
        "legend": {"font": {"size": 8}, "orientation": "h", "y": -0.4},
        "hovermode": "x unified",
    }
    return {"data": traces, "layout": layout}


def passes(corr, sign, threshold):
    if corr is None or not np.isfinite(corr):
        return False
    return corr >= threshold if sign == "+" else corr <= -threshold


def per_sample_spread(std_probs, median_probs=None, mode="mean"):
    """Summarize how much individual sample probability curves disagree at
    each checkpoint.

    `std_probs[t]` is the cross-sample std at checkpoint t. We summarize
    the trajectory of stds into a single scalar via `mode`:
      - "mean": mean std across checkpoints (absolute spread)
      - "max": max std across checkpoints
      - "cv": mean(std) / |mean(median)|  (relative spread, if median given)

    Returns None if data is missing/insufficient.
    """
    if not std_probs:
        return None
    arr = np.array([x for x in std_probs if x is not None and np.isfinite(x)],
                   dtype=float)
    if arr.size == 0:
        return None
    if mode == "max":
        return float(np.max(arr))
    mean_std = float(np.mean(arr))
    if mode == "cv":
        if not median_probs:
            return None
        med = np.array([x for x in median_probs
                        if x is not None and np.isfinite(x)], dtype=float)
        if med.size == 0:
            return None
        m = float(np.mean(np.abs(med)))
        if m < 1e-12:
            return None
        return mean_std / m
    return mean_std


def feats_passing_for_task(corrs_by_fid, task, sign, threshold, meta=None,
                           require_desc=False,
                           max_spread=None, spread_mode="mean",
                           std_by_task=None, median_by_task=None):
    """Returns {fid: corr_on_task} for fids whose corr on `task` passes.

    If `max_spread` is set, drops fids whose per-sample spread on `task`
    (summarized by `spread_mode`) exceeds it. Features missing std data
    are kept.
    """
    out = {}
    for fid, tc in corrs_by_fid.items():
        c = tc.get(task)
        if not passes(c, sign, threshold):
            continue
        feat_meta = (meta or {}).get(fid, {}) or {}
        if require_desc and not feat_meta.get("description"):
            continue
        if max_spread is not None:
            stds = (std_by_task or {}).get(fid, {}).get(task)
            meds = (median_by_task or {}).get(fid, {}).get(task)
            spread = per_sample_spread(stds, meds, mode=spread_mode)
            if spread is not None and spread > max_spread:
                continue
        out[fid] = c
    return out


def other_tasks_passing(corrs_by_fid, fid, sign, threshold, exclude_task,
                        valid_tasks):
    tc = corrs_by_fid.get(fid, {})
    rows = []
    for t, c in tc.items():
        if t == exclude_task or t not in valid_tasks:
            continue
        if passes(c, sign, threshold):
            rows.append((t, c))
    rows.sort(key=lambda kv: -abs(kv[1]))
    return rows


def render_task_block(task, sign_label, sign, threshold,
                      o_pass, olmo_corrs, olmo_meta,
                      olmo_med, olmo_std, olmo_task_perf, olmo_ckpts,
                      valid_tasks, top_k, spread_mode,
                      plot_specs, plot_idx_ref, color_o):

    def render_other_task_chips(fid, sign, all_corrs):
        rows = other_tasks_passing(all_corrs, fid, sign, threshold,
                                   exclude_task=task, valid_tasks=valid_tasks)
        if not rows:
            return ("<span style='color:#aaa;font-style:italic;font-size:0.78em;'>"
                    "no other tasks pass</span>")
        chips = []
        for t, c in rows:
            cls = "pos" if c > 0 else "neg"
            chips.append(
                f"<span class='taskchip'>{html_mod.escape(t)}: "
                f"<span class='{cls}'>{c:+.3f}</span></span>"
            )
        return " ".join(chips)

    def render_one(fid, corr):
        meta = (olmo_meta.get(fid) or {})
        desc = html_mod.escape(truncate(meta.get("description", "")))
        sign_class = "pos" if corr > 0 else "neg"
        samples = meta.get("samples", [])
        med = (olmo_med.get(fid) or {}).get(task)
        stds = (olmo_std.get(fid) or {}).get(task)
        spread = per_sample_spread(stds, med, mode=spread_mode)
        spread_str = (f" <span class='meta'>spread({spread_mode})={spread:.3f}</span>"
                      if spread is not None else "")
        perf = olmo_task_perf.get(task)
        pid = f"plot_olmo_{plot_idx_ref[0]}"
        plot_idx_ref[0] += 1
        if med is not None:
            plot_specs.append((pid, make_plot_spec(
                olmo_ckpts, med, perf, f"OLMO f{fid}", color_o)))
        other_chips = render_other_task_chips(fid, sign, olmo_corrs)
        other_block = f"<div class='otherrow'><b>also passes:</b> {other_chips}</div>"
        return (
            "<div class='fcell'>"
            f"<div class='fhdr'><code>OLMO f{fid}</code> "
            f"<span class='{sign_class}'>r={corr:+.3f}</span>{spread_str}</div>"
            f"<div class='fdesc'>{desc}</div>"
            f"{other_block}"
            "<details><summary style='cursor:pointer;color:#06c;font-size:0.85em;'>▸ trajectory + samples</summary>"
            f"<div id='{pid}' style='width:100%;height:220px;'></div>"
            f"{render_samples(samples)}"
            "</details></div>"
        )

    o_sorted = sorted(o_pass.items(), key=lambda kv: -abs(kv[1]))
    if top_k > 0:
        o_sorted = o_sorted[:top_k]

    parts = [
        f"<h3><span class='sectlabel'>{sign_label}</span>"
        f"<span class='meta'>OLMO: {len(o_sorted)}/{len(o_pass)} feats</span></h3>",
        "<div class='panel'><div class='paneltitle' style='color:#1976d2'>"
        f"OLMO ({len(o_sorted)})</div>",
    ]
    for fid, corr in o_sorted:
        parts.append(render_one(fid, corr))
    parts.append("</div>")
    return "".join(parts)


def render(tasks_to_render, valid_tasks,
           olmo_pos_corrs, olmo_neg_corrs,
           olmo_meta, olmo_med, olmo_std, olmo_task_perf, olmo_ckpts,
           threshold, top_k, require_desc,
           max_spread, spread_mode, out_path):

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Per-task features (OLMO)</title>",
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
        "<h1>Per-task features — OLMO</h1>",
        f"<div class='banner'>For each task: <b>OLMO</b> features that pass "
        f"the sign-specific threshold for that task.<br>"
        f"&nbsp;&nbsp;• <b>POS</b>: corr ≥ {threshold} (<code>{POS_CORR_KEY}</code>).<br>"
        f"&nbsp;&nbsp;• <b>NEG</b>: corr ≤ −{threshold} (<code>{NEG_CORR_KEY}</code>).<br>"
        f"&nbsp;&nbsp;• Each feature's <b>also passes</b> row shows OTHER tasks "
        f"(of the same sign) where it also passes.<br>"
        + (f"&nbsp;&nbsp;• <b>Per-sample spread filter</b>: dropping features "
           f"where {spread_mode}(std_probs) &gt; {max_spread} (computed from "
           f"<code>std_probs</code> across checkpoints).<br>"
           if max_spread is not None else "") +
        "</div>",
    ]

    parts.append("<div class='toc'><b>Jump to:</b> ")
    for t in tasks_to_render:
        parts.append(f"<a href='#task-{html_mod.escape(t)}'>{html_mod.escape(t)}</a>")
    parts.append("</div>")

    plot_specs = []
    plot_idx_ref = [0]
    color_o = "#1976d2"

    for task in tasks_to_render:
        parts.append(f"<h2 id='task-{html_mod.escape(task)}'>{html_mod.escape(task)}</h2>")
        for sign_label, sign in [("POSITIVE", "+"), ("NEGATIVE", "-")]:
            o_corrs = olmo_pos_corrs if sign == "+" else olmo_neg_corrs
            o_pass = feats_passing_for_task(
                o_corrs, task, sign, threshold, olmo_meta, require_desc,
                max_spread=max_spread, spread_mode=spread_mode,
                std_by_task=olmo_std, median_by_task=olmo_med)
            if not o_pass:
                parts.append(
                    f"<h3><span class='sectlabel'>{sign_label}</span>"
                    f"<span class='meta'>no features pass</span></h3>"
                )
                continue
            parts.append(render_task_block(
                task, sign_label, sign, threshold,
                o_pass, o_corrs, olmo_meta,
                olmo_med, olmo_std, olmo_task_perf, olmo_ckpts,
                valid_tasks, top_k, spread_mode,
                plot_specs, plot_idx_ref, color_o))

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
    ap.add_argument("--task", type=str, default=None,
                    help="If set, only render this single task. Otherwise "
                         "renders all available tasks.")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=0,
                    help="If > 0, limit feature lists to top-k per (task, sign) "
                         "by |corr|. 0 = unlimited.")
    ap.add_argument("--require-desc", action="store_true", default=True)
    ap.add_argument("--no-require-desc", dest="require_desc", action="store_false")
    ap.add_argument("--max-spread", type=float, default=0.1,
                    help="If set, drop features whose per-sample probability "
                         "spread on the task exceeds this value. The spread "
                         "summarizes std_probs across checkpoints; mode is "
                         "controlled by --spread-mode.")
    ap.add_argument("--spread-mode", choices=["mean", "max", "cv"],
                    default="mean",
                    help="How to reduce per-checkpoint std_probs to one "
                         "number. mean: mean std across checkpoints. "
                         "max: max std. cv: mean(std)/|mean(median)|.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "group_features_by_task_olmo.html")
    args = ap.parse_args()

    print("loading OLMO native...")
    (o_ckpts, o_perf, o_pos, o_neg, o_meta, o_med, o_std) = load_native_corrs_and_meta(NATIVE["OLMO"]["model_id"])
    print(f"  {len(o_pos)} feats with partial_spearman / {len(o_neg)} with spearman / {len(o_perf)} tasks")

    tasks_available = sorted(o_perf)
    print(f"  tasks: {tasks_available}")
    if args.task:
        if args.task not in tasks_available:
            raise SystemExit(f"--task {args.task!r} not in available tasks: {tasks_available}")
        tasks_to_render = [args.task]
    else:
        tasks_to_render = tasks_available

    render(tasks_to_render, tasks_available,
           o_pos, o_neg, o_meta, o_med, o_std, o_perf, o_ckpts,
           args.threshold, args.top_k, args.require_desc,
           args.max_spread, args.spread_mode, args.out)


if __name__ == "__main__":
    main()
