"""HTML viewer of features whose biggest jump (or biggest drop) lands on the same
checkpoint interval as a task's biggest jump (or drop).

For each task we z-normalize its performance curve, take consecutive-checkpoint
differences, and locate the single largest positive step (its "jump") and the
single most negative step (its "drop"). For every feature we do the same to its
median-probability curve. A feature is:

  * jump-aligned  if its biggest jump interval == the task's, within --tol;
  * drop-aligned  if its biggest drop interval == the task's, within --tol.

Per task, two columns list the aligned features (ranked by the magnitude of the
feature's own jump/drop), each expandable to a dual-axis curve plot (task perf +
feature median) and its top activating samples. A chance-rate note is shown so
above/below-random alignment is obvious.

Usage:
    python jump_alignment_to_html.py
    python jump_alignment_to_html.py --tol 1 --top-n 30
"""

import argparse
import glob
import html
import json
import os

import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STEM = "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm"
VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


# ── curve helpers ──────────────────────────────────────────────────────────

def znorm(vals):
    """Z-normalize a curve, dropping None. Returns (idxs, zvals) or None."""
    pairs = [(i, v) for i, v in enumerate(vals or []) if v is not None]
    if len(pairs) < 3:
        return None
    idxs = [i for i, _ in pairs]
    x = np.array([v for _, v in pairs], dtype=float)
    if x.std() < 1e-12:
        return None
    return idxs, (x - x.mean()) / x.std()


def jump_drop_intervals(curve):
    """Left-checkpoint index of the largest positive step (jump) and most
    negative step (drop), plus their signed magnitudes (in z units)."""
    idxs, z = curve
    iv = idxs[1:]
    dv = np.diff(z)
    j = int(np.argmax(dv))
    d = int(np.argmin(dv))
    return (iv[j], float(dv[j])), (iv[d], float(dv[d]))


# ── data loading ───────────────────────────────────────────────────────────

def load_task(stem, task):
    fp = os.path.join(ANALYSIS_DIR, f"precomputed_data_{task}", f"{stem}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def build_task(stem, task, tol):
    """Return a dict with the task's curve, jump/drop intervals, and the two
    aligned feature lists, or None if unavailable."""
    data = load_task(stem, task)
    if data is None:
        return None
    tcur = znorm(data.get("overall_performance"))
    if tcur is None:
        return None
    (t_jump_iv, _), (t_drop_iv, _) = jump_drop_intervals(tcur)

    ckpts = data.get("checkpoints") or []
    jump_rows, drop_rows = [], []
    seen = set()
    n_features = 0
    for feat in data.get("features", []):
        fid = feat["feature_id"]
        if fid in seen:
            continue
        seen.add(fid)
        fcur = znorm(feat.get("median_probs"))
        if fcur is None:
            continue
        n_features += 1
        (f_jump_iv, f_jump_mag), (f_drop_iv, f_drop_mag) = jump_drop_intervals(fcur)
        row = {
            "fid": fid,
            "desc": feat.get("description", ""),
            "median_probs": feat.get("median_probs", []),
            "samples": feat.get("samples", []) or [],
        }
        if abs(f_jump_iv - t_jump_iv) <= tol:
            jump_rows.append({**row, "mag": f_jump_mag, "iv": f_jump_iv})
        if abs(f_drop_iv - t_drop_iv) <= tol:
            drop_rows.append({**row, "mag": f_drop_mag, "iv": f_drop_iv})

    jump_rows.sort(key=lambda r: -r["mag"])   # sharpest rises first
    drop_rows.sort(key=lambda r: r["mag"])    # sharpest drops first (most negative)

    return {
        "task": task,
        "ckpts": ckpts,
        "task_perf": data.get("overall_performance", []),
        "t_jump_iv": t_jump_iv,
        "t_drop_iv": t_drop_iv,
        "jump_rows": jump_rows,
        "drop_rows": drop_rows,
        "n_features": n_features,
        "n_intervals": len(tcur[0]) - 1,
    }


# ── rendering ──────────────────────────────────────────────────────────────

def esc(s):
    return html.escape(str(s) if s is not None else "")


def ckpt_label(ckpts, i):
    return ckpts[i] if ckpts and i < len(ckpts) else f"idx{i}"


def interval_label(ckpts, iv):
    return f"{ckpt_label(ckpts, iv)} → {ckpt_label(ckpts, iv + 1)}"


def make_plot_spec(ckpts, feat_med, task_perf, fid_label, task_label,
                   hi_interval, feat_color):
    traces = []
    if task_perf is not None:
        traces.append({
            "x": list(ckpts), "y": list(task_perf), "type": "scatter",
            "mode": "lines+markers", "name": task_label,
            "line": {"color": "#444", "width": 1.5},
            "marker": {"size": 4}, "yaxis": "y",
        })
    if feat_med is not None:
        traces.append({
            "x": list(ckpts), "y": list(feat_med), "type": "scatter",
            "mode": "lines+markers", "name": fid_label,
            "line": {"color": feat_color, "width": 2.5},
            "marker": {"size": 6}, "yaxis": "y2",
        })
    shapes = []
    if hi_interval is not None and ckpts and hi_interval + 1 < len(ckpts):
        shapes.append({
            "type": "rect", "xref": "x", "yref": "paper",
            "x0": ckpts[hi_interval], "x1": ckpts[hi_interval + 1],
            "y0": 0, "y1": 1,
            "fillcolor": feat_color, "opacity": 0.12, "line": {"width": 0},
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
        "hovermode": "x unified", "shapes": shapes,
    }
    return {"data": traces, "layout": layout}


def render_samples(samples, max_samples):
    ordered = sorted(
        samples,
        key=lambda s: s.get("cos_sim") if s.get("cos_sim") is not None else -1e9,
        reverse=True,
    )[:max_samples]
    rows = []
    for s in ordered:
        rows.append(
            f"<tr><td class='before'>{esc(s.get('before',''))}</td>"
            f"<td class='word'>{esc(s.get('word',''))}</td>"
            f"<td class='after'>{esc(s.get('after',''))}</td>"
            f"<td class='num'>{(s.get('cos_sim') or 0):.3f}</td>"
            f"<td class='num'>{(s.get('activation') or 0):.3f}</td></tr>"
        )
    return (
        "<div class='samples-wrap'><table class='samples'><thead><tr>"
        "<th>Before</th><th>Word</th><th>After</th><th>Cos</th><th>Act</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_column(rows, label, color, ckpts, task_perf, task_label,
                  plot_specs, pid_prefix, top_n, max_samples):
    blocks = []
    for k, r in enumerate(rows[:top_n]):
        pid = f"{pid_prefix}_{k}"
        plot_specs[pid] = make_plot_spec(
            ckpts, r["median_probs"], task_perf, f"f{r['fid']}", task_label,
            r["iv"], color)
        metrics = (
            f"<span class='metric'>mag={r['mag']:+.2f}z</span>"
            f"<span class='metric'>@{interval_label(ckpts, r['iv'])}</span>"
        )
        head = (f"<summary><span class='fid'>#{esc(r['fid'])}</span>{metrics}"
                f"<span class='desc'>{esc(r['desc'])}</span></summary>")
        body = (f"<div class='feat-body'><div class='feat-plot' id='{pid}'></div>"
                f"<div class='feat-samples'>{render_samples(r['samples'], max_samples)}"
                f"</div></div>")
        blocks.append(f"<details class='feature'>{head}{body}</details>")
    if not blocks:
        blocks.append("<div class='empty'>No aligned features.</div>")
    return (f"<div class='col'><h4>{label} <small>({len(rows)})</small></h4>"
            f"{''.join(blocks)}</div>")


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; color: #222; background: #fafafa; }
header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #ddd;
         position: sticky; top: 0; z-index: 10; }
h1 { margin: 0 0 4px; font-size: 18px; }
header .meta { color: #666; font-size: 12px; }
.task { background: #fff; border: 1px solid #ddd; border-radius: 6px;
        padding: 16px; margin: 18px 24px; }
.task h3 { margin: 0 0 4px; font-size: 15px; }
.task .sub { color: #666; font-size: 12px; margin: 0 0 10px; }
.feat-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.col h4 { margin: 0 0 8px; font-size: 13px; padding-bottom: 6px;
          border-bottom: 1px solid #eee; }
.col h4 small { color: #888; font-weight: normal; }
.feature { background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px;
           margin-bottom: 4px; }
.feature[open] { background: #fff; border-color: #b8c8e0; }
.feature summary { padding: 6px 10px; cursor: pointer; font-size: 12px;
                   display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.feature summary .fid { font-family: monospace; font-weight: 600; color: #335; }
.feature summary .metric { font-family: monospace; font-size: 11px; color: #666; }
.feature summary .desc { color: #555; font-size: 11.5px; flex: 1 1 100%; margin-top: 2px; }
.feat-body { display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
             padding: 8px; align-items: start; }
.feat-plot { width: 100%; height: 240px; background: #fff;
             border: 1px solid #e0e0e0; border-radius: 3px; }
.empty { color: #999; font-size: 12px; padding: 8px; font-style: italic; }
.samples-wrap { max-height: 280px; overflow-y: auto; border: 1px solid #e0e0e0;
                border-radius: 3px; background: #fff; }
table.samples { width: 100%; border-collapse: collapse; font-size: 11.5px; }
table.samples thead th { position: sticky; top: 0; background: #f0f0f0;
                         padding: 5px 6px; border-bottom: 1px solid #ddd; text-align: left; }
table.samples td { padding: 3px 6px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
table.samples td.before { text-align: right; color: #888; max-width: 220px;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
table.samples td.word { font-weight: 700; white-space: nowrap; }
table.samples td.after { color: #888; max-width: 220px; overflow: hidden;
                         text-overflow: ellipsis; white-space: nowrap; }
table.samples td.num { font-family: monospace; text-align: right; white-space: nowrap; }
@media (max-width: 1100px) { .feat-body { grid-template-columns: 1fr; } }
"""

JS = """
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
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--tol", type=int, default=0,
                    help="Jump/drop coincidence tolerance in #intervals (0=exact).")
    ap.add_argument("--top-n", type=int, default=25,
                    help="Max features shown per column.")
    ap.add_argument("--max-samples", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR, "jump_alignment.html"))
    args = ap.parse_args()

    tasks = sorted(VALID_TASKS)
    plot_specs = {}
    sections = []
    for t in tasks:
        info = build_task(args.stem, t, args.tol)
        if info is None:
            continue
        ck = info["ckpts"]
        chance = (2 * args.tol + 1) / max(1, info["n_intervals"])
        jrate = len(info["jump_rows"]) / max(1, info["n_features"])
        drate = len(info["drop_rows"]) / max(1, info["n_features"])
        sub = (f"biggest jump @ {interval_label(ck, info['t_jump_iv'])} "
               f"&middot; biggest drop @ {interval_label(ck, info['t_drop_iv'])} "
               f"&middot; {info['n_features']} features &middot; "
               f"jump-aligned {jrate:.0%}, drop-aligned {drate:.0%} "
               f"(chance ≈ {chance:.0%})")
        jcol = render_column(
            info["jump_rows"], "Jump-aligned (rises with task)", "#1976d2",
            ck, info["task_perf"], f"{t} perf", plot_specs,
            f"{t}_j", args.top_n, args.max_samples)
        dcol = render_column(
            info["drop_rows"], "Drop-aligned (falls with task)", "#c62828",
            ck, info["task_perf"], f"{t} perf", plot_specs,
            f"{t}_d", args.top_n, args.max_samples)
        sections.append(
            f"<section class='task'><h3>{esc(t)}</h3>"
            f"<p class='sub'>{sub}</p>"
            f"<div class='feat-cols'>{jcol}{dcol}</div></section>")

    plots_json = json.dumps(plot_specs)
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Jump/Drop Alignment — {esc(args.stem)}</title>
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>{CSS}</style></head><body>
<header><h1>Feature Jump / Drop Alignment with Task Performance</h1>
<div class='meta'>Dataset: <code>{esc(args.stem)}</code> &middot; tolerance: {args.tol} interval(s)
&middot; features whose largest rise (or fall) lands on the same checkpoint interval
as the task's largest rise (or fall), ranked by jump/drop magnitude.</div></header>
{''.join(sections)}
<script>const PLOTS = {plots_json};{JS}</script>
</body></html>"""

    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
