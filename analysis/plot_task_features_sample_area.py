"""Self-contained: per task, plot the features ranked by PER-SAMPLE area.

No precomputed_data_*/*.json, no pickle. Computes everything in memory and
writes one HTML file. Reads only:

  1. eval results   -> task performance curve (one value per checkpoint)
  2. SAE folder     -> top-50_activations.csv  (feature -> sample word_ids)
                       top-50_words_in_context.json (word_id -> before/word/after)
  3. memmap cache   -> each sample's per-checkpoint output trajectory

Per feature, each sample curve is z-normalized individually, then combined
against each z-normalized task curve two ways:
  (i)  med_of_area : area(sample_z, task_z) per sample, median over samples.
  (ii) area_of_med : area(median-of-sample-z-curves, task_z).
Features are ranked per task by area_of_med (lower = better shape match), in a
positive column (raw curve) and a negative column (sign-flipped).

z-norm means/stds: task curve over its valid checkpoints; each sample over its
own full trajectory.

Usage:
    python plot_task_features_sample_area.py                 # all tasks
    python plot_task_features_sample_area.py --tasks gsm8k arc_challenge
    python plot_task_features_sample_area.py --diff --top-n 30
"""

import argparse
import glob
import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config — defaults target olmo-gsm8k; all overridable via CLI.
# -----------------------------------------------------------------------------
EVAL_RESULTS_DIR = "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3"
COLUMN_PREFIX = "olmo3-"
SAE_FOLDER = "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/n_only_early_and_late_olmo3_seed=42_ofw=0.25_subset=sae_sample_by_task_gsm8k_942k_N=3000_k=25_lp=None_pznorm=0.01"
CACHE_DIR = "/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen/ofw=0.25_pznorm=0.01"

METRIC_PREFERENCE = [
    "acc_norm,none", "acc,none",
    "exact_match,strict-match", "exact_match,get-answer",
    "exact_match,flexible-extract", "exact,none", "em,none",
    "f1,none", "perplexity,none",
]


# -----------------------------------------------------------------------------
# Task curves (eval results)
# -----------------------------------------------------------------------------
def discover_tasks(eval_dir):
    ckpts = [d for d in glob.glob(os.path.join(eval_dir, "*")) if os.path.isdir(d)]
    if not ckpts:
        return []
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(ckpts[0], "*"))
                  if os.path.isdir(p))


def _detect_metric(task_results):
    for m in METRIC_PREFERENCE:
        if m in task_results:
            return m
    for k in task_results:
        if "stderr" not in k and k != "alias":
            return k
    return None


def load_task_curve(task, eval_dir, ckpt_names):
    perf, metric_used, result_key = [], None, None
    for cp in ckpt_names:
        model_dirs = glob.glob(os.path.join(eval_dir, cp, task, "*/"))
        rfiles = (glob.glob(os.path.join(model_dirs[0], "results_*.json"))
                  if model_dirs else [])
        if not rfiles:
            perf.append(None)
            continue
        results = json.load(open(rfiles[0])).get("results", {})
        if result_key is None:
            result_key = (task if task in results
                          else (next(iter(results)) if results else None))
        if metric_used is None and result_key in results:
            metric_used = _detect_metric(results[result_key])
        val = results.get(result_key, {}).get(metric_used)
        perf.append(float(val) if val is not None else None)
    return perf, metric_used


def znorm(values):
    """Z-norm over valid (non-None) entries. Returns (idxs, z) or None."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 3:
        return None
    idxs = [i for i, _ in valid]
    vals = np.array([v for _, v in valid], dtype=float)
    sd = vals.std()
    if sd < 1e-9:
        return None
    return idxs, (vals - vals.mean()) / sd


# -----------------------------------------------------------------------------
# Per-sample trajectories (inlined memmap reader)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_index(cache_dir):
    cache = Path(cache_dir)
    info = json.loads((cache / "cached_data_info.json").read_text())
    shape = tuple(info["shape"])
    n_outputs = int(info["output_feature_dim"])
    model_names = list(info["model_names"])
    dtype = {"np16": np.float16, "np32": np.float32,
             "float16": np.float16, "float32": np.float32}.get(
                 info.get("dtype", "np16"), np.float16)
    mm = np.memmap(info["filename"], dtype=dtype, mode="r", shape=shape)
    out_dir = Path(info["data_dir"]) / "output_features"
    step_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    f2d = pd.read_csv(step_dirs[0] / "file_to_doc.csv")
    offsets = np.cumsum(f2d["num_words"].values) - f2d["num_words"].values
    doc_offset = dict(zip(f2d["doc_id"].astype(int).tolist(),
                          offsets.astype(np.int64).tolist()))
    return mm, doc_offset, n_outputs, model_names


def get_sample_curve(word_id, cache_dir):
    parts = str(word_id).split("_")
    if len(parts) < 2:
        return None
    try:
        doc_id, w = int(parts[-2]), int(parts[-1])
    except ValueError:
        return None
    mm, doc_offset, n_outputs, _ = _load_index(cache_dir)
    if doc_id not in doc_offset:
        return None
    row = doc_offset[doc_id] + w
    if row < 0 or row >= mm.shape[0]:
        return None
    return np.asarray(mm[row, -n_outputs:], dtype=np.float32)


def batch_load_curves(word_ids, cache_dir):
    """One sorted, batched memmap read for many samples. Returns {wid: curve}.

    The memmap is tens of GB on NFS, so per-sample random reads dominate
    runtime. Resolving every row up front, reading them in ascending order
    (for readahead locality), and scattering back turns ~150k tiny scattered
    reads into a single vectorized gather."""
    mm, doc_offset, n_outputs, _ = _load_index(cache_dir)
    wids, rows = [], []
    for wid in dict.fromkeys(str(w) for w in word_ids):   # dedup, keep order
        parts = wid.split("_")
        if len(parts) < 2:
            continue
        try:
            doc_id, w = int(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        if doc_id not in doc_offset:
            continue
        r = doc_offset[doc_id] + w
        if 0 <= r < mm.shape[0]:
            wids.append(wid)
            rows.append(r)
    if not rows:
        return {}
    rows = np.asarray(rows)
    order = np.argsort(rows, kind="stable")
    block = np.empty((len(rows), n_outputs), dtype=np.float32)
    block[order] = np.asarray(mm[rows[order], -n_outputs:], dtype=np.float32)
    return {w: block[i] for i, w in enumerate(wids)}


def z_full(vec):
    v = np.asarray(vec, dtype=float)
    sd = v.std()
    return None if sd < 1e-9 else (v - v.mean()) / sd


# -----------------------------------------------------------------------------
# Area metric (vectorized over a feature's samples)
# -----------------------------------------------------------------------------
def _areas(rows, tvals):
    """rows: (n,k); tvals: (k,). Mean |rows - tvals| per row over checkpoints
    (uniform spacing). Matches the selection metric in sample_by_task_centroids.py
    (mean |Δz|, uniform), so the distances shown here are directly comparable to its
    --max-dist threshold (e.g. 0.5). Despite the name 'area', it's a mean now — not a
    log-step trapezoid integral."""
    return np.mean(np.abs(rows - tvals), axis=1)


def score_against_task(S, task_z, ckpt_names, diff):
    """S: (n_samples, n_ckpts) z-curves. Returns dict of the four scalars +
    per-sample positive areas (for sorting samples in the view)."""
    idxs, tvals = task_z
    if not idxs or max(idxs) >= S.shape[1]:
        nan = float("nan")
        return {"med_of_area_pos": nan, "med_of_area_neg": nan,
                "area_of_med_pos": nan, "area_of_med_neg": nan,
                "best_of_area_pos": nan, "best_of_area_neg": nan,
                "per_sample_pos": np.full(S.shape[0], nan),
                "per_sample_neg": np.full(S.shape[0], nan)}
    sub = S[:, np.array(idxs, dtype=int)]
    tv = np.asarray(tvals, dtype=float)
    if diff:
        sub, tv = np.diff(sub, axis=1), np.diff(tv)
        # Renormalize the differenced task curve to unit variance too, so the
        # distance compares jump *shapes* (both unit-var) rather than being skewed
        # by the task's diff amplitude — otherwise near-uncorrelated features
        # can win on scale alone.
        tvz = z_full(tv)
        if tvz is not None:
            tv = tvz

    ps_pos = _areas(sub, tv)
    ps_neg = _areas(-sub, tv)
    # Re-z-normalize the median curve before measuring area. Median of unit-var
    # z-curves shrinks toward 0 (uncorrelated jumps cancel), so without this the
    # area is dominated by amplitude mismatch and flat features win. Renorming
    # makes area_of_med a pure shape comparison; a degenerate (flat) median
    # becomes NaN and is dropped instead of rewarded.
    med_z = z_full(np.median(sub, axis=0))
    if med_z is None:
        aom_pos = aom_neg = float("nan")
    else:
        med = med_z[None, :]
        aom_pos = float(_areas(med, tv)[0])
        aom_neg = float(_areas(-med, tv)[0])
    return {
        "med_of_area_pos": float(np.median(ps_pos)),
        "med_of_area_neg": float(np.median(ps_neg)),
        "area_of_med_pos": aom_pos,
        "area_of_med_neg": aom_neg,
        # best single sample: the feature's closest-matching individual curve
        "best_of_area_pos": float(np.min(ps_pos)),
        "best_of_area_neg": float(np.min(ps_neg)),
        "per_sample_pos": ps_pos,
        "per_sample_neg": ps_neg,
    }


# -----------------------------------------------------------------------------
# HTML rendering
# -----------------------------------------------------------------------------
def esc(s):
    return html.escape(str(s) if s is not None else "")


def make_samples_plot_spec(ckpts, sample_curves, task_curve, task_label,
                           color, ylab="z-score"):
    """Overlay each top sample's z-curve (thin, translucent) with the task curve
    (bold). All curves are unit-var z (or Δz), so shape alignment is visible."""
    traces = []
    for k, sc in enumerate(sample_curves):
        traces.append({
            "x": list(ckpts), "y": list(sc), "type": "scatter", "mode": "lines",
            "name": "samples" if k == 0 else None,
            "legendgroup": "samples", "showlegend": k == 0,
            "line": {"color": color, "width": 1}, "opacity": 0.35,
            "hoverinfo": "skip"})
    traces.append({
        "x": list(ckpts), "y": list(task_curve), "type": "scatter",
        "mode": "lines+markers", "name": task_label,
        "line": {"color": "#111", "width": 3}, "marker": {"size": 5}})
    layout = {
        "height": 360, "margin": {"l": 50, "r": 20, "t": 10, "b": 80},
        "xaxis": {"tickangle": -45, "tickfont": {"size": 9}},
        "yaxis": {"title": ylab, "titlefont": {"size": 10}, "zeroline": True},
        "legend": {"font": {"size": 9}, "orientation": "h", "y": -0.35},
        "hovermode": "closest",
    }
    return {"data": traces, "layout": layout}


def render_samples_table(samples):
    rows = []
    for s in samples:
        d = s.get("dist")
        d_cell = f"{d:.4f}" if d is not None and d == d else "—"
        rows.append(
            f"<tr><td class='before'>{esc(s.get('before',''))}</td>"
            f"<td class='word'>{esc(s.get('word',''))}</td>"
            f"<td class='after'>{esc(s.get('after',''))}</td>"
            f"<td class='num'>{d_cell}</td>"
            f"<td class='num'>{(s.get('act') or 0):.4f}</td></tr>")
    return ("<div class='samples-wrap'><table class='samples'><thead><tr>"
            "<th>Before</th><th>Word</th><th>After</th>"
            "<th>z-area</th><th>Act</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def render_feature_block(fid, desc, metrics_html, plot_id, samples_html):
    head = (f"<summary><span class='fid'>#{esc(fid)}</span>{metrics_html}"
            f"<span class='desc'>{esc(desc)}</span></summary>")
    body = (f"<div class='feat-body'><div class='feat-plot' id='{esc(plot_id)}'></div>"
            f"<div class='feat-samples'>{samples_html}</div></div>")
    return f"<details class='feature'>{head}{body}</details>"


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; color: #222; background: #fafafa; }
header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #ddd; }
h1 { margin: 0 0 4px; font-size: 18px; }
header .meta { color: #666; font-size: 12px; }
.wrap { padding: 24px; }
.task-row { background: #fff; border: 1px solid #ddd; border-radius: 6px;
            padding: 16px; margin-bottom: 20px; }
.task-row h3 { margin: 0 0 6px; font-size: 15px; }
.counts { color: #888; font-size: 11px; margin: 0 0 8px; }
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
.feat-body { display: grid; grid-template-columns: 1fr; gap: 10px; padding: 8px;
             align-items: start; }
.feat-plot { width: 100%; height: 360px; background: #fff; border: 1px solid #e0e0e0;
             border-radius: 3px; }
.empty { color: #999; font-size: 12px; padding: 8px; font-style: italic; }
.samples-wrap { max-height: 280px; overflow-y: auto; border: 1px solid #e0e0e0;
                border-radius: 3px; background: #fff; }
table.samples { width: 100%; border-collapse: collapse; font-size: 11.5px; }
table.samples thead th { position: sticky; top: 0; background: #f0f0f0; padding: 5px 6px;
                         border-bottom: 1px solid #ddd; text-align: left; }
table.samples td { padding: 3px 6px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
table.samples td.before { text-align: right; color: #888; max-width: 220px; overflow: hidden;
                          text-overflow: ellipsis; white-space: nowrap; }
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sae-folder", default=SAE_FOLDER)
    ap.add_argument("--eval-dir", default=EVAL_RESULTS_DIR)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--column-prefix", default=COLUMN_PREFIX)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Eval task subfolder names. Default: auto-discover all.")
    ap.add_argument("--rank-by", default="best_of_area",
                    choices=["best_of_area", "area_of_med", "med_of_area"],
                    help="Which metric ranks features per task. Default "
                         "best_of_area: the feature's single closest-matching "
                         "sample curve.")
    ap.add_argument("--diff", action="store_true", default=False,
                    help="First-difference z-curves before integrating.")
    ap.add_argument("--top-n", type=int, default=25, help="Top features per column.")
    ap.add_argument("--max-samples", type=int, default=30,
                    help="Samples listed in the table per feature.")
    ap.add_argument("--plot-samples", type=int, default=12,
                    help="Top sample curves overlaid in each feature's plot.")
    ap.add_argument("--limit", type=int, default=None, help="Debug: first N features.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or Path(f"{Path(args.sae_folder).name}_sample_area_viz.html")

    # Checkpoint order from the memmap; eval folders strip the prefix.
    _, _, n_outputs, model_names = _load_index(args.cache_dir)
    eval_ckpts = [m[len(args.column_prefix):] if m.startswith(args.column_prefix)
                  else m for m in model_names]
    print(f"{len(model_names)} checkpoints (n_outputs={n_outputs})")

    tasks = args.tasks or discover_tasks(args.eval_dir)
    task_z, task_perf_raw, task_metric = {}, {}, {}
    for t in tasks:
        perf, metric = load_task_curve(t, args.eval_dir, eval_ckpts)
        tz = znorm(perf)
        if tz is None:
            print(f"  [skip] {t}: no usable curve")
            continue
        task_z[t] = tz
        task_perf_raw[t] = perf
        task_metric[t] = metric
        print(f"  {t}: metric={metric}, {len(tz[0])}/{len(perf)} valid")
    if not task_z:
        raise SystemExit("No usable task curves.")
    task_names = sorted(task_z)

    # feature -> sample word_ids + activations
    csv_path = os.path.join(args.sae_folder, "top-50_activations.csv")
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=["feature", "word_id", "act_value"])
    df["feature"] = df["feature"].astype(str)
    ctx = json.load(open(os.path.join(args.sae_folder, "top-50_words_in_context.json")))

    # feature descriptions (LLM labels)
    label_files = glob.glob(os.path.join(args.sae_folder,
                                         "feature_labels_validated", "*.json"))
    labels = json.load(open(label_files[0])) if label_files else {}
    desc_of = {str(k): v.get("Description", "") for k, v in labels.items()}
    if label_files:
        print(f"Loaded {len(desc_of)} feature descriptions from "
              f"{os.path.basename(label_files[0])}")

    fids = list(dict.fromkeys(df["feature"]))
    if args.limit:
        fids = fids[:args.limit]
    grouped = dict(list(df.groupby("feature")))

    # Single batched memmap read for every sample we'll need (the slow part).
    all_wids = [w for fid in fids for w in grouped[fid]["word_id"]]
    print(f"batch-reading sample curves from memmap "
          f"({len(set(map(str, all_wids)))} unique) ...")
    curves = batch_load_curves(all_wids, args.cache_dir)
    print(f"  resolved {len(curves)} curves")
    print(f"{len(fids)} features; scoring ({'diff' if args.diff else 'raw'} z-curves)...")

    # Per feature: build z-curve matrix S, raw median, sample metadata, scores.
    feats = {}
    for i, fid in enumerate(fids, 1):
        sub = grouped[fid]
        S_rows, smeta = [], []
        for wid, act in zip(sub["word_id"].astype(str), sub["act_value"]):
            raw = curves.get(wid)
            if raw is None:
                continue
            sz = z_full(raw)
            if sz is None:
                continue
            S_rows.append(sz)
            c = ctx.get(wid, {})
            smeta.append({"word_id": wid, "act": float(act) if pd.notna(act) else None,
                          "before": c.get("before", ""), "word": c.get("word", ""),
                          "after": c.get("after", "")})
        if not S_rows:
            continue
        S = np.vstack(S_rows)
        scores = {t: score_against_task(S, task_z[t], model_names, args.diff)
                  for t in task_names}
        feats[fid] = {
            "sample_z": S,          # (n_samples, n_ckpts) unit-var sample curves
            "samples": smeta,
            "scores": scores,
        }
        if i % 500 == 0:
            print(f"  {i}/{len(fids)}")
    print(f"Scored {len(feats)} features")

    # Full-length z-normalized task curves for plotting (None where missing).
    def task_z_full(t):
        arr = [None] * len(model_names)
        for i, v in zip(*task_z[t]):
            arr[i] = float(v)
        return arr
    task_z_plot = {t: task_z_full(t) for t in task_names}

    # When --diff, plot the differenced+renormalized curves (what the metric
    # compares) so jump alignment is what you actually see; else plot levels.
    def to_plot_curve(level_vals):
        if not args.diff:
            return list(level_vals)
        arr = np.array([np.nan if v is None else v for v in level_vals], float)
        d = np.diff(arr)
        dz = z_full(d) if not np.any(np.isnan(d)) else None
        d = dz if dz is not None else d
        return [None if (v != v) else float(v) for v in d]

    plot_ckpts = model_names[1:] if args.diff else model_names
    task_plot = {t: to_plot_curve(task_z_plot[t]) for t in task_names}
    ylab = "Δ z-score" if args.diff else "z-score"

    # Render: per task, two columns ranked by the chosen metric.
    plot_specs, sections, pid = {}, [], [0]

    def make_pid():
        pid[0] += 1
        return f"p{pid[0]}"

    for t in task_names:
        def column(sign, color, label):
            key = f"{args.rank_by}_{sign}"
            ranked = sorted(
                ((fid, fd["scores"][t][key]) for fid, fd in feats.items()
                 if fd["scores"][t][key] == fd["scores"][t][key]
                 and desc_of.get(fid, "").strip()),
                key=lambda x: x[1])[:args.top_n]
            blocks = []
            for fid, score in ranked:
                fd = feats[fid]
                sc = fd["scores"][t]
                # rank this feature's samples by their own match to the task
                ps = sc["per_sample_pos"] if sign == "pos" else sc["per_sample_neg"]
                order = [j for j in np.argsort(ps) if ps[j] == ps[j]]
                shown = []
                for j in order[:args.max_samples]:
                    s = dict(fd["samples"][j])
                    s["dist"] = float(ps[j])
                    shown.append(s)
                p = make_pid()
                # overlay the top samples' own z-curves (Δz if --diff), flipped
                # for the negative column, with the task curve
                sample_curves = []
                for j in order[:args.plot_samples]:
                    pc = to_plot_curve(fd["sample_z"][j].tolist())
                    if sign == "neg":
                        pc = [None if v is None else -v for v in pc]
                    sample_curves.append(pc)
                plot_specs[p] = make_samples_plot_spec(
                    plot_ckpts, sample_curves, task_plot[t],
                    f"{t} perf", color, ylab)
                metrics = (
                    f"<span class='metric'>best_sample={sc['best_of_area_'+sign]:.4f}</span>"
                    f"<span class='metric'>area_of_med={sc['area_of_med_'+sign]:.4f}</span>"
                    f"<span class='metric'>med_of_area={sc['med_of_area_'+sign]:.4f}</span>"
                    f"<span class='metric'>n={len(fd['samples'])}</span>")
                blocks.append(render_feature_block(
                    fid, desc_of.get(fid, ""), metrics, p, render_samples_table(shown)))
            if not blocks:
                blocks.append("<div class='empty'>No features.</div>")
            return (f"<div class='col'><h4>{label} <small>({len(ranked)})</small></h4>"
                    f"{''.join(blocks)}</div>")

        header = (f"<h3>{esc(t)} <small>metric={esc(task_metric[t])}</small></h3>"
                  f"<p class='counts'>ranked by {esc(args.rank_by)} "
                  f"({'differenced' if args.diff else 'raw'} z-curves) · "
                  f"showing top {args.top_n} each</p>")
        sections.append(
            f"<section class='task-row'>{header}<div class='feat-cols'>"
            f"{column('pos', '#1976d2', 'Positive (curve-aligned)')}"
            f"{column('neg', '#c62828', 'Negative (anti-aligned)')}"
            f"</div></section>")

    summary = (f"{len(task_names)} tasks · {len(feats)} features · "
               f"rank-by {args.rank_by} · top-{args.top_n} per side")
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Per-task features — sample area — {esc(Path(args.sae_folder).name)}</title>
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>{CSS}</style></head><body>
<header><h1>Per-task features (per-sample area)</h1>
<div class='meta'>SAE: <code>{esc(Path(args.sae_folder).name)}</code> · {summary}</div></header>
<div class='wrap'>{''.join(sections)}</div>
<script>const PLOTS = {json.dumps(plot_specs)};{JS}</script>
</body></html>"""
    out_path.write_text(doc)
    print(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
