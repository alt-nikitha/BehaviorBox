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
SAE_FOLDER = "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/n_only_early_and_late_olmo3_seed=42_ofw=0.5_psign_subset=sae_sample_by_tau_alltasks_t0.7_n1856450_N=3000_k=25_lp=None"
CACHE_DIR = "/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen/ofw=0.5_pznorm=0.01"

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


def build_families(task_z, threshold, metric="euclid"):
    """Group task curves into families by curve SHAPE, then return each family's SHARED
    curve. Tasks with the same trajectory shape can't earn separate features, so the
    family — not the task — is the natural unit to rank features against.

    metric:
      'euclid' (default) — distance = mean |Δz| per checkpoint between z-curves. Respects
        WHEN the rise happens, so early-risers and late-risers land in different families.
        This is the shape-correct default; cut at `threshold` (raw mean|Δz|, e.g. 0.6).
      'jump' — distance between unit-normalized DIFFERENCE (jump-timing) profiles; groups
        by where the biggest gains occur. Cut at `threshold` (e.g. 0.9).
      'tau' — 1 - Kendall tau of the orderings. WARNING: tau only checks rank agreement,
        and since nearly every task curve is monotone-ish it saturates and OVER-MERGES
        early- and late-risers into one family. Kept for comparison; `threshold` is the
        min within-family tau (e.g. 0.75) -> cut distance 1-threshold.

    Family curve = mean of members' z-curves, re-z-normed. Returns (fam_z, fam_members),
    families ordered by size desc, keyed 'family1', 'family2', ... All surviving task
    curves span the full checkpoint set, so members share one idx set."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform, pdist
    names = sorted(task_z)
    n = len(names)
    Z = np.vstack([np.asarray(task_z[t][1], float) for t in names])
    if metric == "tau":
        tau = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                tau[i, j] = tau[j, i] = _kendall(Z[i], Z[j])
        D = np.clip(1 - tau, 0, None)
        cut = 1 - threshold
    elif metric == "jump":
        Zd = np.diff(Z, axis=1)
        Zd = Zd / (np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9)
        D = squareform(pdist(Zd, metric="euclidean"))
        cut = threshold
    else:  # 'euclid' — mean |Δz| per checkpoint
        D = squareform(pdist(Z, metric="euclidean")) / np.sqrt(Z.shape[1])
        cut = threshold
    np.fill_diagonal(D, 0)
    if n == 1:
        labels = np.array([1])
    else:
        L = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(L, t=cut, criterion="distance")
    groups = {}
    for t, c in zip(names, labels):
        groups.setdefault(c, []).append(t)
    fam_z, fam_members = {}, {}
    for rank, c in enumerate(sorted(groups, key=lambda c: -len(groups[c])), 1):
        members = groups[c]
        idxs = task_z[members[0]][0]
        z = z_full(np.mean([task_z[m][1] for m in members], axis=0))
        if z is None:
            continue
        key = f"family{rank}"
        fam_z[key] = (list(idxs), z)
        fam_members[key] = members
    return fam_z, fam_members


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


def _pearson(a, b):
    """Pearson correlation; NaN if either input is (near-)constant."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def _pearson_rows(rows, tv):
    """Pearson of each row of `rows` (n,k) with `tv` (k,). Returns (n,); NaN for
    (near-)constant rows."""
    r = np.asarray(rows, float); t = np.asarray(tv, float)
    r = r - r.mean(axis=1, keepdims=True); t = t - t.mean()
    num = (r * t).sum(axis=1)
    den = np.sqrt((r * r).sum(axis=1)) * np.sqrt((t * t).sum())
    out = np.full(r.shape[0], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def _kendall(a, b):
    """Kendall tau-b between two curves via pairwise sign agreement. Depends only on
    the ranking of the checkpoints, so it is invariant to the per-curve z-norm and,
    unlike Pearson, does not saturate near 1 on short monotone-ish trajectories. NaN
    if either side has no resolvable (non-tied) pairs."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    k = len(a)
    if k < 2:
        return float("nan")
    iu, ju = np.triu_indices(k, k=1)
    sa = np.sign(a[ju] - a[iu]); sb = np.sign(b[ju] - b[iu])
    nx = (sa != 0).sum(); ny = (sb != 0).sum()
    if nx == 0 or ny == 0:
        return float("nan")
    return float((sa * sb).sum() / np.sqrt(nx * ny))


def _kendall_rows(rows, tv):
    """Kendall tau-b of each row of `rows` (n,k) with `tv` (k,). Returns (n,); NaN for
    rows with no resolvable pairs."""
    r = np.asarray(rows, float); t = np.asarray(tv, float)
    k = r.shape[1]
    if k < 2:
        return np.full(r.shape[0], np.nan)
    iu, ju = np.triu_indices(k, k=1)
    sr = np.sign(r[:, ju] - r[:, iu])            # (n, npairs)
    st = np.sign(t[ju] - t[iu])                  # (npairs,)
    ny = (st != 0).sum()
    nx = (sr != 0).sum(axis=1).astype(float)     # (n,)
    out = np.full(r.shape[0], np.nan)
    ok = (nx > 0) & (ny > 0)
    out[ok] = (sr[ok] @ st) / np.sqrt(nx[ok] * ny)
    return out


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


def score_against_task(S, task_z, ckpt_names, diff, acts=None, act_top_k=0):
    """S: (n_samples, n_ckpts) z-curves. Returns dict of the four scalars +
    per-sample positive areas (for sorting samples in the view).

    If act_top_k>0 and acts given, restrict to the act_top_k strongest-activating
    samples BEFORE scoring, so the median-based metrics (area_of_med, med_of_area)
    reflect the feature's canonical members instead of being diluted by the weakly-
    activating tail of the top-50."""
    if act_top_k and acts is not None and 0 < act_top_k < S.shape[0]:
        keep = np.argsort(-np.asarray(acts, dtype=float))[:act_top_k]
        S = S[keep]
    idxs, tvals = task_z
    if not idxs or max(idxs) >= S.shape[1]:
        nan = float("nan")
        return {"med_of_area_pos": nan, "med_of_area_neg": nan,
                "area_of_med_pos": nan, "area_of_med_neg": nan,
                "best_of_area_pos": nan, "best_of_area_neg": nan,
                "pearson_pos": nan, "pearson_neg": nan,
                "diff_pearson_pos": nan, "diff_pearson_neg": nan,
                "med_of_pearson_pos": nan, "med_of_pearson_neg": nan,
                "med_of_diff_pearson_pos": nan, "med_of_diff_pearson_neg": nan,
                "best_of_pearson_pos": nan, "best_of_pearson_neg": nan,
                "best_of_diff_pearson_pos": nan, "best_of_diff_pearson_neg": nan,
                "kendall_pos": nan, "kendall_neg": nan,
                "med_of_kendall_pos": nan, "med_of_kendall_neg": nan,
                "best_of_kendall_pos": nan, "best_of_kendall_neg": nan,
                "per_sample_pos": np.full(S.shape[0], nan),
                "per_sample_neg": np.full(S.shape[0], nan)}
    sub = S[:, np.array(idxs, dtype=int)]
    tv = np.asarray(tvals, dtype=float)

    # Pearson-correlation metrics on the feature's median curve vs the task curve
    # (computed on the RAW z-curves, independent of --diff). Stored signed: the pos
    # column ranks by highest corr, the neg column by most anti-correlated. diff_pearson
    # correlates the consecutive-difference (jump-shape) curves instead of the levels.
    med_raw = np.median(sub, axis=0)
    pear = _pearson(med_raw, tv)
    dpear = _pearson(np.diff(med_raw), np.diff(tv)) if len(tv) >= 3 else float("nan")
    # Per-sample correlations -> med_of_pearson (median = consistency) and
    # best_of_pearson (max = the single best-correlated sample, corr analog of
    # best_of_area). neg column uses the most anti-correlated sample.
    _psr = _pearson_rows(sub, tv)
    _has = bool(np.isfinite(_psr).any())
    med_pear = float(np.nanmedian(_psr)) if _has else float("nan")
    best_pear_pos = float(np.nanmax(_psr)) if _has else float("nan")
    best_pear_neg = float(-np.nanmin(_psr)) if _has else float("nan")
    if len(tv) >= 3:
        _psrd = _pearson_rows(np.diff(sub, axis=1), np.diff(tv))
        _hasd = bool(np.isfinite(_psrd).any())
        med_dpear = float(np.nanmedian(_psrd)) if _hasd else float("nan")
        best_dpear_pos = float(np.nanmax(_psrd)) if _hasd else float("nan")
        best_dpear_neg = float(-np.nanmin(_psrd)) if _hasd else float("nan")
    else:
        med_dpear = best_dpear_pos = best_dpear_neg = float("nan")

    # Kendall tau (rank-agreement) analogs of the pearson metrics, on the RAW z-curves.
    # kendall: tau of the feature's median curve vs the task curve. med_of_kendall:
    # median over samples of each sample's own tau (consistency). best_of_kendall: the
    # single best-matching sample. Rank-based, so no saturation on short monotone curves
    # and consistent with the SAE's pairwise-sign grouping.
    kend = _kendall(med_raw, tv)
    _ksr = _kendall_rows(sub, tv)
    _hask = bool(np.isfinite(_ksr).any())
    med_kend = float(np.nanmedian(_ksr)) if _hask else float("nan")
    best_kend_pos = float(np.nanmax(_ksr)) if _hask else float("nan")
    best_kend_neg = float(-np.nanmin(_ksr)) if _hask else float("nan")

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
        # correlation of the median curve with the task curve (signed; higher=better)
        "pearson_pos": pear, "pearson_neg": -pear,
        "diff_pearson_pos": dpear, "diff_pearson_neg": -dpear,
        "med_of_pearson_pos": med_pear, "med_of_pearson_neg": -med_pear,
        "med_of_diff_pearson_pos": med_dpear, "med_of_diff_pearson_neg": -med_dpear,
        "best_of_pearson_pos": best_pear_pos, "best_of_pearson_neg": best_pear_neg,
        "best_of_diff_pearson_pos": best_dpear_pos,
        "best_of_diff_pearson_neg": best_dpear_neg,
        # Kendall tau (rank agreement); signed, higher=better
        "kendall_pos": kend, "kendall_neg": -kend,
        "med_of_kendall_pos": med_kend, "med_of_kendall_neg": -med_kend,
        "best_of_kendall_pos": best_kend_pos, "best_of_kendall_neg": best_kend_neg,
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
    ap.add_argument("--family-tau", type=float, default=None,
                    help="If set, turn on FAMILY mode: group tasks by curve shape and rank "
                         "features against each family's SHARED curve instead of per task. "
                         "Value is the clustering cut threshold, interpreted per "
                         "--family-metric (euclid: max mean|Δz|, e.g. 0.6; tau: min "
                         "within-family tau, e.g. 0.75; jump: max profile dist, e.g. 0.9).")
    ap.add_argument("--family-metric", default="euclid",
                    choices=["euclid", "jump", "tau"],
                    help="Distance for grouping task curves into families. "
                         "euclid (default): mean |Δz| per checkpoint — sees WHEN the rise "
                         "happens, so early- and late-risers separate (the shape-correct "
                         "choice). jump: unit difference-profile distance (groups by where "
                         "gains occur). tau: 1-Kendall — WARNING, saturates on monotone "
                         "curves and OVER-MERGES early+late risers; comparison only.")
    ap.add_argument("--contrast", action="store_true", default=False,
                    help="Family mode only: rank each family by DISCRIMINATIVE fit — "
                         "score(feat,fam_k) minus the best score against any other family "
                         "— so shape-generic features (that match every family) drop out "
                         "and only family-specific shapes survive. Needs a higher=better "
                         "--rank-by (kendall/pearson variants).")
    ap.add_argument("--rank-by", default="best_of_area",
                    choices=["best_of_area", "area_of_med", "med_of_area",
                             "pearson", "diff_pearson",
                             "med_of_pearson", "med_of_diff_pearson",
                             "best_of_pearson", "best_of_diff_pearson",
                             "kendall", "med_of_kendall", "best_of_kendall"],
                    help="Which metric ranks features per task. area/best: lower=better. "
                         "pearson: corr of the feature's MEDIAN curve with the task curve. "
                         "med_of_pearson: median over samples of each sample's own corr "
                         "with the task (consistency). diff_ variants use jump-shape "
                         "(consecutive-difference) curves. corr metrics: higher=better. "
                         "kendall/med_of_kendall/best_of_kendall: rank-agreement (tau-b) "
                         "analogs of the pearson trio; rank-based so no saturation on "
                         "short monotone curves, and consistent with the pairwise-sign "
                         "SAE grouping. higher=better.")
    ap.add_argument("--diff", action="store_true", default=False,
                    help="First-difference z-curves before integrating.")
    ap.add_argument("--show-negative", action="store_true", default=False,
                    help="Also render the anti-aligned (negative) column. Off by "
                         "default — only the curve-aligned features are shown.")
    ap.add_argument("--act-top-k", type=int, default=0,
                    help="If >0, score each feature using only its N strongest-activating "
                         "samples (median-based metrics reflect canonical members, not the "
                         "weakly-activating tail). 0 = use all top-50.")
    ap.add_argument("--min-samples", type=int, default=50,
                    help="Drop features with fewer than this many activating samples "
                         "(rare/weak features). Default 50; set 0 to keep all.")
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
        # Require every checkpoint valid; drop the task otherwise.
        n_valid, n_total = len(tz[0]), len(perf)
        if n_valid < n_total:
            print(f"  [skip] {t}: only {n_valid}/{n_total} valid")
            continue
        task_z[t] = tz
        task_perf_raw[t] = perf
        task_metric[t] = metric
        print(f"  {t}: metric={metric}, {len(tz[0])}/{len(perf)} valid")
    if not task_z:
        raise SystemExit("No usable task curves.")
    task_names = sorted(task_z)

    # Family mode: collapse tasks into shape-families and rank features against the
    # family's shared curve. Downstream code treats each family as a virtual "task".
    if args.family_tau is not None:
        fam_z, fam_members = build_families(task_z, args.family_tau, args.family_metric)
        task_z = fam_z
        task_metric = {k: " + ".join(m) for k, m in fam_members.items()}
        task_perf_raw = {}
        task_names = list(fam_z)   # already size-ordered (family1 = largest)
        print(f"\nGrouped {sum(len(m) for m in fam_members.values())} tasks into "
              f"{len(task_names)} families (metric={args.family_metric}, "
              f"thr={args.family_tau}):")
        for k in task_names:
            print(f"  {k}: {task_metric[k]}")

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
    n_dropped_small = 0
    for i, fid in enumerate(fids, 1):
        sub = grouped[fid]
        if args.min_samples and len(sub) < args.min_samples:
            n_dropped_small += 1
            continue
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
        acts = np.array([m["act"] if m["act"] is not None else 0.0 for m in smeta])
        scores = {t: score_against_task(S, task_z[t], model_names, args.diff,
                                        acts=acts, act_top_k=args.act_top_k)
                  for t in task_names}
        feats[fid] = {
            "sample_z": S,          # (n_samples, n_ckpts) unit-var sample curves
            "samples": smeta,
            "scores": scores,
        }
        if i % 500 == 0:
            print(f"  {i}/{len(fids)}")
    print(f"Scored {len(feats)} features"
          + (f" (dropped {n_dropped_small} with <{args.min_samples} samples)"
             if args.min_samples else ""))

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
            # correlation metrics: higher is better -> sort descending; area/best: ascending
            higher_better = args.rank_by in ("pearson", "diff_pearson",
                                             "med_of_pearson", "med_of_diff_pearson",
                                             "best_of_pearson",
                                             "best_of_diff_pearson",
                                             "kendall", "med_of_kendall",
                                             "best_of_kendall")
            mult = -1.0 if higher_better else 1.0
            contrast_of = {}
            if args.contrast and args.family_tau is not None and len(task_names) > 1:
                # discriminative fit: this family's score minus the best score the same
                # feature earns against ANY other family. Shape-generic features match
                # every family, so their contrast collapses toward 0 and they drop out;
                # a family-specific shape stays high. Uses the same key as --rank-by.
                cand = []
                for fid, fd in feats.items():
                    if not desc_of.get(fid, "").strip():
                        continue
                    v = fd["scores"][t][key]
                    if v != v:
                        continue
                    others = [fd["scores"][tt][key] for tt in task_names if tt != t]
                    others = [o for o in others if o == o]
                    if not others:
                        c = v
                    else:
                        c = (v - max(others)) if higher_better else (min(others) - v)
                    contrast_of[fid] = c
                    cand.append((fid, v, c))
                ranked = [(fid, v) for fid, v, _ in
                          sorted(cand, key=lambda x: -x[2])[:args.top_n]]
            else:
                ranked = sorted(
                    ((fid, fd["scores"][t][key]) for fid, fd in feats.items()
                     if fd["scores"][t][key] == fd["scores"][t][key]
                     and desc_of.get(fid, "").strip()),
                    key=lambda x: mult * x[1])[:args.top_n]
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
                    f"<span class='metric'>pearson={sc['pearson_'+sign]:+.3f}</span>"
                    f"<span class='metric'>Δpearson={sc['diff_pearson_'+sign]:+.3f}</span>"
                    f"<span class='metric'>medP={sc['med_of_pearson_'+sign]:+.3f}</span>"
                    f"<span class='metric'>ΔmedP={sc['med_of_diff_pearson_'+sign]:+.3f}</span>"
                    f"<span class='metric'>τ={sc['kendall_'+sign]:+.3f}</span>"
                    f"<span class='metric'>medτ={sc['med_of_kendall_'+sign]:+.3f}</span>"
                    f"<span class='metric'>bestτ={sc['best_of_kendall_'+sign]:+.3f}</span>"
                    + (f"<span class='metric'>contrast={contrast_of[fid]:+.3f}</span>"
                       if fid in contrast_of else "")
                    + f"<span class='metric'>n={len(fd['samples'])}</span>")
                blocks.append(render_feature_block(
                    fid, desc_of.get(fid, ""), metrics, p, render_samples_table(shown)))
            if not blocks:
                blocks.append("<div class='empty'>No features.</div>")
            return (f"<div class='col'><h4>{label} <small>({len(ranked)})</small></h4>"
                    f"{''.join(blocks)}</div>")

        _lbl = "members" if args.family_tau is not None else "metric"
        header = (f"<h3>{esc(t)} <small>{_lbl}={esc(task_metric[t])}</small></h3>"
                  f"<p class='counts'>ranked by {esc(args.rank_by)} "
                  f"({'differenced' if args.diff else 'raw'} z-curves) · "
                  f"showing top {args.top_n} each</p>")
        cols = column('pos', '#1976d2', 'Positive (curve-aligned)')
        if args.show_negative:
            cols += column('neg', '#c62828', 'Negative (anti-aligned)')
        sections.append(
            f"<section class='task-row'>{header}<div class='feat-cols'>{cols}</div></section>")

    summary = (f"{len(task_names)} tasks · {len(feats)} features · "
               f"rank-by {args.rank_by} · top-{args.top_n}"
               f"{' per side' if args.show_negative else ' (positive only)'}")
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
