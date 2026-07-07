"""Emergence-timeline visualization: group tasks by *when* they jump most, and for
each task list the SAE features that jump at the same checkpoint, ranked by z-curve
shape-fit to that task's performance curve.

Unlike plot_task_features_sample_area.py (which ranks every feature by full-curve area
against every task), this:
  1. locates each task's dominant jump from the RAW performance curve (argmax of
     consecutive gains) -> avoids z-norming the flat-at-chance region;
  2. groups tasks by that jump checkpoint (the emergence timeline);
  3. for each task, restricts to features whose own dominant step-up is at the same
     checkpoint, and sorts them by mean|Δz| distance between the feature's z-curve and
     that task's z-curve (lower = better shape fit).

Self-contained HTML with inline SVG curve overlays (task curve bold + feature curve),
no Plotly/CDN. Reads only the SAE folder's top-50_activations.csv, top-50_words_in_
context.json, feature_labels/*.json, and the eval_results task curves.

Usage:
    python plot_task_jump_features.py                 # defaults below
    python plot_task_jump_features.py --top-n 12 --jump-thresh 0.6
"""
import argparse, glob, html, json, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr


def fit_score(fzv, tzv, metric):
    """Feature-vs-task shape fit over shared valid checkpoints.
    Returns (value, label, higher_is_better)."""
    if metric == "distance":
        f = (fzv - fzv.mean()) / (fzv.std() + 1e-9)
        t = (tzv - tzv.mean()) / (tzv.std() + 1e-9)
        return float(np.mean(np.abs(f - t))), "d", False
    if metric == "pearson":
        if fzv.std() < 1e-9 or tzv.std() < 1e-9: return -2.0, "r", True
        return float(pearsonr(fzv, tzv)[0]), "r", True
    if fzv.std() < 1e-9 or tzv.std() < 1e-9: return -2.0, "ρ", True
    return float(spearmanr(fzv, tzv).correlation), "ρ", True

SAE = "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/n_only_early_and_late_olmo3_seed=42_ofw=0.5_N=3000_k=25_lp=None_pznorm=0.01"
EVAL = "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3"
STEPS = [1000, 2000, 4000, 8000, 16000, 32000, 68000, 103000, 154000, 205000, 256000]
COLS = [f"olmo3-stage1-step{s}_zscore" for s in STEPS]
METRIC_PREFERENCE = ["acc_norm,none", "acc,none", "exact_match,strict-match",
    "exact_match,get-answer", "exact_match,flexible-extract", "exact_match,remove_whitespace",
    "exact,none", "em,none", "f1,none", "perplexity,none"]
esc = lambda s: html.escape(str(s) if s is not None else "")
klbl = lambda i: f"{STEPS[i] // 1000}k"


def load_task_curve(task, eval_dir, ckpt_names):
    perf, metric, key = [], None, None
    for cp in ckpt_names:
        md = glob.glob(os.path.join(eval_dir, cp, task, "*/"))
        rf = glob.glob(os.path.join(md[0], "results_*.json")) if md else []
        if not rf:
            perf.append(None); continue
        res = json.load(open(rf[0])).get("results", {})
        if key is None: key = task if task in res else (next(iter(res)) if res else None)
        if metric is None and key in res:
            for m in METRIC_PREFERENCE:
                if m in res[key]: metric = m; break
            else:
                metric = next((k for k in res[key] if "stderr" not in k and k != "alias"), None)
        v = res.get(key, {}).get(metric)
        perf.append(float(v) if v is not None else None)
    return perf, metric


def znorm_align(perf):
    """Z-norm valid entries, scatter back to a length-11 vector aligned to STEPS (NaN elsewhere)."""
    valid = [(i, v) for i, v in enumerate(perf) if v is not None]
    if len(valid) < 3: return None
    vals = np.array([v for _, v in valid], float)
    if vals.std() < 1e-9: return None
    z = (vals - vals.mean()) / vals.std()
    out = np.full(11, np.nan)
    for (i, _), zz in zip(valid, z): out[i] = zz
    return out


def svg_overlay(task_z, feat_z, jump_i, w=190, h=54, color="#c44"):
    """Two z-curves on shared mini-axes; dashed vline at the jump boundary."""
    pad = 6
    both = np.concatenate([task_z[~np.isnan(task_z)], feat_z[~np.isnan(feat_z)]])
    lo, hi = float(np.min(both)), float(np.max(both))
    rng = (hi - lo) or 1.0
    xs = [pad + (w - 2 * pad) * i / (len(STEPS) - 1) for i in range(len(STEPS))]
    def pts(y):
        out = []
        for i, v in enumerate(y):
            if np.isnan(v): continue
            yy = h - pad - (h - 2 * pad) * (v - lo) / rng
            out.append(f"{xs[i]:.1f},{yy:.1f}")
        return " ".join(out)
    y0 = h - pad - (h - 2 * pad) * (0 - lo) / rng          # zero line
    vx = xs[jump_i]                                         # jump boundary (arrival ckpt)
    return (f"<svg width='{w}' height='{h}' class='spark'>"
            f"<line x1='{pad}' y1='{y0:.1f}' x2='{w-pad}' y2='{y0:.1f}' class='zero'/>"
            f"<line x1='{vx:.1f}' y1='{pad}' x2='{vx:.1f}' y2='{h-pad}' class='vline'/>"
            f"<polyline points='{pts(task_z)}' class='taskline'/>"
            f"<polyline points='{pts(feat_z)}' style='stroke:{color}' class='featline'/>"
            f"</svg>")


def svg_raw(perf, jump_i, w=150, h=40):
    pad = 5
    v = np.array([np.nan if p is None else p for p in perf], float)
    lo, hi = np.nanmin(v), np.nanmax(v); rng = (hi - lo) or 1.0
    xs = [pad + (w - 2 * pad) * i / (len(v) - 1) for i in range(len(v))]
    pts = " ".join(f"{xs[i]:.1f},{h-pad-(h-2*pad)*(v[i]-lo)/rng:.1f}"
                   for i in range(len(v)) if not np.isnan(v[i]))
    vx = xs[jump_i]
    return (f"<svg width='{w}' height='{h}' class='spark'>"
            f"<line x1='{vx:.1f}' y1='{pad}' x2='{vx:.1f}' y2='{h-pad}' class='vline'/>"
            f"<polyline points='{pts}' class='taskline'/></svg>")


CSS = """
*{box-sizing:border-box} body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
margin:0;color:#222;background:#fafafa} header{padding:16px 24px;background:#fff;border-bottom:1px solid #ddd}
h1{margin:0 0 4px;font-size:18px} header .meta{color:#666;font-size:12px;max-width:900px}
.wrap{padding:20px 24px} .epoch{margin-bottom:26px}
.epoch>h2{font-size:14px;margin:0 0 10px;color:#556;border-bottom:2px solid #dde;padding-bottom:4px}
.task{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 14px;margin-bottom:12px}
.task .thead{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:8px}
.task h3{margin:0;font-size:14px} .task .tmeta{color:#888;font-size:11px;font-family:monospace}
table.f{width:100%;border-collapse:collapse;font-size:12px}
table.f td{padding:4px 8px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
td.rk{color:#aaa;font-family:monospace;width:22px} td.num{font-family:monospace;text-align:right;color:#556;white-space:nowrap}
td.fid{font-family:monospace;font-weight:600;color:#335;white-space:nowrap}
td.desc{color:#444} td.sp{width:200px}
details.w{margin-top:2px} details.w summary{cursor:pointer;color:#88a;font-size:11px}
table.ctx{width:100%;border-collapse:collapse;font-size:11px;margin-top:3px}
table.ctx td{padding:1px 5px;border-bottom:1px solid #f6f6f6}
td.bf{text-align:right;color:#999;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
td.wd{font-weight:700;white-space:nowrap} td.af{color:#999;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
svg.spark{background:#fff;border:1px solid #eee;border-radius:3px}
svg.spark .taskline{fill:none;stroke:#111;stroke-width:2}
svg.spark .featline{fill:none;stroke-width:1.5;opacity:.85}
svg.spark .zero{stroke:#ddd;stroke-width:1} svg.spark .vline{stroke:#e0b050;stroke-width:1;stroke-dasharray:2,2}
.legend{font-size:11px;color:#777;margin:2px 0 14px}.legend b{color:#111}.legend i{color:#c44;font-style:normal}
"""

COLORS = ["#c44", "#4a7", "#47c", "#a5a", "#c82", "#2aa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default=SAE)
    ap.add_argument("--eval", default=EVAL)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--metric", default="spearman", choices=["spearman", "pearson", "distance"],
                    help="Feature ranking within each task's jump group.")
    ap.add_argument("--jump-thresh", type=float, default=0.6)
    ap.add_argument("--n-words", type=int, default=6)
    ap.add_argument("--out", default="/home/nsrikant/BehaviorBoxNew/analysis/task_jump_features_viz.html")
    a = ap.parse_args()

    ckdirs = [os.path.basename(p) for p in glob.glob(a.eval + "/*") if os.path.isdir(p)]
    stepof = lambda n: int(re.search(r"step(\d+)", n).group(1))
    ckpt = [d for _, d in sorted((stepof(d), d) for d in ckdirs)]
    es = [stepof(d) for d in ckpt]

    # tasks: dominant jump + z-curve
    tasks = sorted(os.listdir(a.eval + "/" + ckpt[0]))
    TZ, RAW, JB, JINFO, MET = {}, {}, {}, {}, {}
    for t in tasks:
        perf, metric = load_task_curve(t, a.eval, ckpt)
        v = np.array([np.nan if p is None else p for p in perf], float)
        df = np.diff(v); df[np.isnan(df)] = -9
        loc = int(np.nanargmax(df))
        z = znorm_align([None if p is None else p for i, p in enumerate(perf)])
        # re-align raw perf & z-curve to STEPS positions
        rawA = np.full(11, np.nan); zA = np.full(11, np.nan)
        for i, s in enumerate(es):
            if s in STEPS and perf[i] is not None: rawA[STEPS.index(s)] = perf[i]
        if z is not None:
            for i, s in enumerate(es):
                if s in STEPS: zA[STEPS.index(s)] = z[i]
        if np.all(np.isnan(zA)): continue
        rng = np.nanmax(v) - np.nanmin(v)
        arrival_step = es[loc + 1]; jb = STEPS.index(arrival_step)
        TZ[t], RAW[t], JB[t], MET[t] = zA, rawA, jb, metric
        JINFO[t] = (df[loc], df[loc] / rng if rng > 0 else 0)

    # features
    d = pd.read_csv(a.sae + "/top-50_activations.csv", usecols=["feature", "word_id", "act_value"] + COLS)
    F = d.groupby("feature")[COLS].mean(); fid = F.index.values
    Fz = F.values.astype(float); Fz = (Fz - Fz.mean(1, keepdims=True)) / (Fz.std(1, keepdims=True) + 1e-9)
    fdiff = np.diff(Fz, axis=1); floc = fdiff.argmax(1); fmag = fdiff[np.arange(len(fid)), floc]
    fpos = {f: k for k, f in enumerate(fid)}
    # feature -> top word_ids
    top_words = (d.sort_values("act_value", ascending=False)
                   .groupby("feature")["word_id"].apply(lambda s: list(s)[:a.n_words]).to_dict())
    ctx = json.load(open(a.sae + "/top-50_words_in_context.json"))
    lab = json.load(open(a.sae + "/feature_labels/gemini-gemini-2.5-pro.json"))
    desc = lambda f: (lab.get(str(f), {}).get("Description", "?") if isinstance(lab.get(str(f)), dict) else "?")
    coh = lambda f: (isinstance(lab.get(str(f)), dict) and lab[str(f)].get("Coherent") == "YES")
    JUNK = re.compile(r"preceded by a space|begin(s|ning)? with a space|whitespace|structural tokens", re.I)

    groups = {}
    for t in TZ: groups.setdefault(JB[t], []).append(t)

    def words_html(f):
        rows = []
        for wid in top_words.get(f, []):
            c = ctx.get(str(wid))
            if not c: continue
            rows.append(f"<tr><td class='bf'>{esc(c.get('before',''))}</td>"
                        f"<td class='wd'>{esc(c.get('word',''))}</td>"
                        f"<td class='af'>{esc(c.get('after',''))}</td></tr>")
        if not rows: return ""
        return ("<details class='w'><summary>example words</summary>"
                f"<table class='ctx'>{''.join(rows)}</table></details>")

    body = []
    for gi, b in enumerate(sorted(groups)):
        body.append(f"<div class='epoch'><h2>Jump @ {klbl(b)} &nbsp;·&nbsp; "
                    f"{len(groups[b])} task(s)</h2>")
        for t in sorted(groups[b], key=lambda t: -JINFO[t][0]):
            tz = TZ[t]; ok = ~np.isnan(tz)
            fi = b - 1
            cand = [j for j in range(len(fid)) if floc[j] == fi and fmag[j] > a.jump_thresh
                    and coh(fid[j]) and not JUNK.search(desc(fid[j]))]
            scores, lbl_pfx, hib = [], "d", False
            for j in cand:
                v, lbl_pfx, hib = fit_score(Fz[j, ok], tz[ok], a.metric)
                scores.append(v)
            order = (np.argsort([-s for s in scores]) if hib else np.argsort(scores))[:a.top_n]
            mag, frac = JINFO[t]
            rows = ""
            for rk, r in enumerate(order, 1):
                j = cand[r]; f = fid[j]; col = COLORS[gi % len(COLORS)]
                fzc = np.full(11, np.nan); fzc[ok] = (Fz[j, ok] - Fz[j, ok].mean()) / (Fz[j, ok].std() + 1e-9)
                rows += (f"<tr><td class='rk'>{rk}</td>"
                         f"<td class='sp'>{svg_overlay(tz, fzc, b, color=col)}</td>"
                         f"<td class='num'>{lbl_pfx}={scores[r]:+.2f}</td>"
                         f"<td class='fid'>f{f}</td>"
                         f"<td class='desc'>{esc(desc(f))}{words_html(f)}</td></tr>")
            if not rows:
                rows = "<tr><td colspan='5' style='color:#999;font-style:italic'>no coherent features jump here</td></tr>"
            body.append(
                f"<div class='task'><div class='thead'>"
                f"<h3>{esc(t)}</h3>{svg_raw(RAW[t], b)}"
                f"<span class='tmeta'>{esc(MET[t])} · Δ=+{mag:.2f} ({frac*100:.0f}% of total gain) · "
                f"{len(cand)} candidate features</span></div>"
                f"<table class='f'>{rows}</table></div>")
        body.append("</div>")

    page = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Task jump features</title><style>{CSS}</style></head><body>"
            f"<header><h1>Emergence timeline — features that jump with each task</h1>"
            f"<div class='meta'>Tasks grouped by their dominant performance jump (from raw curves). "
            f"Per task: features whose own step-up is at that checkpoint, ranked by z-curve shape-fit "
            f"({esc(a.metric)} to the task curve). SAE: {esc(os.path.basename(a.sae))}</div></header>"
            f"<div class='wrap'><div class='legend'>In each sparkline: <b>bold black</b> = task z-curve, "
            f"<i>colored</i> = feature z-curve, dashed line = jump checkpoint.</div>"
            f"{''.join(body)}</div></body></html>")
    with open(a.out, "w") as f: f.write(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
