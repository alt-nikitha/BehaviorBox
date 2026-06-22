"""Quick look: for each task, the top token trajectories that are 'nearest' to
that task's performance curve, under two notions of nearest:

  * LEVEL  — distance between z-normalized curves (mean |diff|). Rewards tokens
             whose overall trajectory tracks the task's.
  * JUMP   — distance between the curves' consecutive differences (mean |diff|
             of diffs). Rewards tokens whose *jumps* land where the task's do.

For speed we sample a random subset of memmap rows rather than scanning all
tokens. Each task panel overlays the top-k token z-curves (faint) on the task
z-curve (bold), one sub-panel per case, so you can see whether 'nearest' tokens
actually follow the task shape or are just generic rises.

Usage:
    python centroid_token_trajectories_to_html.py --sample 300000 --top 40
"""

import argparse
import glob
import json
import os

import numpy as np

from sample_curves import _load_index, DEFAULT_CACHE

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STEM = "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm"
VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


def task_centroids(stem, n_outputs):
    cents, ckpts = {}, None
    for d in sorted(glob.glob(os.path.join(ANALYSIS_DIR, "precomputed_data_*"))):
        task = os.path.basename(d).replace("precomputed_data_", "")
        if task not in VALID_TASKS:
            continue
        fp = os.path.join(d, f"{stem}.json")
        if not os.path.exists(fp):
            continue
        j = json.load(open(fp))
        perf = [p for p in (j.get("overall_performance") or []) if p is not None]
        if len(perf) != n_outputs:
            continue
        x = np.array(perf, float)
        if x.std() < 1e-12:
            continue
        cents[task] = ((x - x.mean()) / x.std()).astype(np.float32)
        if ckpts is None:
            ckpts = j.get("checkpoints")
    return cents, ckpts


def zrows(b):
    mu = b.mean(1, keepdims=True)
    sd = b.std(1, keepdims=True)
    out = (b - mu) / sd
    out[sd[:, 0] < 1e-9] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--sample", type=int, default=300_000,
                    help="Random memmap rows to scan.")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR,
                    "centroid_token_trajectories.html"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mm, doc_offset, n_outputs, _ = _load_index(args.cache_dir)
    cents, ckpts = task_centroids(args.stem, n_outputs)
    if not cents:
        raise SystemExit("no task centroids matched n_outputs")
    tasks = sorted(cents)
    C = np.stack([cents[t] for t in tasks])          # (K, T)
    Cd = np.diff(C, axis=1)                            # (K, T-1)
    Cd = (Cd - Cd.mean(1, keepdims=True))             # center diffs for fairness
    print(f"rows={mm.shape[0]:,} T={n_outputs} tasks={len(tasks)}")

    rows = rng.choice(mm.shape[0], size=min(args.sample, mm.shape[0]),
                      replace=False)
    rows.sort()
    block = np.asarray(mm[rows][:, -n_outputs:], dtype=np.float32)
    Z = zrows(block)
    ok = ~np.isnan(Z).any(1)
    Z = Z[ok]
    rows = rows[ok]
    print(f"usable sampled tokens: {Z.shape[0]:,}")

    # level distances: (N,K)
    dlev = np.mean(np.abs(Z[:, None, :] - C[None, :, :]), axis=2)
    # jump distances: diffs of token curve vs centered diffs of task curve
    Zd = np.diff(Z, axis=1)
    Zd = Zd - Zd.mean(1, keepdims=True)
    djmp = np.mean(np.abs(Zd[:, None, :] - Cd[None, :, :]), axis=2)

    x = ckpts if ckpts else list(range(n_outputs))

    def panel(task_i, dmat, case):
        order = np.argsort(dmat[:, task_i])[:args.top]
        traces = []
        for r in order:
            traces.append({
                "x": x, "y": Z[r].tolist(), "type": "scatter", "mode": "lines",
                "line": {"color": "rgba(25,118,210,0.18)", "width": 1},
                "hoverinfo": "skip", "showlegend": False,
            })
        traces.append({
            "x": x, "y": C[task_i].tolist(), "type": "scatter",
            "mode": "lines+markers", "name": f"{tasks[task_i]} (task)",
            "line": {"color": "#c62828", "width": 3}, "marker": {"size": 5},
        })
        layout = {
            "height": 260, "margin": {"l": 40, "r": 12, "t": 28, "b": 60},
            "title": {"text": f"{case}", "font": {"size": 12}},
            "xaxis": {"tickangle": -45, "tickfont": {"size": 8}},
            "yaxis": {"title": "z", "titlefont": {"size": 9}},
            "showlegend": False, "hovermode": False,
        }
        return {"data": traces, "layout": layout}

    plot_specs, sections = {}, []
    for i, t in enumerate(tasks):
        pid_l, pid_j = f"{t}_lvl", f"{t}_jmp"
        plot_specs[pid_l] = panel(i, dlev, "nearest by LEVEL (z-curve)")
        plot_specs[pid_j] = panel(i, djmp, "nearest by JUMP (diffs)")
        sections.append(
            f"<section class='task'><h3>{t}</h3>"
            f"<div class='cols'>"
            f"<div class='plot' id='{pid_l}'></div>"
            f"<div class='plot' id='{pid_j}'></div></div></section>")

    css = """
    body{font-family:-apple-system,'Segoe UI',sans-serif;margin:0;background:#fafafa;color:#222}
    header{padding:14px 22px;background:#fff;border-bottom:1px solid #ddd}
    h1{font-size:17px;margin:0 0 4px}.meta{color:#666;font-size:12px}
    .task{background:#fff;border:1px solid #ddd;border-radius:6px;margin:16px 22px;padding:12px}
    .task h3{margin:0 0 6px;font-size:14px}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .plot{width:100%;height:260px;border:1px solid #eee;border-radius:3px}
    @media(max-width:1000px){.cols{grid-template-columns:1fr}}
    """
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Top token trajectories per task</title>
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>{css}</style></head><body>
<header><h1>Top token trajectories nearest each task curve</h1>
<div class='meta'>Dataset <code>{args.stem}</code> &middot; sampled {Z.shape[0]:,} tokens &middot;
top {args.top} per case &middot; red = task curve, blue = nearest token z-curves.
Left: nearest by overall level. Right: nearest by jump shape (diffs).</div></header>
{''.join(sections)}
<script>const P={json.dumps(plot_specs)};
for(const id in P){{Plotly.newPlot(id,P[id].data,P[id].layout,{{responsive:true,displaylogo:false}});}}
</script></body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
