"""Self-contained: for each task-shape FAMILY (see build_families in
plot_task_features_sample_area.py), list the top-N closest raw pretraining TOKENS by
trajectory distance, with before/word/after context, for manual inspection. Unlike the
SAE-feature viz, this operates directly on individual tokens -- no SAE, no feature
grouping.

Single full-corpus pass over the memmap (RMS z-curve distance to each family curve),
keeping a running top-N per family, then resolves word_id -> context text via
input_features parquets. Renders one collapsible, searchable table per family; each row
expands to a small plot of that token's own z-curve vs the family curve.

Usage:
    python plot_family_tokens_manual.py --family-tau 0.30 --top-n 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_task_features_sample_area as M
from sae.data_utils import get_words_in_context


def full_corpus_topn_per_family(fam_z, fam_names, cache_dir, chunk, top_n):
    info = json.loads((Path(cache_dir) / "cached_data_info.json").read_text())
    shape = tuple(info["shape"])
    n_out = info["output_feature_dim"]
    mm = np.memmap(info["filename"], dtype=np.float16, mode="r", shape=shape)
    n_rows = shape[0]
    fam_curves = np.vstack([fam_z[k][1] for k in fam_names])

    best_d = [np.full(top_n, np.inf) for _ in fam_names]
    best_row = [np.full(top_n, -1, dtype=np.int64) for _ in fam_names]
    best_curve = [np.zeros((top_n, n_out)) for _ in fam_names]

    t0 = time.time()
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        block = np.asarray(mm[start:end, -n_out:], dtype=np.float32)
        std = block.std(axis=1)
        ok = std > 1e-6
        if not ok.any():
            continue
        b = block[ok]
        idxs = np.nonzero(ok)[0] + start
        z = (b - b.mean(axis=1, keepdims=True)) / std[ok, None]
        for fi, fc in enumerate(fam_curves):
            d = np.sqrt(np.mean((z - fc[None, :]) ** 2, axis=1))
            cat_d = np.concatenate([best_d[fi], d])
            cat_row = np.concatenate([best_row[fi], idxs])
            cat_curve = np.vstack([best_curve[fi], z])
            order_k = np.argsort(cat_d)[:top_n]
            best_d[fi] = cat_d[order_k]
            best_row[fi] = cat_row[order_k]
            best_curve[fi] = cat_curve[order_k]
        if (start // chunk) % 10 == 0:
            print(f"  {end:,}/{n_rows:,} ({time.time()-t0:.0f}s)")
    print(f"scan done in {time.time()-t0:.0f}s")

    # row -> doc_id/word_pos, via file_to_doc.csv offsets
    out_dir = Path(info["data_dir"]) / "output_features"
    step_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    f2d = pd.read_csv(step_dirs[0] / "file_to_doc.csv")
    offsets = (np.cumsum(f2d["num_words"].values) - f2d["num_words"].values).astype(np.int64)
    doc_ids = f2d["doc_id"].astype(str).values
    order = np.argsort(offsets)
    offsets_sorted = offsets[order]
    doc_ids_sorted = doc_ids[order]
    counts_sorted = f2d["num_words"].values[order]

    def row_to_wordid(row):
        i = np.searchsorted(offsets_sorted, row, side="right") - 1
        doc = doc_ids_sorted[i]
        pos = row - offsets_sorted[i]
        if pos < 0 or pos >= counts_sorted[i]:
            return None
        return f"{doc}_{pos}"

    input_feat_dir = str(Path(info["data_dir"]) / "input_features")
    all_wids = []
    for fi in range(len(fam_names)):
        for r in best_row[fi]:
            if r >= 0:
                w = row_to_wordid(int(r))
                if w:
                    all_wids.append(w)
    print(f"resolving context for {len(set(all_wids))} unique tokens...")
    ctx = get_words_in_context(input_feat_dir, list(set(all_wids)), N=10)

    results = {}
    for fi, k in enumerate(fam_names):
        rows = []
        for j in range(top_n):
            r = int(best_row[fi][j])
            if r < 0:
                continue
            wid = row_to_wordid(r)
            c = ctx.get(wid, {}) if wid else {}
            rows.append({
                "rank": j + 1,
                "dist": float(best_d[fi][j]),
                "word_id": wid,
                "before": c.get("before", ""),
                "word": c.get("word", ""),
                "after": c.get("after", ""),
                "curve": best_curve[fi][j].tolist(),
            })
        results[k] = rows
    return results, list(model_names_from_info(info))


def model_names_from_info(info):
    return info["model_names"]


def esc(s):
    import html
    return html.escape(str(s) if s is not None else "")


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; color: #222; background: #fafafa; }
header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 5;}
h1 { margin: 0 0 4px; font-size: 18px; }
header .meta { color: #666; font-size: 12px; }
.wrap { padding: 24px; }
.fam-row { background: #fff; border: 1px solid #ddd; border-radius: 6px;
           padding: 16px; margin-bottom: 24px; }
.fam-row h2 { margin: 0 0 4px; font-size: 16px; }
.fam-row .members { color: #666; font-size: 12px; margin: 0 0 10px; }
.controls { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
.controls input[type=text] { flex: 1; padding: 6px 10px; border: 1px solid #ccc;
                             border-radius: 4px; font-size: 13px; }
.controls .count { font-size: 12px; color: #888; white-space: nowrap; }
table.tokens { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.tokens thead th { position: sticky; top: 56px; background: #f0f0f0; padding: 6px 8px;
                        border-bottom: 1px solid #ddd; text-align: left; cursor: pointer; }
table.tokens td { padding: 4px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
table.tokens tr.row-main { cursor: pointer; }
table.tokens tr.row-main:hover { background: #f5f8ff; }
table.tokens td.before { text-align: right; color: #888; max-width: 320px; overflow: hidden;
                         text-overflow: ellipsis; white-space: nowrap; }
table.tokens td.word { font-weight: 700; white-space: nowrap; color: #113; }
table.tokens td.after { color: #888; max-width: 320px; overflow: hidden;
                        text-overflow: ellipsis; white-space: nowrap; }
table.tokens td.num { font-family: monospace; text-align: right; white-space: nowrap; }
tr.plot-row td { padding: 0; border-bottom: 1px solid #eee; }
.plot-holder { width: 100%; height: 220px; background: #fff; }
.hidden { display: none; }
"""

JS = """
function filterFamily(famId) {
  const input = document.getElementById('search-' + famId);
  const q = input.value.toLowerCase();
  const rows = document.querySelectorAll('#tbody-' + famId + ' tr.row-main');
  let shown = 0;
  rows.forEach(function(tr) {
    const txt = tr.dataset.text;
    const match = txt.indexOf(q) !== -1;
    tr.classList.toggle('hidden', !match);
    const plotRow = tr.nextElementSibling;
    if (plotRow && plotRow.classList.contains('plot-row')) {
      plotRow.classList.toggle('hidden', !match || !plotRow.dataset.open);
    }
    if (match) shown++;
  });
  document.getElementById('count-' + famId).textContent = shown + ' shown';
}

function toggleRow(famId, rank) {
  const plotRow = document.getElementById('plot-' + famId + '-' + rank);
  const isOpen = plotRow.dataset.open === '1';
  plotRow.dataset.open = isOpen ? '' : '1';
  plotRow.classList.toggle('hidden', isOpen);
  if (!isOpen && !plotRow.dataset.plotted) {
    const spec = PLOTS[famId + '-' + rank];
    const div = plotRow.querySelector('.plot-holder');
    Plotly.newPlot(div, spec.data, spec.layout, {responsive: true, displaylogo: false});
    plotRow.dataset.plotted = '1';
  }
}

let sortState = {};
function sortTable(famId, col, numeric) {
  const tbody = document.getElementById('tbody-' + famId);
  const rows = Array.from(tbody.querySelectorAll('tr.row-main'));
  const key = famId + '-' + col;
  const asc = !(sortState[key] === 'asc');
  sortState[key] = asc ? 'asc' : 'desc';
  rows.sort(function(a, b) {
    let va = a.dataset[col], vb = b.dataset[col];
    if (numeric) { va = parseFloat(va); vb = parseFloat(vb); }
    if (va < vb) return asc ? -1 : 1;
    if (va > vb) return asc ? 1 : -1;
    return 0;
  });
  rows.forEach(function(tr) {
    tbody.appendChild(tr);
    const plotRow = tr.nextElementSibling;
    if (plotRow && plotRow.classList.contains('plot-row')) tbody.appendChild(plotRow);
  });
}
"""


def make_plot_spec(ckpts, token_curve, family_curve, family_label):
    traces = [
        {"x": list(ckpts), "y": list(token_curve), "type": "scatter", "mode": "lines+markers",
         "name": "token", "line": {"color": "#1976d2", "width": 2}, "marker": {"size": 4}},
        {"x": list(ckpts), "y": list(family_curve), "type": "scatter", "mode": "lines+markers",
         "name": family_label, "line": {"color": "#111", "width": 2, "dash": "dot"},
         "marker": {"size": 4}},
    ]
    layout = {
        "height": 210, "margin": {"l": 45, "r": 15, "t": 8, "b": 30},
        "xaxis": {"tickfont": {"size": 8}},
        "yaxis": {"title": "z-score", "titlefont": {"size": 9}, "zeroline": True},
        "legend": {"font": {"size": 9}, "orientation": "h", "y": -0.3},
    }
    return {"data": traces, "layout": layout}


def render_family_section(fam_id, fam_key, members, family_curve, ckpts, rows):
    thead = ("<tr><th onclick=\"sortTable('%s','rank',true)\">#</th>"
             "<th onclick=\"sortTable('%s','dist',true)\">dist</th>"
             "<th>Before</th><th>Word</th><th>After</th></tr>") % (fam_id, fam_id)
    trs = []
    for r in rows:
        rank = r["rank"]
        text = f"{r['before']} {r['word']} {r['after']}".lower()
        trs.append(
            f"<tr class='row-main' data-rank='{rank}' data-dist='{r['dist']:.6f}' "
            f"data-text=\"{esc(text)}\" onclick=\"toggleRow('{fam_id}',{rank})\">"
            f"<td class='num'>{rank}</td><td class='num'>{r['dist']:.4f}</td>"
            f"<td class='before'>{esc(r['before'])}</td>"
            f"<td class='word'>{esc(r['word'])}</td>"
            f"<td class='after'>{esc(r['after'])}</td></tr>")
        trs.append(
            f"<tr class='plot-row hidden' id='plot-{fam_id}-{rank}'>"
            f"<td colspan='5'><div class='plot-holder'></div></td></tr>")
    table = (f"<table class='tokens'><thead>{thead}</thead>"
             f"<tbody id='tbody-{fam_id}'>{''.join(trs)}</tbody></table>")
    controls = (f"<div class='controls'>"
                f"<input type='text' id='search-{fam_id}' placeholder='filter by text...' "
                f"oninput=\"filterFamily('{fam_id}')\">"
                f"<span class='count' id='count-{fam_id}'>{len(rows)} shown</span></div>")
    return (f"<section class='fam-row'><h2>{esc(fam_key)}</h2>"
            f"<p class='members'>{esc(' + '.join(members))} &middot; "
            f"{len(rows)} tokens (closest by trajectory distance)</p>"
            f"{controls}{table}</section>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", default=M.EVAL_RESULTS_DIR)
    ap.add_argument("--cache-dir", default=M.CACHE_DIR)
    ap.add_argument("--column-prefix", default=M.COLUMN_PREFIX)
    ap.add_argument("--family-tau", type=float, default=0.30)
    ap.add_argument("--family-metric", default="euclid", choices=["euclid", "jump", "tau"])
    ap.add_argument("--top-n", type=int, default=200,
                    help="Closest tokens per family to include.")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or Path(f"family_tokens_manual_tau{args.family_tau}.html")

    _, _, n_outputs, model_names = M._load_index(args.cache_dir)
    eval_ckpts = [m[len(args.column_prefix):] if m.startswith(args.column_prefix) else m
                  for m in model_names]

    task_z = {}
    for t in M.discover_tasks(args.eval_dir):
        perf, _ = M.load_task_curve(t, args.eval_dir, eval_ckpts)
        tz = M.znorm(perf)
        if tz is None or len(tz[0]) < len(perf):
            continue
        task_z[t] = tz
    fam_z, fam_members = M.build_families(task_z, args.family_tau, args.family_metric)
    fam_names = list(fam_z)
    print(f"{len(fam_names)} families (metric={args.family_metric}, thr={args.family_tau}):")
    for k in fam_names:
        print(f"  {k}: {' + '.join(fam_members[k])}")

    results, ckpt_names = full_corpus_topn_per_family(
        fam_z, fam_names, args.cache_dir, args.chunk, args.top_n)

    plot_specs, sections = {}, []
    for k in fam_names:
        idxs, tv = fam_z[k]
        full_curve = [None] * len(ckpt_names)
        for i, v in zip(idxs, tv):
            full_curve[i] = float(v)
        rows = results[k]
        for r in rows:
            fam_id = k
            plot_specs[f"{fam_id}-{r['rank']}"] = make_plot_spec(
                ckpt_names, r["curve"], full_curve, k)
        sections.append(render_family_section(k, k, fam_members[k], full_curve, ckpt_names, rows))

    summary = f"{len(fam_names)} families &middot; top {args.top_n} tokens each &middot; manual inspection"
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Family tokens — manual inspection</title>
<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>
<style>{CSS}</style></head><body>
<header><h1>Raw pretraining tokens closest to each family curve</h1>
<div class='meta'>{summary} &middot; click a row to plot its curve vs the family curve; type to filter</div></header>
<div class='wrap'>{''.join(sections)}</div>
<script>const PLOTS = {json.dumps(plot_specs)};{JS}</script>
</body></html>"""
    out_path.write_text(doc)
    print(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
