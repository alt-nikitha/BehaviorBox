"""HTML report of the top task-tracking tokens: for each token, show its text
context (target word highlighted) and its z-normalized probability trajectory
superimposed on the task's z-normalized performance curve (inline SVG).

Reads the ranked tokens from find_similar_to_tasks.py's JSONL output.

Usage:
    python tokens_to_html.py --jsonl tokens_similar_to_tasks.jsonl --top 40
"""

import argparse
import glob
import html
import json
import os

import numpy as np
import pandas as pd

from sample_curves import _load_index, DEFAULT_CACHE, get_sample_curve
from sample_by_task_centroids import (
    ANALYSIS_DIR, DEFAULT_STEM, build_task_centroids,
)

INPUT_DIR = ("/data/user_data/nsrikant/bbox_data/output/"
             "olmo_256000_unseen/input_features")


def _z(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-12 else v - v.mean()


def gather_contexts(targets, window):
    """targets: {word_id -> (doc_id, word_idx)}. Returns {word_id -> html ctx}.
    Reconstructs context from neighbouring words of the same doc in the input
    parquets; the target word is wrapped in <mark>."""
    need_docs = {d for d, _ in targets.values()}
    doc_words = {}  # doc_id -> dict(word_idx -> word)
    for pq in sorted(glob.glob(os.path.join(INPUT_DIR, "longformer-*.parquet"))):
        if not need_docs - doc_words.keys():
            break
        df = pd.read_parquet(pq, columns=["word_id", "doc_id", "word"])
        hit = df[df["doc_id"].isin(need_docs - doc_words.keys())]
        for doc_id, sub in hit.groupby("doc_id"):
            idx = sub["word_id"].str.rsplit("_", n=1).str[-1].astype(int)
            doc_words[doc_id] = dict(zip(idx, sub["word"]))

    def render(word):
        return (str(word).replace("Ġ", " ").replace("Ċ", "\n")
                .replace("ĉ", "\t").replace("Ī", " "))

    out = {}
    for wid, (doc_id, wi) in targets.items():
        words = doc_words.get(doc_id, {})
        lo, hi = wi - window, wi + window
        left = "".join(render(words[i]) for i in range(lo, wi) if i in words)
        tgt = render(words.get(wi, "?"))
        right = "".join(render(words[i]) for i in range(wi + 1, hi + 1)
                        if i in words)
        out[wid] = (f"{html.escape(left)}"
                    f"<mark>{html.escape(tgt) or '·'}</mark>"
                    f"{html.escape(right)}")
    return out


def svg_overlay(task_z, tok_z, w=320, h=120, pad=16):
    """Two z-curves on shared axes: task (blue, thick) vs token (orange)."""
    allv = np.concatenate([task_z, tok_z])
    lo, hi = float(allv.min()), float(allv.max())
    rng = hi - lo or 1.0
    n = len(task_z)

    def pts(curve):
        xs = np.linspace(pad, w - pad, n)
        ys = h - pad - (np.asarray(curve) - lo) / rng * (h - 2 * pad)
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    return (
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
        f"<rect x='0' y='0' width='{w}' height='{h}' fill='#fff'/>"
        f"<polyline fill='none' stroke='#1f77b4' stroke-width='2.5' "
        f"points='{pts(task_z)}'/>"
        f"<polyline fill='none' stroke='#ff7f0e' stroke-width='1.5' "
        f"points='{pts(tok_z)}'/>"
        f"</svg>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(
        ANALYSIS_DIR, "tokens_similar_to_tasks.jsonl"))
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--top", type=int, default=40,
                    help="Tokens shown per task (by rank).")
    ap.add_argument("--window", type=int, default=25,
                    help="Context words on each side of the target.")
    ap.add_argument("--out", default=os.path.join(
        ANALYSIS_DIR, "tokens_similar_to_tasks.html"))
    args = ap.parse_args()

    mm, doc_offset, n_outputs, _ = _load_index(args.cache_dir)
    centroids = build_task_centroids(args.stem, n_outputs)  # level z-curves

    rows = [json.loads(l) for l in open(args.jsonl)]
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    for t in by_task:
        by_task[t] = sorted(by_task[t], key=lambda r: r["rank"])[:args.top]

    # resolve contexts for all shown tokens at once
    targets = {}
    for items in by_task.values():
        for r in items:
            doc_id, wi = r["word_id"].rsplit("_", 1)
            targets[r["word_id"]] = (doc_id, int(wi))
    print(f"resolving context for {len(targets)} tokens...")
    ctx = gather_contexts(targets, args.window)

    sections = []
    for t, items in by_task.items():
        if t not in centroids:
            continue
        task_z = centroids[t]
        cards = []
        for r in items:
            raw = get_sample_curve(r["word_id"], cache_dir=args.cache_dir)
            if raw is None:
                continue
            tok_z = _z(raw)
            label = (f"r={r['r']:.4f}" if "r" in r
                     else f"dist={r['dist']:.4f}")
            cards.append(
                f"<div class='card'>"
                f"<div class='hd'>#{r['rank']} "
                f"<b>{label}</b> "
                f"<span class='wid'>{r['word_id']}</span></div>"
                f"{svg_overlay(task_z, tok_z)}"
                f"<div class='ctx'>{ctx.get(r['word_id'], '')}</div>"
                f"</div>")
        sections.append(
            f"<section><h2>{t} <small>top {len(cards)} tokens</small></h2>"
            f"<div class='grid'>{''.join(cards)}</div></section>")

    css = """
    body{font-family:-apple-system,'Segoe UI',sans-serif;margin:0;background:#fafafa;color:#222}
    header{padding:14px 22px;background:#fff;border-bottom:1px solid #ddd}
    h1{font-size:17px;margin:0 0 4px}.legend{font-size:12px;color:#666}
    .legend b.t{color:#1f77b4}.legend b.k{color:#ff7f0e}
    section{margin:16px 22px}h2{font-size:15px;margin:0 0 10px}h2 small{color:#888;font-weight:normal}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
    .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
    .hd{font-size:12px;margin-bottom:4px}.hd .wid{color:#999;font-family:monospace}
    .ctx{font-size:11px;line-height:1.5;color:#444;max-height:120px;overflow:auto;
         white-space:pre-wrap;border-top:1px solid #eee;margin-top:6px;padding-top:6px}
    mark{background:#ffe08a;font-weight:bold;padding:0 1px}
    """
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Task-tracking tokens</title><style>{css}</style></head><body>
<header><h1>Top task-tracking tokens — context + trajectory</h1>
<div class='legend'>Each card: token z-normalized probability trajectory
(<b class='k'>orange</b>) over <b class='t'>the task performance curve</b> (blue,
thick), then the token's text context with the target <mark>highlighted</mark>.
Ranked by the area (mean |Δ|) distance on the level z-curves, so closer = better
superimposition. Dataset <code>{html.escape(args.stem)}</code>.</div>
</header>{''.join(sections)}</body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
