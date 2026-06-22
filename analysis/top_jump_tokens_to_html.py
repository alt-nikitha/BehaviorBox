"""Per-task HTML of the actual TOKENS whose probability jump aligns with the
task's jump.

For each task we take its z-normalized performance curve, find its biggest jump
interval, and among sampled tokens select those whose own biggest jump is on the
same interval (within --tol), ranked by jump magnitude. We resolve each token's
surface word from the input-feature parquets and list them with their jump
interval and magnitude, plus a per-task histogram of jump intervals so you can
see how concentrated the match is.

Usage:
    python top_jump_tokens_to_html.py --sample 400000 --top 60 --tol 0
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from sample_curves import _load_index, DEFAULT_CACHE

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STEM = "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm"
INPUT_DIR = "/data/user_data/nsrikant/bbox_data/output/olmo_256000_unseen/input_features"
VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


def task_centroids(stem, T):
    cents, ckpts = {}, None
    for d in sorted(glob.glob(os.path.join(ANALYSIS_DIR, "precomputed_data_*"))):
        t = os.path.basename(d).replace("precomputed_data_", "")
        if t not in VALID_TASKS:
            continue
        fp = os.path.join(d, f"{stem}.json")
        if not os.path.exists(fp):
            continue
        j = json.load(open(fp))
        perf = [p for p in (j.get("overall_performance") or []) if p is not None]
        if len(perf) != T:
            continue
        x = np.array(perf, float)
        if x.std() < 1e-12:
            continue
        cents[t] = ((x - x.mean()) / x.std()).astype(np.float32)
        if ckpts is None:
            ckpts = j.get("checkpoints")
    return cents, ckpts


def resolve_words(word_ids):
    """Map a set of '<doc>_<word>' ids to surface words via input parquets."""
    want = set(word_ids)
    out = {}
    for pq in sorted(glob.glob(os.path.join(INPUT_DIR, "longformer-*.parquet"))):
        df = pd.read_parquet(pq, columns=["word_id", "word"])
        hit = df[df["word_id"].isin(want)]
        for wid, w in zip(hit["word_id"], hit["word"]):
            out[wid] = w
        if len(out) >= len(want):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--tol", type=int, default=0)
    ap.add_argument("--min-jump", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR,
                    "top_jump_tokens.html"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mm, doc_offset, T, _ = _load_index(args.cache_dir)
    cents, ckpts = task_centroids(args.stem, T)
    tasks = sorted(cents)

    docs = np.array(sorted(doc_offset, key=lambda d: doc_offset[d]))
    starts = np.array([doc_offset[d] for d in docs], dtype=np.int64)

    def row_to_word_id(r):
        i = np.searchsorted(starts, r, side="right") - 1
        return f"{int(docs[i])}_{int(r - starts[i])}"

    rows = rng.choice(mm.shape[0], size=min(args.sample, mm.shape[0]),
                      replace=False)
    rows.sort()
    B = np.asarray(mm[rows][:, -T:], dtype=np.float32)
    mu = B.mean(1, keepdims=True)
    sd = B.std(1, keepdims=True)
    ok = sd[:, 0] >= 1e-9
    Z = (B[ok] - mu[ok]) / sd[ok]
    rows = rows[ok]
    Dz = np.diff(Z, axis=1)
    tok_iv = np.argmax(Dz, axis=1)
    tok_mag = Dz[np.arange(Dz.shape[0]), tok_iv]
    real = tok_mag >= args.min_jump
    Z, rows, tok_iv, tok_mag = Z[real], rows[real], tok_iv[real], tok_mag[real]
    print(f"usable jumping tokens: {len(rows):,}")

    # task jump interval
    task_iv = {t: int(np.argmax(np.diff(cents[t]))) for t in tasks}

    # collect top tokens per task; gather word_ids to resolve once
    per_task = {}
    need = set()
    for t in tasks:
        ti = task_iv[t]
        sel = np.nonzero(np.abs(tok_iv - ti) <= args.tol)[0]
        sel = sel[np.argsort(-tok_mag[sel])][:args.top]
        items = [(row_to_word_id(int(rows[s])), int(tok_iv[s]), float(tok_mag[s]))
                 for s in sel]
        per_task[t] = items
        need.update(w for w, _, _ in items)
    print(f"resolving {len(need):,} word ids...")
    words = resolve_words(need)

    def ck(i):
        return ckpts[i] if ckpts and i < len(ckpts) else f"i{i}"

    sections = []
    for t in tasks:
        ti = task_iv[t]
        items = per_task[t]
        # interval histogram across ALL jumping tokens (context)
        hist = np.bincount(tok_iv, minlength=T - 1)
        hbar = " ".join(f"{i}:{hist[i]}" for i in range(T - 1))
        chips = []
        for wid, iv, mag in items:
            w = words.get(wid, "?")
            wtxt = (w.replace("Ġ", "␣").replace("Ċ", "⏎").replace("ĉ", "⇥"))
            chips.append(
                f"<span class='tok' title='{wid} | {ck(iv)}->{ck(iv+1)} | mag={mag:.2f}z'>"
                f"{_esc(wtxt)}</span>")
        sections.append(
            f"<section class='task'><h3>{t} "
            f"<small>jump @ {ck(ti)}→{ck(ti+1)} (interval {ti}) · "
            f"{len(items)} tokens</small></h3>"
            f"<div class='toks'>{''.join(chips) or '<i>none</i>'}</div></section>")

    css = """
    body{font-family:-apple-system,'Segoe UI',sans-serif;margin:0;background:#fafafa;color:#222}
    header{padding:14px 22px;background:#fff;border-bottom:1px solid #ddd}
    h1{font-size:17px;margin:0 0 4px}.meta{color:#666;font-size:12px}
    .task{background:#fff;border:1px solid #ddd;border-radius:6px;margin:14px 22px;padding:12px}
    .task h3{margin:0 0 8px;font-size:14px}.task small{color:#888;font-weight:normal}
    .toks{display:flex;flex-wrap:wrap;gap:5px}
    .tok{font-family:monospace;font-size:12px;background:#eef;border:1px solid #ccd;
         border-radius:3px;padding:1px 5px;color:#224;cursor:help;white-space:pre}
    """
    doc = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Top jump-matched tokens per task</title><style>{css}</style></head><body>
<header><h1>Top tokens whose probability jump aligns with each task's jump</h1>
<div class='meta'>Dataset <code>{args.stem}</code> · sampled {len(rows):,} jumping tokens
(|step|≥{args.min_jump}z) · tol {args.tol} interval · hover a token for word_id,
interval, magnitude · ␣=leading-space, ⏎=newline, ⇥=tab.</div></header>
{''.join(sections)}</body></html>"""
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


def _esc(s):
    import html
    return html.escape(str(s))


if __name__ == "__main__":
    main()
