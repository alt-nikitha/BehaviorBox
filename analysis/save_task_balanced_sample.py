"""Save a size-balanced token sample for a set of tasks: keep all tokens whose
level-area distance to a task's performance curve is below --max-dist, then
downsample every task to the smallest task's count (closest-N) so the clusters
are equal-sized. Emits one row per kept token, schema matching the other
sae_sample_*.jsonl files:
    {"word_id", "task", "dist", "weight"}
weight = (task's true count below threshold) / (kept count), so reweighting the
sample recovers corpus prevalence.

Usage:
    python save_task_balanced_sample.py --tasks gsm8k blimp --max-dist 0.5 \
        --out sae_sample_by_task_gsm8k_blimp_942k.jsonl
"""

import argparse
import json
import os

import numpy as np

from sample_curves import _load_index, DEFAULT_CACHE
from sample_by_task_centroids import (
    ANALYSIS_DIR, DEFAULT_STEM, build_task_centroids, zrows,
)
from find_similar_to_tasks import prep_centroids, block_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "blimp"])
    ap.add_argument("--metric", default="area", choices=["area", "pearson"])
    ap.add_argument("--max-dist", type=float, default=0.5)
    ap.add_argument("--levels", dest="diff", action="store_false", default=False)
    ap.add_argument("--diff", dest="diff", action="store_true")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mm, doc_offset, n_outputs, _ = _load_index(args.cache_dir)
    centroids = build_task_centroids(args.stem, n_outputs)
    tasks = list(args.tasks)
    C = np.stack([centroids[t] for t in tasks])
    if args.diff:
        C = np.diff(C, axis=1)
    Cproc = prep_centroids(C, args.metric)
    print(f"keeping tokens with {args.metric} dist <= {args.max_dist} | "
          f"shape={'jump' if args.diff else 'level'} | tasks={tasks}")

    docs = np.array(sorted(doc_offset, key=lambda d: doc_offset[d]))
    starts = np.array([doc_offset[d] for d in docs], dtype=np.int64)

    def row_to_word_id(r):
        i = np.searchsorted(starts, r, side="right") - 1
        return f"{int(docs[i])}_{int(r - starts[i])}"

    # collect ALL tokens below threshold per task (rows + dists)
    keep_rows = {t: [] for t in tasks}
    keep_dist = {t: [] for t in tasks}
    n_rows = mm.shape[0]
    for start in range(0, n_rows, args.chunk):
        end = min(start + args.chunk, n_rows)
        block = np.asarray(mm[start:end, -n_outputs:], dtype=np.float32)
        z = zrows(block)
        ok = ~np.isnan(z).any(axis=1)
        if not ok.any():
            continue
        zb = z[ok]
        if args.diff:
            zb = np.diff(zb, axis=1)
        d = block_dist(zb, Cproc, args.metric)
        rows = start + np.nonzero(ok)[0]
        for ci, t in enumerate(tasks):
            m = d[:, ci] <= args.max_dist
            if m.any():
                keep_rows[t].append(rows[m])
                keep_dist[t].append(d[m, ci])
        if (start // args.chunk) % 10 == 0:
            print(f"  rows {end:,}/{n_rows:,}", flush=True)

    counts = {}
    for t in tasks:
        keep_rows[t] = (np.concatenate(keep_rows[t]) if keep_rows[t]
                        else np.empty(0, np.int64))
        keep_dist[t] = (np.concatenate(keep_dist[t]) if keep_dist[t]
                        else np.empty(0, np.float32))
        counts[t] = len(keep_rows[t])
    N = min(counts.values())   # balance to smallest task
    print(f"\ncounts below {args.max_dist}: "
          f"{ {t: counts[t] for t in tasks} }")
    print(f"balancing every task to N = {N:,} (closest-N)")

    with open(args.out, "w") as f:
        for t in tasks:
            order = np.argsort(keep_dist[t])[:N]   # N closest
            rows, dist = keep_rows[t][order], keep_dist[t][order]
            weight = counts[t] / N                 # recover corpus prevalence
            for r, dd in zip(rows, dist):
                f.write(json.dumps({
                    "word_id": row_to_word_id(int(r)),
                    "task": t,
                    "dist": round(float(dd), 5),
                    "weight": round(float(weight), 5),
                }) + "\n")
    print(f"\nwrote {N * len(tasks):,} rows ({N:,}/task) to {args.out}")


if __name__ == "__main__":
    main()
