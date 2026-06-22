"""Find tokens whose z-normalized output trajectory is most similar to a
specific task's performance curve, ranked independently per named task.

Unlike sample_by_task_centroids.py (which assigns each token to its *nearest*
centroid among all tasks and samples per cluster), this script ranks every token
by distance to each *named* task curve and emits the top-K closest tokens per
task. Use it to answer "which tokens look most like gsm8k?" / "...like blimp?".

Metrics (on the jump curves by default; --levels to use raw z-levels):
  area    : mean |Δ| between token and task curve (smaller = closer). Default.
  pearson : 1 - r (smaller = closer); scale-invariant.

Because the area distance is NOT comparable across tasks in absolute terms,
calibrate thresholds per task with task_metric_distribution.py (percentiles),
then pass --max-dist for that task's cutoff.

Usage:
    python find_similar_to_tasks.py --tasks gsm8k blimp --top-k 200
    python find_similar_to_tasks.py --tasks gsm8k blimp --metric pearson
"""

import argparse
import json
import os

import numpy as np

from sample_curves import _load_index, DEFAULT_CACHE
from sample_by_task_centroids import (
    ANALYSIS_DIR, DEFAULT_STEM, build_task_centroids, zrows, dist_to_centroids,
)


def _center_unit(M):
    """Center each row and scale to unit L2 norm; degenerate rows -> NaN."""
    A = M - M.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n < 1e-12] = np.nan
    return A / n


def pearson_dist(zblock, Cn):
    """(N, T) curves vs pre-normalized (K, T) centroids -> (N, K) of 1 - r."""
    a = _center_unit(zblock)
    return np.nan_to_num(1.0 - a @ Cn.T, nan=2.0)


def prep_centroids(C, metric):
    """Pre-process centroids for the chosen metric (curves already diffed if
    in jump mode). Returns the array to hand to the per-chunk distance fn."""
    if metric == "pearson":
        return _center_unit(C)
    return C - C.mean(axis=1, keepdims=True)          # area: recenter levels


def block_dist(zb, Cproc, metric):
    if metric == "pearson":
        return pearson_dist(zb, Cproc)
    zb = zb - zb.mean(axis=1, keepdims=True)
    return dist_to_centroids(zb, Cproc, "area")       # mean |Δ|


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "blimp"],
                    help="Task curves to rank tokens against.")
    ap.add_argument("--metric", default="area", choices=["area", "pearson"])
    ap.add_argument("--top-k", type=int, default=200,
                    help="Number of closest tokens to keep per task.")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="Only keep tokens with distance below this (per-task "
                         "calibrated cutoff). Applied on top of --top-k.")
    ap.add_argument("--levels", dest="diff", action="store_false", default=False,
                    help="Match on raw z-curve LEVELS (default): the token's "
                         "trajectory must track the task performance curve, so "
                         "it superimposes in the plot.")
    ap.add_argument("--diff", dest="diff", action="store_true",
                    help="Match on JUMP shape (difference z-curves first) "
                         "instead. Admits level-anticorrelated tokens that only "
                         "share a spike; useful for separating similar tasks.")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR,
                    "tokens_similar_to_tasks.jsonl"))
    args = ap.parse_args()

    mm, doc_offset, n_outputs, _ = _load_index(args.cache_dir)
    print(f"memmap rows: {mm.shape[0]:,} | n_outputs: {n_outputs}")

    centroids = build_task_centroids(args.stem, n_outputs)
    missing = [t for t in args.tasks if t not in centroids]
    if missing:
        raise SystemExit(f"no usable centroid for {missing}; available: "
                         f"{sorted(centroids)}")
    tasks = list(args.tasks)
    C = np.stack([centroids[t] for t in tasks])  # (K, n_outputs)
    if args.diff:
        C = np.diff(C, axis=1)
    Cproc = prep_centroids(C, args.metric)
    print(f"ranking against {tasks} | metric={args.metric} | "
          f"shape={'jump' if args.diff else 'level'}")

    docs = np.array(sorted(doc_offset, key=lambda d: doc_offset[d]))
    starts = np.array([doc_offset[d] for d in docs], dtype=np.int64)

    def row_to_word_id(r):
        i = np.searchsorted(starts, r, side="right") - 1
        return f"{int(docs[i])}_{int(r - starts[i])}"

    k = args.top_k
    best_rows = {t: np.empty(0, dtype=np.int64) for t in tasks}
    best_dist = {t: np.empty(0, dtype=np.float32) for t in tasks}
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
        d = block_dist(zb, Cproc, args.metric)         # (n_ok, K)
        rows = start + np.nonzero(ok)[0]
        for ci, t in enumerate(tasks):
            cr = np.concatenate([best_rows[t], rows])
            cd = np.concatenate([best_dist[t], d[:, ci]])
            if cd.shape[0] > k:
                keep = np.argpartition(cd, k)[:k]
                cr, cd = cr[keep], cd[keep]
            best_rows[t], best_dist[t] = cr, cd
        if (start // args.chunk) % 10 == 0:
            print(f"  rows {end:,}/{n_rows:,}", flush=True)

    with open(args.out, "w") as f:
        for t in tasks:
            order = np.argsort(best_dist[t])      # ascending: closest first
            rows, dist = best_rows[t][order], best_dist[t][order]
            if args.max_dist is not None:
                m = dist <= args.max_dist
                rows, dist = rows[m], dist[m]
            print(f"\n=== top {len(rows)} tokens closest to '{t}' "
                  f"({args.metric}) ===")
            for rank, (row, dd) in enumerate(zip(rows, dist)):
                wid = row_to_word_id(int(row))
                if rank < 10:
                    print(f"  {rank:3d}  dist={dd:.5f}  {wid}")
                rec = {"word_id": wid, "task": t,
                       "dist": round(float(dd), 5), "rank": rank}
                if args.metric == "pearson":
                    rec["r"] = round(float(1 - dd), 5)
                f.write(json.dumps(rec) + "\n")

    print(f"\nwrote top-{k} per task to {args.out}")


if __name__ == "__main__":
    main()
