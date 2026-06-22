"""One-pass empirical distribution of token-vs-task distance over the whole
corpus, per task. Emits a percentile ladder + tail counts so a per-task,
null-calibrated "tracks task t" cutoff can be chosen (areas/correlations are NOT
comparable across tasks in absolute terms, so calibrate by percentile).

  area    : mean |Δ| jump distance (CLOSEST = LOWER tail).
  pearson : 1 - r              (CLOSEST = LOWER tail; r near 1).
"""

import argparse

import numpy as np

from sample_curves import _load_index, DEFAULT_CACHE
from sample_by_task_centroids import DEFAULT_STEM, build_task_centroids, zrows
from find_similar_to_tasks import prep_centroids, block_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "blimp"])
    ap.add_argument("--metric", default="area", choices=["area", "pearson"])
    ap.add_argument("--levels", dest="diff", action="store_false", default=False)
    ap.add_argument("--diff", dest="diff", action="store_true")
    ap.add_argument("--hist-max", type=float, default=3.0,
                    help="Upper edge of the distance histogram.")
    ap.add_argument("--bins", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()

    mm, doc_offset, n_outputs, _ = _load_index(args.cache_dir)
    centroids = build_task_centroids(args.stem, n_outputs)
    tasks = list(args.tasks)
    C = np.stack([centroids[t] for t in tasks])
    if args.diff:
        C = np.diff(C, axis=1)
    Cproc = prep_centroids(C, args.metric)

    edges = np.linspace(0.0, args.hist_max, args.bins + 1)
    hist = {t: np.zeros(args.bins, dtype=np.int64) for t in tasks}
    over = {t: 0 for t in tasks}        # count beyond hist-max
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
        d = block_dist(zb, Cproc, args.metric)          # (n_ok, K)
        for ci, t in enumerate(tasks):
            col = d[:, ci]
            col = col[~np.isnan(col)]
            hist[t] += np.histogram(col, bins=edges)[0]
            over[t] += int((col > args.hist_max).sum())
        if (start // args.chunk) % 10 == 0:
            print(f"  rows {end:,}/{n_rows:,}", flush=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    # closest tokens are the LOWER tail of distance -> small percentiles.
    pcts = [0.001, 0.01, 0.1, 1, 50]
    for t in tasks:
        h = hist[t]
        N = int(h.sum()) + over[t]
        cum = np.cumsum(h)
        print(f"\n=== '{t}'  metric={args.metric}  N={N:,} ===")
        print("  closest-token cutoffs (keep tokens with dist <= value):")
        for p in pcts:
            target = p / 100.0 * N
            idx = int(np.searchsorted(cum, target))
            idx = min(idx, len(centers) - 1)
            below = int(cum[idx])
            print(f"    p{p:<7} dist<={centers[idx]:.5f}   "
                  f"(~{below:,} tokens at/below)")
        print("  tail counts:")
        for thr in (0.05, 0.1, 0.2, 0.3, 0.5):
            cnt = int(h[centers <= thr].sum())
            print(f"    dist<={thr:<4}: {cnt:>12,}  ({100*cnt/N:.4g}%)")


if __name__ == "__main__":
    main()
