"""Build an SAE training subsample balanced across *jump timing*, task-agnostic.

Problem: most token probability trajectories simply increase, and their rises
concentrate in a couple of checkpoint intervals, so an unsupervised SAE rarely
sees tokens that jump at other times. We fix the input imbalance directly:

  1. For each token, z-normalize its trajectory and find its single largest
     consecutive step (the "jump"); separately its most negative step ("drop").
  2. Keep only tokens whose dominant step is a *real* transition (magnitude in z
     >= --min-jump), to avoid bucketing flat / noisy tokens.
  3. Bucket by (sign, interval) — e.g. jump@interval-5, drop@interval-2 — and
     sample uniformly across buckets.

This never looks at task curves, so a later "feature jumps when task t jumps"
result is a finding, not something injected by the sampling. Per-token weights
(true bucket share / sampled share) let corpus prevalence be recovered.

Output JSONL rows:
    {"word_id","bucket":"jump@5"|"drop@2","interval":int,"sign":"jump"|"drop",
     "mag":float,"weight":float}

Usage:
    python sample_by_jump_interval.py --per-bucket 15000 --min-jump 1.0
    python sample_by_jump_interval.py --per-bucket 15000 --jumps-only
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from sample_curves import _load_index, DEFAULT_CACHE

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--per-bucket", type=int, default=15000,
                    help="Tokens to sample per (sign, interval) bucket.")
    ap.add_argument("--min-jump", type=float, default=1.0,
                    help="Min |dominant step| in z units to count as a real "
                         "jump/drop (guards against flat/noisy tokens).")
    ap.add_argument("--dominance", type=float, default=0.0,
                    help="Optional: require |largest step| >= dominance * "
                         "|2nd largest step| so the jump is localized (0=off).")
    ap.add_argument("--jumps-only", action="store_true",
                    help="Bucket only by jumps (rises), ignore drops.")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR,
                    "sae_sample_by_jump_interval.jsonl"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mm, doc_offset, T, _ = _load_index(args.cache_dir)
    n_intervals = T - 1
    print(f"rows={mm.shape[0]:,} T={T} intervals={n_intervals}")

    docs = np.array(sorted(doc_offset, key=lambda d: doc_offset[d]))
    starts = np.array([doc_offset[d] for d in docs], dtype=np.int64)

    def row_to_word_id(r):
        i = np.searchsorted(starts, r, side="right") - 1
        return f"{int(docs[i])}_{int(r - starts[i])}"

    # buckets: ("jump", iv) and optionally ("drop", iv)
    reservoir = defaultdict(list)      # bucket -> [(row, mag)]
    seen = defaultdict(int)            # bucket -> count seen (reservoir repl.)
    bucket_total = defaultdict(int)    # bucket -> true corpus count
    k = args.per_bucket
    n_rows = mm.shape[0]

    for start in range(0, n_rows, args.chunk):
        end = min(start + args.chunk, n_rows)
        B = np.asarray(mm[start:end, -T:], dtype=np.float32)
        mu = B.mean(1, keepdims=True)
        sd = B.std(1, keepdims=True)
        ok = sd[:, 0] >= 1e-9
        if not ok.any():
            continue
        Z = (B[ok] - mu[ok]) / sd[ok]
        D = np.diff(Z, axis=1)                       # (n_ok, intervals)
        rows = start + np.nonzero(ok)[0]

        def process(steps, sign):
            iv = np.argmax(steps, axis=1)
            mag = steps[np.arange(steps.shape[0]), iv]
            keep = mag >= args.min_jump
            if args.dominance > 0:
                # second-largest step per row
                part = np.partition(steps, -2, axis=1)
                second = part[:, -2]
                keep &= mag >= args.dominance * np.abs(second)
            for r, c, m in zip(rows[keep], iv[keep], mag[keep]):
                b = (sign, int(c))
                bucket_total[b] += 1
                seen[b] += 1
                res = reservoir[b]
                if len(res) < k:
                    res.append((int(r), float(m)))
                else:
                    j = rng.integers(0, seen[b])
                    if j < k:
                        res[j] = (int(r), float(m))

        process(D, "jump")
        if not args.jumps_only:
            process(-D, "drop")   # most-negative step becomes max of -D

        if (start // args.chunk) % 10 == 0:
            print(f"  rows {end:,}/{n_rows:,}", flush=True)

    total = sum(bucket_total.values())
    n_samp = sum(len(v) for v in reservoir.values())
    print(f"\nqualifying tokens (>= min-jump): {total:,} | sampled: {n_samp:,}")
    print("bucket sizes (corpus -> sampled):")
    for b in sorted(reservoir, key=lambda x: (x[0], x[1])):
        print(f"  {b[0]}@{b[1]:<2d}  corpus={bucket_total[b]:>9,}  "
              f"sampled={len(reservoir[b]):>7,}")

    with open(args.out, "w") as f:
        for b, res in reservoir.items():
            if not res:
                continue
            true_share = bucket_total[b] / total
            samp_share = len(res) / n_samp
            w = true_share / samp_share if samp_share > 0 else 0.0
            for r, m in res:
                f.write(json.dumps({
                    "word_id": row_to_word_id(r),
                    "bucket": f"{b[0]}@{b[1]}",
                    "interval": b[1],
                    "sign": b[0],
                    "mag": round(m, 5),
                    "weight": round(float(w), 5),
                }) + "\n")
    print(f"\nwrote {n_samp:,} rows to {args.out}")


if __name__ == "__main__":
    main()
