"""Build an SAE training subsample that is balanced across task-performance
"shapes".

Motivation: most token probability trajectories simply increase over training,
so an unsupervised SAE spends its dictionary on that majority shape and rarely
allocates features to rarer trajectories. To force coverage of task-relevant
shapes, we use each downstream task's (z-normalized) performance curve as a
cluster centroid, assign every token's z-normalized output trajectory to its
nearest task centroid, and then sample roughly uniformly across these clusters.

NOTE (honesty): because task curves are used to select training data, any later
"feature f tracks task t" result is partly engineered by this sampling. Report
this as deliberate shape-stratified sampling, not as unsupervised discovery.
Per-token importance weights (cluster size / overall) are emitted so corpus
prevalence can be recovered at analysis time.

Output: a JSONL with one row per selected token:
    {"word_id": "<doc>_<word>", "cluster": "<task>", "dist": float, "weight": float}

Usage:
    python sample_by_task_centroids.py --per-cluster 20000 --metric area
    python sample_by_task_centroids.py --per-cluster 20000 --max-dist 1.5
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from sample_curves import _load_index, DEFAULT_CACHE

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STEM = "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm"
VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


# ── task centroids ─────────────────────────────────────────────────────────

def build_task_centroids(stem, n_outputs):
    """Return {task: z-curve (n_outputs,)} from precomputed task JSONs.
    Only tasks whose performance curve has no missing checkpoints and matches
    the memmap's n_outputs are kept (so alignment is positional & complete)."""
    centroids = {}
    for d in sorted(glob.glob(os.path.join(ANALYSIS_DIR, "precomputed_data_*"))):
        task = os.path.basename(d).replace("precomputed_data_", "")
        if task not in VALID_TASKS:
            continue
        fp = os.path.join(d, f"{stem}.json")
        if not os.path.exists(fp):
            continue
        perf = json.load(open(fp)).get("overall_performance") or []
        vals = [p for p in perf if p is not None]
        if len(vals) != n_outputs:
            # require a complete curve aligned to the trajectory length
            continue
        x = np.array(vals, dtype=np.float64)
        if x.std() < 1e-12:
            continue
        centroids[task] = ((x - x.mean()) / x.std()).astype(np.float32)
    return centroids


# ── distance ───────────────────────────────────────────────────────────────

def zrows(block):
    """Row-wise z-normalize a (N, C) block; rows with ~0 std get NaN."""
    mu = block.mean(axis=1, keepdims=True)
    sd = block.std(axis=1, keepdims=True)
    out = (block - mu) / sd
    out[sd[:, 0] < 1e-9] = np.nan
    return out


def dist_to_centroids(zblock, C, metric):
    """(N, T) z-block vs (K, T) centroid matrix -> (N, K) distances.
    area = mean |diff| ; mse = mean diff^2. (Cheap, vectorized; spacing uniform.)

    NOTE: when called in jump mode, both `zblock` and `C` are already the
    consecutive-difference (and re-centered) curves, so this compares jump shapes
    rather than levels."""
    diff = zblock[:, None, :] - C[None, :, :]
    if metric == "mse":
        return np.mean(diff ** 2, axis=2)
    return np.mean(np.abs(diff), axis=2)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="sample_curves memmap cache dir.")
    ap.add_argument("--metric", default="area", choices=["area", "mse"])
    ap.add_argument("--diff", dest="diff", action="store_true", default=True,
                    help="Match on JUMP shape: difference the z-curves before "
                         "comparing, so tokens are assigned by where they jump, "
                         "not by overall level. Default on (task level-curves are "
                         "near-redundant; jump curves are distinct).")
    ap.add_argument("--levels", dest="diff", action="store_false",
                    help="Match on raw z-curve levels instead of jump shape.")
    ap.add_argument("--per-cluster", type=int, default=20000,
                    help="Tokens to sample per task cluster (uniform target).")
    ap.add_argument("--max-dist", type=float, default=None,
                    help="Drop tokens whose nearest-centroid distance exceeds "
                         "this (keeps only tokens that actually match a shape).")
    ap.add_argument("--chunk", type=int, default=200_000,
                    help="Memmap rows processed per chunk.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR,
                    "sae_sample_by_task_centroids.jsonl"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mm, doc_offset, n_outputs, model_names = _load_index(args.cache_dir)
    print(f"memmap rows: {mm.shape[0]:,} | n_outputs: {n_outputs}")

    centroids = build_task_centroids(args.stem, n_outputs)
    if not centroids:
        raise SystemExit("No task centroids matched n_outputs; check stem / "
                         "that performance curves are complete.")
    tasks = sorted(centroids)
    C = np.stack([centroids[t] for t in tasks])  # (K, n_outputs)
    print(f"task centroids: {len(tasks)} -> {tasks}")

    # invert doc_offset so a memmap row -> (doc_id, word_idx)
    # offsets are start row of each doc; build a sorted boundary array.
    docs = np.array(sorted(doc_offset, key=lambda d: doc_offset[d]))
    starts = np.array([doc_offset[d] for d in docs], dtype=np.int64)

    def row_to_word_id(r):
        i = np.searchsorted(starts, r, side="right") - 1
        return f"{int(docs[i])}_{int(r - starts[i])}"

    # Reservoir per cluster so memory is bounded regardless of corpus size.
    k = args.per_cluster
    reservoir = {t: [] for t in tasks}     # list of (row, dist)
    seen_per = defaultdict(int)            # for reservoir replacement
    assigned = np.zeros(len(tasks), dtype=np.int64)  # total assigned per cluster
    n_rows = mm.shape[0]

    for start in range(0, n_rows, args.chunk):
        end = min(start + args.chunk, n_rows)
        block = np.asarray(mm[start:end, -n_outputs:], dtype=np.float32)
        z = zrows(block)
        ok = ~np.isnan(z).any(axis=1)
        if not ok.any():
            continue
        d = dist_to_centroids(z[ok], C, args.metric)  # (n_ok, K)
        nearest = d.argmin(axis=1)
        ndist = d[np.arange(d.shape[0]), nearest]
        rows = (start + np.nonzero(ok)[0])
        for r, c, dd in zip(rows, nearest, ndist):
            if args.max_dist is not None and dd > args.max_dist:
                continue
            t = tasks[c]
            assigned[c] += 1
            seen_per[t] += 1
            res = reservoir[t]
            if len(res) < k:
                res.append((int(r), float(dd)))
            else:  # reservoir replacement
                j = rng.integers(0, seen_per[t])
                if j < k:
                    res[j] = (int(r), float(dd))
        if (start // args.chunk) % 10 == 0:
            print(f"  rows {end:,}/{n_rows:,} | assigned per cluster: "
                  f"{dict(zip(tasks, assigned.tolist()))}", flush=True)

    total_assigned = int(assigned.sum())
    print(f"\ntotal assigned tokens: {total_assigned:,}")
    print("cluster sizes (full corpus) and sampled counts:")
    for t, a in zip(tasks, assigned):
        print(f"  {t:22s} corpus={int(a):>9,}  sampled={len(reservoir[t]):>7,}")

    # importance weight = (cluster's true share) / (cluster's sampled share),
    # so reweighting the sample recovers corpus prevalence at analysis time.
    n_sampled = sum(len(v) for v in reservoir.values())
    with open(args.out, "w") as f:
        for t, a in zip(tasks, assigned):
            res = reservoir[t]
            if not res or a == 0:
                continue
            true_share = a / total_assigned
            samp_share = len(res) / n_sampled
            weight = true_share / samp_share if samp_share > 0 else 0.0
            for r, dd in res:
                f.write(json.dumps({
                    "word_id": row_to_word_id(r),
                    "cluster": t,
                    "dist": round(dd, 5),
                    "weight": round(float(weight), 5),
                }) + "\n")

    print(f"\nwrote {n_sampled:,} sampled tokens to {args.out}")
    print("NOTE: clusters are task-curve centroids; treat resulting feature/task "
          "matches as shape-stratified sampling, not unsupervised discovery.")


if __name__ == "__main__":
    main()
