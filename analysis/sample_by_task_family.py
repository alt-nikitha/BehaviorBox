"""Cluster the downstream tasks into FAMILIES (tasks whose performance curves
share a shape), then sample SAE-training tokens balanced across families.

Why families, not tasks: the 15 task performance curves are ~0.84 mutually
correlated, so per-task token strata overlap 78-92%. But the overlap is
structured -- tasks co-occur in consistent families (knowledge MCQ, early-LM,
reading, late-transition). Clustering first collapses that redundancy into
separable strata.

Step 1: hierarchical clustering of task centroids (z-normed perf curves) by mean
        |Δz| distance -> task families.
Step 2: each family's centroid = mean of its tasks' z-curves.
Step 3: assign every token to its nearest family centroid; keep tokens within
        --max-dist; sample uniformly across families (reservoir).

Output JSONL: {"word_id","family":int,"family_tasks":[...],"dist":float,"weight":float}

Usage:
    python sample_by_task_family.py --n-families 5 --max-dist 0.6 --per-family 200000
    python sample_by_task_family.py --print-families-only   # just show the clustering
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


def cluster_tasks(cents, n_families):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    tasks = sorted(cents)
    C = np.stack([cents[t] for t in tasks])
    K = len(tasks)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = D[j, i] = np.mean(np.abs(C[i] - C[j]))
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=n_families, criterion="maxclust")
    fam = defaultdict(list)
    for t, l in zip(tasks, labels):
        fam[int(l)].append(t)
    # family centroid = mean of member z-curves
    fam_cent = {f: np.mean([cents[t] for t in members], axis=0).astype(np.float32)
                for f, members in fam.items()}
    return fam, fam_cent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--n-families", type=int, default=5)
    ap.add_argument("--max-dist", type=float, default=0.6)
    ap.add_argument("--per-family", type=int, default=200000)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-families-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR,
                    "sae_sample_by_task_family.jsonl"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    mm, doc_off, T, _ = _load_index(args.cache_dir)
    cents, ckpts = task_centroids(args.stem, T)
    fam, fam_cent = cluster_tasks(cents, args.n_families)

    fids = sorted(fam)
    FC = np.stack([fam_cent[f] for f in fids])  # (F, T)
    print(f"{len(cents)} tasks -> {len(fids)} families:")
    for f in fids:
        print(f"  family {f}: {fam[f]}")
    if args.print_families_only:
        return

    docs = np.array(sorted(doc_off, key=lambda d: doc_off[d]))
    starts = np.array([doc_off[d] for d in docs], dtype=np.int64)

    def row_to_word_id(r):
        i = np.searchsorted(starts, r, side="right") - 1
        return f"{int(docs[i])}_{int(r - starts[i])}"

    k = args.per_family
    reservoir = {f: [] for f in fids}
    seen = defaultdict(int)
    assigned = np.zeros(len(fids), dtype=np.int64)
    n_rows = mm.shape[0]

    for s in range(0, n_rows, args.chunk):
        e = min(s + args.chunk, n_rows)
        B = np.asarray(mm[s:e, -T:], dtype=np.float32)
        mu = B.mean(1, keepdims=True)
        sd = B.std(1, keepdims=True)
        ok = sd[:, 0] >= 1e-9
        if not ok.any():
            continue
        Z = (B[ok] - mu[ok]) / sd[ok]
        d = np.mean(np.abs(Z[:, None, :] - FC[None, :, :]), axis=2)  # (n,F)
        nn = d.argmin(1)
        nd = d[np.arange(d.shape[0]), nn]
        rows = s + np.nonzero(ok)[0]
        for r, c, dd in zip(rows, nn, nd):
            if dd > args.max_dist:
                continue
            f = fids[c]
            assigned[c] += 1
            seen[f] += 1
            res = reservoir[f]
            if len(res) < k:
                res.append((int(r), float(dd)))
            else:
                j = rng.integers(0, seen[f])
                if j < k:
                    res[j] = (int(r), float(dd))
        if (s // args.chunk) % 10 == 0:
            print(f"  rows {e:,}/{n_rows:,} assigned={dict(zip(fids, assigned.tolist()))}",
                  flush=True)

    total = int(assigned.sum())
    n_samp = sum(len(v) for v in reservoir.values())
    print(f"\nwithin max_dist {args.max_dist} of a family: {total:,}")
    print("family sizes (corpus -> sampled):")
    for f, a in zip(fids, assigned):
        print(f"  family {f} {str(fam[f])[:50]:50s} corpus={int(a):>9,} sampled={len(reservoir[f]):>7,}")

    with open(args.out, "w") as fout:
        for f, a in zip(fids, assigned):
            res = reservoir[f]
            if not res or a == 0:
                continue
            w = (a / total) / (len(res) / n_samp) if n_samp else 0.0
            for r, dd in res:
                fout.write(json.dumps({
                    "word_id": row_to_word_id(r),
                    "family": f,
                    "family_tasks": fam[f],
                    "dist": round(dd, 5),
                    "weight": round(float(w), 5),
                }) + "\n")
    print(f"\nwrote {n_samp:,} tokens to {args.out}")


if __name__ == "__main__":
    main()
