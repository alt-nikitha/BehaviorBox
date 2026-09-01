"""Uniformly-sampled SAE subset balanced across task SHAPE FAMILIES (not per task, and
not by a Kendall-tau threshold).

Difference from build_tau_subset_all_tasks.py:
  * unit = shape FAMILY (tasks with the same trajectory shape merged), ~4 not 12/13.
  * assignment = each token goes to its single NEAREST family by euclidean shape distance
    (mean |z_token - z_family| per checkpoint), NOT a tau>=x pool. Euclid respects WHEN the
    rise happens, so early- and late-risers separate (tau over-merges them).
  * sampling = equal count per family, drawn uniformly at random from that family's members.

Families come from clustering the task performance curves (euclidean, average linkage).

Usage:
  python build_family_subset.py --threshold 0.55            # see families + build
  python build_family_subset.py --threshold 0.55 --per-family 1000000
  python build_family_subset.py --families-only             # just print the families
"""
import argparse, json, pickle, time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist

from task_shape_content_clusters import CACHE_DIR, STEPS, open_memmap
from sample_by_task_centroids import DEFAULT_STEM, build_task_centroids

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DIR = Path(__file__).resolve().parent


def build_families(threshold):
    """Cluster task z-curves by mean|Δz| shape distance. Returns
    (family_names list, F (n_fam, 11) z-normed family curves, {fam: [tasks]})."""
    cents = build_task_centroids(DEFAULT_STEM, len(STEPS))   # {task: z-curve(11,)}
    names = sorted(cents)
    Z = np.vstack([cents[t] for t in names]).astype(np.float64)
    D = squareform(pdist(Z, metric="euclidean")) / np.sqrt(Z.shape[1])   # mean|Δz|
    np.fill_diagonal(D, 0)
    L = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(L, t=threshold, criterion="distance")
    groups = {}
    for t, c in zip(names, labels):
        groups.setdefault(c, []).append(t)
    fam_names, fam_curves, members = [], [], {}
    for rank, c in enumerate(sorted(groups, key=lambda c: -len(groups[c])), 1):
        ms = groups[c]
        z = np.mean([cents[m] for m in ms], axis=0)
        z = (z - z.mean()) / z.std()
        key = "+".join(ms) if len(ms) <= 3 else f"fam{rank}_{ms[0]}_n{len(ms)}"
        fam_names.append(key); fam_curves.append(z); members[key] = ms
    return fam_names, np.array(fam_curves), members


def assign_families(F, chunk=2_000_000):
    """Assign every valid token to its nearest family by mean|Δz|. Cached per family set."""
    tag = f"{F.shape[0]}fam_{hash(F.tobytes()) & 0xffffff:06x}"
    cache = CACHE_DIR / f"family_assign_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"loaded cached assignment from {cache}")
        return z["assign"], z["dist"]
    mm, edim, n_out = open_memmap()
    N = mm.shape[0]
    assign = np.full(N, -1, np.int8)
    dist = np.full(N, np.inf, np.float32)
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -n_out:], dtype=np.float32)
        mu = b.mean(1, keepdims=True); sd = b.std(1, keepdims=True)
        z = (b - mu) / (sd + 1e-9)
        # mean |z - fam| per checkpoint -> (rows, n_fam)
        d = np.abs(z[:, None, :] - F[None, :, :]).mean(2)
        a = d.argmin(1)
        assign[s:e] = a.astype(np.int8)
        dist[s:e] = d[np.arange(e - s), a]
        assign[s:e][sd[:, 0] <= 1e-6] = -1
        print(f"  {e:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(cache, assign=assign, dist=dist)
    print(f"cached -> {cache}")
    return assign, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="euclidean (mean|Δz|) cut for merging tasks into a family")
    ap.add_argument("--per-family", type=int, default=0,
                    help="tokens per family; 0 = auto (smallest family's pool)")
    ap.add_argument("--families-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fam_names, F, members = build_families(a.threshold)
    print(f"\n{len(fam_names)} shape families at euclid cut {a.threshold}:")
    for i, k in enumerate(fam_names):
        spark = " ".join(f"{v:+.1f}" for v in F[i])
        print(f"  [{i}] {k}\n        curve z: {spark}\n        members: {', '.join(members[k])}")
    if a.families_only:
        return

    assign, dist = assign_families(F)
    counts = np.bincount(assign[assign >= 0], minlength=len(fam_names))
    print("\ntokens assigned per family (nearest by shape):")
    for i, k in enumerate(fam_names):
        print(f"  [{i}] {k:<40} {counts[i]:>12,}  ({100*counts[i]/counts.sum():.1f}%)")

    per = a.per_family or int(counts.min())
    print(f"\nper-family n = {per:,}"
          + (f" (auto: smallest family)" if not a.per_family else ""))
    rng = np.random.default_rng(a.seed)
    sel = []
    for i in range(len(fam_names)):
        pool = np.where(assign == i)[0]
        take = min(per, len(pool))
        sel.append((i, rng.choice(pool, take, replace=False) if len(pool) > take else pool))

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    rows = np.concatenate([s for _, s in sel])
    fam_of = np.concatenate([np.full(len(s), i) for i, s in sel])
    perm = rng.permutation(len(rows))
    rows, fam_of = rows[perm], fam_of[perm]
    out = Path(a.out) if a.out else OUT_DIR / f"sae_sample_by_family_n{per}.jsonl"
    with open(out, "w") as f:
        for r, fi in zip(rows.tolist(), fam_of.tolist()):
            f.write(json.dumps({"word_id": str(wids[r]), "family": fam_names[fi],
                                "weight": 1.0}) + "\n")
    print(f"\nWROTE {len(rows):,} tokens ({len(fam_names)} families x {per:,}) -> {out}")


if __name__ == "__main__":
    main()
