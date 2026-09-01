"""Sanity-check build_family_subset.py's L1 (mean|Δz|) family assignment: do tokens
assigned to family i actually have a trajectory close to family i's curve, and is that
closeness MEANINGFUL (family i noticeably closer than the other 3 families), or is "nearest
of 4" trivially satisfied by everything being roughly equidistant?

For a sample of tokens per assigned family:
  1. recompute each token's raw z-curve and its L1 distance to ALL 4 family curves
  2. confirm the recomputed nearest family matches the cached assignment (pipeline check)
  3. report assigned-distance vs distance-to-other-families (margin) vs the family curves'
     own pairwise spacing (the reference scale for what "different" means here)
  4. print a few example token curves per family next to the family curve, and Pearson r

Usage:
  python verify_family_assignment.py --n-per-family 5000
"""
import argparse, pickle, time
from pathlib import Path

import numpy as np

from task_shape_content_clusters import CACHE_DIR, STEPS, open_memmap
from build_family_subset import build_families

WORD_IDS = CACHE_DIR / "word_ids.pkl"


def load_assignment(name):
    d = np.load(CACHE_DIR / name)
    return d["assign"], d["dist"]


def zcurve(b):
    mu = b.mean(1, keepdims=True)
    sd = b.std(1, keepdims=True)
    return (b - mu) / (sd + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-cache", default="family_assign_4fam_c4df2b.npz")
    ap.add_argument("--n-per-family", type=int, default=5000)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fam_names, F, members = build_families(0.45)  # reproduces the 4-family split by content
    print(f"{len(fam_names)} families:")
    for i, k in enumerate(fam_names):
        print(f"  [{i}] {k}  members={members[k]}")

    # reference scale: how far apart are the family curves from EACH OTHER?
    print("\npairwise family-curve L1 distance (mean|Δz| per checkpoint):")
    for i in range(len(fam_names)):
        for j in range(i + 1, len(fam_names)):
            dij = np.abs(F[i] - F[j]).mean()
            print(f"  [{i}]<->[{j}]: {dij:.3f}")

    assign, cached_dist = load_assignment(a.assign_cache)
    mm, edim, n_out = open_memmap()
    rng = np.random.default_rng(a.seed)

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb"))) if WORD_IDS.exists() else None

    for i, name in enumerate(fam_names):
        pool = np.where(assign == i)[0]
        n = min(a.n_per_family, len(pool))
        sel = np.sort(rng.choice(pool, n, replace=False))
        t0 = time.time()
        b = np.asarray(mm[sel, -n_out:], dtype=np.float32)  # fancy-index ok, small n
        z = zcurve(b)  # (n, 11)

        # distance from each sampled token to ALL 4 family curves
        d_all = np.abs(z[:, None, :] - F[None, :, :]).mean(2)  # (n, 4)
        nearest = d_all.argmin(1)
        agree = (nearest == i).mean()

        own_d = d_all[:, i]
        others = d_all.copy()
        others[:, i] = np.inf
        next_best = others.min(1)
        margin = next_best - own_d

        # cross-check against the cached dist from assign_families (should match own_d)
        cached_here = cached_dist[sel]
        cache_diff = np.abs(cached_here - own_d).max()

        # Pearson r between each token curve and its assigned family curve, for intuition
        fz = F[i]
        r = np.array([np.corrcoef(z[k], fz)[0, 1] for k in range(min(n, 2000))])

        print(f"\n=== family[{i}] {name}  (n={n:,}, read {time.time()-t0:.1f}s) ===")
        print(f"  recomputed-nearest agrees with cached assignment: {100*agree:.1f}%"
              f"  (max cached-vs-recomputed dist diff: {cache_diff:.2e}, should be ~0)")
        print(f"  own-family L1 dist:    mean={own_d.mean():.3f} median={np.median(own_d):.3f}")
        print(f"  next-best-family dist: mean={next_best.mean():.3f}")
        print(f"  margin (next_best - own): mean={margin.mean():.3f}  "
              f"frac margin<=0.02 (near-tie): {100*(margin<=0.02).mean():.1f}%")
        print(f"  Pearson r(token curve, family curve): mean={r.mean():.3f} median={np.median(r):.3f}"
              f"  frac r>0.5: {100*(r>0.5).mean():.1f}%  frac r>0.7: {100*(r>0.7).mean():.1f}%"
              f"  frac r<0: {100*(r<0).mean():.1f}%")

        print(f"  family curve:        " + " ".join(f"{v:+.2f}" for v in F[i]))
        idxs = rng.choice(n, min(a.n_examples, n), replace=False)
        for k in idxs:
            tag = str(wids[sel[k]]) if wids is not None else str(sel[k])
            print(f"  tok(d={own_d[k]:.2f} r={np.corrcoef(z[k], fz)[0,1]:+.2f} id={tag}): "
                  + " ".join(f"{v:+.2f}" for v in z[k]))


if __name__ == "__main__":
    main()
