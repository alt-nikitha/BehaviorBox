"""Group the task PERFORMANCE curves themselves by Kendall tau of their checkpoint
orderings -- how many distinct trajectory shapes do the downstream tasks actually
fall into?

tau(curve_i, curve_j) = rank agreement of the two 11-point performance curves. We
build the full task x task tau matrix, hierarchically cluster on distance (1 - tau),
and print the shape families + each task's raw curve. This is the task-side analog of
what the pairwise-sign SAE groups tokens by, and explains why same-shape tasks cannot
get separate features.

Usage:
  python cluster_task_curves_by_tau.py --threshold 0.85
"""
import argparse
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from build_tau_subset_all_tasks import task_sign_matrix
from kendall_shape_match import raw_task_perf, STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="merge tasks into one family while pairwise tau >= this")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep tasks with identical orderings as separate rows")
    a = ap.parse_args()

    tasks, S = task_sign_matrix(dedup=not a.no_dedup)   # (55, n_tasks) sign vectors
    n = len(tasks)
    tau = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x, y = S[:, i], S[:, j]
            nx, ny = (x != 0).sum(), (y != 0).sum()
            tau[i, j] = (x * y).sum() / np.sqrt(nx * ny)

    # hierarchical clustering on 1 - tau
    D = np.clip(1 - tau, 0, None)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1 - a.threshold, criterion="distance")

    perf = {t: raw_task_perf(t.split("+")[0]) for t in tasks}
    fam = {}
    for t, c in zip(tasks, labels):
        fam.setdefault(c, []).append(t)

    print(f"{n} task orderings -> {len(fam)} shape families at tau>={a.threshold}\n")
    order = sorted(fam, key=lambda c: -len(fam[c]))
    for rank, c in enumerate(order, 1):
        members = fam[c]
        idx = [tasks.index(m) for m in members]
        within = tau[np.ix_(idx, idx)]
        wmin = within[np.triu_indices(len(idx), 1)].min() if len(idx) > 1 else 1.0
        print(f"FAMILY {rank} (n={len(members)}, min within-tau={wmin:.2f}):")
        for m in members:
            p = perf[m]
            spark = " ".join(f"{v:.2f}" for v in p)
            # describe the shape: where's the argmin (trough) and how much rise
            trough = STEPS[int(np.argmin(p))]
            print(f"    {m:<34} [{spark}]  trough@{trough}k  Δ={p.max()-p.min():+.2f}")
        print()

    # between-family separation
    print("family mean curves (z-normed), to see the distinct shapes:")
    hdr = "        " + "".join(f"{s}k".rjust(7) for s in STEPS)
    print(hdr)
    for rank, c in enumerate(order, 1):
        M = np.mean([perf[m] for m in fam[c]], axis=0)
        z = (M - M.mean()) / M.std()
        print(f"fam{rank:<4}" + "".join(f"{v:>7.2f}" for v in z))


if __name__ == "__main__":
    main()
