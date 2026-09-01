"""Re-run predict_family_from_embedding.py, but restricted to EXCLUSIVE high-confidence
family members: a token is only kept if it clears (r > --min-r) AND (L1 dist < --max-l1)
for EXACTLY ONE of the (non-excluded) families. This fixes a gap in the earlier version:
build_family_subset.py's L1-nearest assignment always picks exactly one family (argmin over
4), then the old highconf filter only checked r/dist against THAT assigned family — so a
token could pass the filter while ALSO correlating well with a different family (this is
expected to matter most between fam0/fam1, which per the project's Kendall-tau findings
share the same generic "gradual rise" macro-shape). Tokens clearing the filter for 0 or 2+
families are discarded entirely, not just relabeled.

Since both the token z-curve and each family z-curve are already zero-mean/unit-std,
Pearson r reduces to a plain dot product: r = mean(z_token * z_family) — cheap enough to
compute against ALL families in one more sequential memmap pass.

Usage:
  python predict_family_from_embedding_highconf.py --min-r 0.7 --max-l1 0.4 --exclude-family 3
"""
import argparse, gc, time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from task_shape_content_clusters import CACHE_DIR, open_memmap
from build_family_subset import build_families

FAM_LABELS = {0: "fam0_late_reasoning(68k)", 1: "fam1_early_risers",
              2: "fam2_blimp", 3: "fam3_medmcqa"}


def compute_r_dist_all(F, chunk=2_000_000):
    """R[N,nfam], D[N,nfam], valid[N] — Pearson r and L1 (mean|Δz|) dist of EVERY token
    against EVERY family curve (not just its argmin-assigned one). One sequential pass;
    cached since it touches the full 57GB."""
    nfam = F.shape[0]
    cache = CACHE_DIR / f"family_rdist_all_{nfam}fam.npz"
    if cache.exists():
        print(f"loaded cached r/dist-all from {cache}")
        d = np.load(cache)
        return d["R"], d["D"], d["valid"]
    mm, edim, n_out = open_memmap()
    N = mm.shape[0]
    R = np.empty((N, nfam), np.float32)
    D = np.empty((N, nfam), np.float32)
    valid = np.zeros(N, bool)
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -n_out:], dtype=np.float32)
        mu = b.mean(1, keepdims=True); sd = b.std(1, keepdims=True)
        z = (b - mu) / (sd + 1e-9)
        valid[s:e] = sd[:, 0] > 1e-6
        R[s:e] = (z @ F.T) / F.shape[1]
        D[s:e] = np.abs(z[:, None, :] - F[None, :, :]).mean(2)
        print(f"  rdist-scan {e:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(cache, R=R, D=D, valid=valid)
    print(f"cached -> {cache}")
    return R, D, valid


def exclusive_labels(R, D, valid, min_r, max_l1, exclude=()):
    """A token counts for family i only if it clears the filter for i AND NO other family.
    Returns (mask of kept rows, label per row — only meaningful where mask is True)."""
    clears = R > min_r
    if max_l1 is not None:
        clears &= (D < max_l1)
    clears &= valid[:, None]
    for f in exclude:
        clears[:, f] = False
    n_clear = clears.sum(1)
    mask = n_clear == 1
    labels = clears.argmax(1)  # only valid where mask True
    return mask, labels


def sample_equal_exclusive(mask, labels, cap, seed, fam_names):
    rng = np.random.default_rng(seed)
    fams = sorted(set(labels[mask].tolist()))
    pools = {}
    for f in fams:
        pool = np.where(mask & (labels == f))[0]
        pools[f] = pool
        print(f"  family {f} ({FAM_LABELS.get(f, f)}): EXCLUSIVE high-conf pool={len(pool):,}")
    n = min(cap, min(len(p) for p in pools.values()))
    print(f"\nequal sample size per family: {n:,} (capped by smallest exclusive pool"
          f" and --per-family={cap:,})")
    rows, out_labels = [], []
    for f, pool in pools.items():
        sel = rng.choice(pool, n, replace=False)
        rows.append(sel)
        out_labels.append(np.full(n, f))
    rows = np.concatenate(rows)
    out_labels = np.concatenate(out_labels)
    order = np.argsort(rows)
    return rows[order], out_labels[order]


def read_embeddings(rows, edim, chunk=2_000_000):
    mm, _, n_out = open_memmap()
    N = mm.shape[0]
    E = np.empty((len(rows), edim), np.float32)
    t = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        lo = np.searchsorted(rows, s, side="left")
        hi = np.searchsorted(rows, e, side="left")
        if hi > lo:
            block = np.asarray(mm[s:e, :edim], dtype=np.float32)
            E[lo:hi] = block[rows[lo:hi] - s]
        print(f"  emb-scan {e:,}/{N:,} ({time.time()-t:.0f}s), matched {hi:,}/{len(rows):,}",
              flush=True)
    return E


MODELS = {
    "logreg": lambda seed: LogisticRegression(max_iter=2000, C=1.0),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=seed),
}


def fit_eval(model_name, seed, X_tr, y_tr, X_te, y_te, tag):
    clf = MODELS[model_name](seed)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    bacc = balanced_accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print(f"[{model_name}/{tag}] accuracy={acc:.4f}  balanced_accuracy={bacc:.4f}  "
          f"({time.time()-t0:.0f}s)")
    del clf, pred
    gc.collect()
    return acc, bacc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-r", type=float, default=0.85)
    ap.add_argument("--max-l1", type=float, default=0.3)
    ap.add_argument("--per-family", type=int, default=548_236,
                     help="upper cap; actual n is min(this, smallest exclusive pool)")
    ap.add_argument("--models", nargs="*", default=["logreg", "rf"], choices=list(MODELS))
    ap.add_argument("--exclude-family", type=int, nargs="*", default=[],
                     help="family indices to drop entirely, e.g. --exclude-family 3 (medmcqa)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fam_names, F, members = build_families(0.45)
    print(f"{len(fam_names)} families: " +
          ", ".join(f"[{i}] {FAM_LABELS.get(i,i)}" for i in range(len(fam_names))))
    if a.exclude_family:
        excl = ", ".join(f"[{f}] {FAM_LABELS.get(f, f)}" for f in a.exclude_family)
        print(f"excluding: {excl}")

    R, D, valid = compute_r_dist_all(F)

    print(f"\nfiltering to EXCLUSIVE r>{a.min_r} & dist<{a.max_l1} (0 or 2+ matches discarded):")
    mask, labels = exclusive_labels(R, D, valid, a.min_r, a.max_l1, exclude=set(a.exclude_family))
    n_zero = ((~mask) & (labels >= 0)).sum()  # informational only, includes ambiguous+unmatched
    print(f"  total valid tokens: {valid.sum():,}  exclusive matches: {mask.sum():,} "
          f"({100*mask.sum()/valid.sum():.1f}%)")

    rows, labels_sel = sample_equal_exclusive(mask, labels, a.per_family, a.seed, fam_names)
    print(f"\nsampled {len(rows):,} tokens total (exclusive, high-confidence, balanced); "
          f"reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=a.seed, stratify=labels_sel)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = labels_sel[idx_tr], labels_sel[idx_te]
    del X, idx, idx_tr, idx_te
    gc.collect()
    print(f"\ntrain={len(y_tr):,} test={len(y_te):,}  n_classes={len(set(labels_sel.tolist()))}")

    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    rng = np.random.default_rng(a.seed + 1)
    y_tr_shuf = rng.permutation(y_tr)
    chance = 1.0 / len(set(labels_sel.tolist()))

    results = {}
    for model_name in a.models:
        acc, bacc, cm = fit_eval(model_name, a.seed, X_tr, y_tr, X_te, y_te, "REAL")
        print(f"[{model_name}/REAL] confusion matrix (rows=true, cols=pred):")
        print(cm)
        acc0, bacc0, _ = fit_eval(model_name, a.seed, X_tr, y_tr_shuf, X_te, y_te,
                                   "NULL(shuffled-train-labels)")
        results[model_name] = (bacc, bacc0)

    print(f"\nchance level (balanced classes): {chance:.4f}")
    for model_name, (bacc, bacc0) in results.items():
        print(f"[{model_name}] REAL balanced_accuracy={bacc:.4f}  "
              f"NULL balanced_accuracy={bacc0:.4f}  chance={chance:.4f}")


if __name__ == "__main__":
    main()
