"""Can the shape-FAMILY a token was assigned to (by L1 trajectory distance, see
build_family_subset.py::assign_families) be predicted from its 768-d Longformer content
embedding? A supervised complement to the earlier KMeans-based decoupling checks: trains
a linear probe (family label -> embedding) and compares real accuracy to a label-shuffled
null, per this project's standard null-model discipline.

Usage:
  python predict_family_from_embedding.py --assign-cache family_assign_4fam_c4df2b.npz \
      --per-family 40000
"""
import argparse, gc, time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from task_shape_content_clusters import CACHE_DIR, open_memmap

FAM_LABELS = {0: "fam0_late_reasoning(68k)", 1: "fam1_early_risers",
              2: "fam2_blimp", 3: "fam3_medmcqa"}


def load_assignment(name):
    d = np.load(CACHE_DIR / name)
    return d["assign"]


def sample_rows(assign, per_family, seed):
    rng = np.random.default_rng(seed)
    fams = sorted(set(assign[assign >= 0].tolist()))
    rows, labels = [], []
    for f in fams:
        pool = np.where(assign == f)[0]
        take = min(per_family, len(pool))
        sel = rng.choice(pool, take, replace=False)
        rows.append(sel)
        labels.append(np.full(take, f))
        print(f"  family {f} ({FAM_LABELS.get(f, f)}): pool={len(pool):,} sampled={take:,}")
    rows = np.concatenate(rows)
    labels = np.concatenate(labels)
    order = np.argsort(rows)  # sorted for memmap locality
    return rows[order], labels[order]


def read_embeddings(rows, edim, chunk=2_000_000):
    """Gather embeddings for `rows` with ONE sequential pass over the memmap (like
    assign_families/corr_all), instead of scattered fancy-indexing. Fancy-indexing
    (mm[rows, :edim]) does one random I/O per row (~900 rows/s observed on this
    filesystem); a full sequential scan reads the whole 57GB file in ~2-5 min regardless
    of how many rows you keep, since row-major storage means a contiguous row-range read
    is bandwidth-bound, not latency-bound. Rows are assumed pre-sorted (sample_rows sorts
    them) so the searchsorted window walk below is O(N) total, not O(N log N)."""
    mm, _, n_out = open_memmap()
    N = mm.shape[0]
    E = np.empty((len(rows), edim), np.float32)
    t = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        lo = np.searchsorted(rows, s, side="left")
        hi = np.searchsorted(rows, e, side="left")
        if hi > lo:
            block = np.asarray(mm[s:e, :edim], dtype=np.float32)  # contiguous, sequential
            E[lo:hi] = block[rows[lo:hi] - s]
        print(f"  scanned {e:,}/{N:,} ({time.time()-t:.0f}s), matched {hi:,}/{len(rows):,}",
              flush=True)
    return E


def fit_eval(X_tr, y_tr, X_te, y_te, tag):
    """Fit+eval only — X_tr/X_te are assumed already scaled and are NOT copied here, so
    REAL and NULL runs can share one scaled copy instead of each allocating their own."""
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    bacc = balanced_accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print(f"[{tag}] accuracy={acc:.4f}  balanced_accuracy={bacc:.4f}")
    del clf, pred
    gc.collect()
    return acc, bacc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assign-cache", default="family_assign_4fam_c4df2b.npz")
    ap.add_argument("--per-family", type=int, default=3_800_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    assign = load_assignment(a.assign_cache)
    print(f"loaded assignment: {assign.shape[0]:,} tokens")
    rows, labels = sample_rows(assign, a.per_family, a.seed)
    print(f"sampled {len(rows):,} tokens total; reading embeddings...")
    X = read_embeddings(rows, edim=768)

    # Split on INDICES first, then materialize X_tr/X_te and drop X immediately —
    # holding X + X_tr + X_te simultaneously is what OOM'd the first run at 3.8M/family
    # (X alone is ~47GB at this scale; every extra full-size copy matters).
    idx = np.arange(len(labels))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=a.seed, stratify=labels)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = labels[idx_tr], labels[idx_te]
    del X, idx, idx_tr, idx_te
    gc.collect()
    print(f"\ntrain={len(y_tr):,} test={len(y_te):,}  n_classes={len(set(labels.tolist()))}")

    # Scale ONCE, in place (copy=False avoids allocating a second full-size array),
    # and reuse the same scaled X_tr/X_te for both the REAL and NULL fits below —
    # the previous version re-scaled from scratch inside each call, doubling peak memory.
    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    acc, bacc, cm = fit_eval(X_tr, y_tr, X_te, y_te, "REAL")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    rng = np.random.default_rng(a.seed + 1)
    y_tr_shuf = rng.permutation(y_tr)
    acc0, bacc0, _ = fit_eval(X_tr, y_tr_shuf, X_te, y_te, "NULL(shuffled-train-labels)")

    chance = 1.0 / len(set(labels.tolist()))
    print(f"\nchance level (balanced classes): {chance:.4f}")
    print(f"REAL balanced_accuracy={bacc:.4f}  NULL balanced_accuracy={bacc0:.4f}  "
          f"chance={chance:.4f}")


if __name__ == "__main__":
    main()
