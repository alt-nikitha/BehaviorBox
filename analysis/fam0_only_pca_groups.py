"""PCA restricted to ONLY fam0's easiest-to-classify tokens (not combined with fam1/fam2)
so "which family" can't be the dominant source of variance — the components should instead
reveal internal sub-structure WITHIN what the classifier considers confidently fam0.

For each top PC, tokens are binned into N quantile groups along that axis (not just the two
extremes) and a sample is shown from each bin, giving the full low-to-high progression rather
than just the tails.

Usage:
  python fam0_only_pca_groups.py --n-easy 5000 --n-pcs 6 --n-bins 5 --per-bin 8
"""
import argparse, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

WORD_IDS = CACHE_DIR / "word_ids.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-r", type=float, default=0.85)
    ap.add_argument("--max-l1", type=float, default=0.3)
    ap.add_argument("--per-family", type=int, default=548_236)
    ap.add_argument("--exclude-family", type=int, nargs="*", default=[3])
    ap.add_argument("--target-family", type=int, default=0)
    ap.add_argument("--n-easy", type=int, default=5000, help="easiest tokens for target family only")
    ap.add_argument("--n-pcs", type=int, default=6)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    fam_names, F, members = build_families(0.45)
    R, D, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D, valid, a.min_r, a.max_l1, exclude=set(a.exclude_family))
    rows, labels_sel = sample_equal_exclusive(mask, labels, a.per_family, a.seed, fam_names)
    print(f"sampled {len(rows):,} tokens total; reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=a.seed, stratify=labels_sel)
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = labels_sel[idx_tr], labels_sel[idx_te]

    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    print(f"training logistic regression on {len(y_tr):,} rows (all families, for ranking)...")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    classes = list(clf.classes_)
    col = classes.index(a.target_family)

    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    rows_all = rows[np.concatenate([idx_tr, idx_te])]
    proba = clf.predict_proba(X_all)[:, col]

    fam_mask = y_all == a.target_family
    correct = proba > 0.5  # for this column, >0.5 implies argmax since it's the true class here
    cand = np.where(fam_mask)[0]
    top = cand[np.argsort(-proba[cand])[: a.n_easy]]
    print(f"fam{a.target_family} ({FAM_LABELS.get(a.target_family)}): "
          f"{len(cand):,} true members, took easiest {len(top):,} "
          f"(p range {proba[top].min():.3f}-{proba[top].max():.3f})")

    X_fam0 = X_all[top]
    rows_fam0 = rows_all[top]

    print(f"running PCA on ONLY fam{a.target_family} tokens (n_components={a.n_pcs})...")
    pca = PCA(n_components=a.n_pcs, random_state=a.seed)
    proj = pca.fit_transform(X_fam0)
    print(f"explained variance ratio: " +
          ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(pca.explained_variance_ratio_)) +
          f"  (cumulative={pca.explained_variance_ratio_.sum():.3f})")

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    rng = np.random.default_rng(a.seed)

    n_ex = a.per_bin * a.n_bins  # keep total count comparable to the old binned view
    for pc in range(a.n_pcs):
        scores = proj[:, pc]
        order = np.argsort(scores)  # ascending: order[0] = most negative
        lo = order[:n_ex]                 # TRUE lowest, sorted most-negative-first
        hi = order[-n_ex:][::-1]          # TRUE highest, sorted most-positive-first
        print(f"\n=== PC{pc+1} (explained var {pca.explained_variance_ratio_[pc]:.3f}) "
              f"range [{scores.min():.1f}, {scores.max():.1f}] ===")

        need = np.concatenate([lo, hi])
        need_wids = [str(w) for w in wids[rows_fam0[need]]]
        wic = get_words_in_context(INPUT_FEATURES, need_wids, N=10)

        print(f"  --- LOW (most negative, n={len(lo)}) ---")
        for i in lo:
            wid = str(wids[rows_fam0[i]])
            e = wic.get(wid, {"before": "?", "word": "?", "after": "?"})
            print(f"    [{scores[i]:+.2f}] ...{e['before']}[[{e['word']}]]{e['after']}...")
        print(f"  --- HIGH (most positive, n={len(hi)}) ---")
        for i in hi:
            wid = str(wids[rows_fam0[i]])
            e = wic.get(wid, {"before": "?", "word": "?", "after": "?"})
            print(f"    [{scores[i]:+.2f}] ...{e['before']}[[{e['word']}]]{e['after']}...")


if __name__ == "__main__":
    main()
