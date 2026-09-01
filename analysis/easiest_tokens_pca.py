"""Take the EASIEST-to-classify tokens (correctly predicted, highest confidence) for each
family, then run PCA on just that subset's embeddings to find the axes of maximum natural
variation WITHIN the confident group — a more targeted question than raw-corpus PCA (too
generic) or the single classifier weight direction (only gives one axis, no internal
substructure). For each top PC: extreme tokens with context (qualitative read), and whether
the PC just re-derives family identity or cuts across families (more interesting).

Usage:
  python easiest_tokens_pca.py --n-easy 3000 --n-pcs 6 --n-extreme 15
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
    ap.add_argument("--n-easy", type=int, default=3000, help="easiest tokens per family")
    ap.add_argument("--n-pcs", type=int, default=6)
    ap.add_argument("--n-extreme", type=int, default=15, help="tokens shown per PC tail")
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

    print(f"training logistic regression on {len(y_tr):,} rows...")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    classes = list(clf.classes_)

    # score EVERYTHING we have (train+test), find easiest-to-classify per family
    X_all = np.concatenate([X_tr, X_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)
    rows_all = rows[np.concatenate([idx_tr, idx_te])]
    proba_all = clf.predict_proba(X_all)

    easy_idx = []
    for f in sorted(set(y_all.tolist())):
        col = classes.index(f)
        fam_mask = y_all == f
        correct = proba_all.argmax(1) == col
        cand = np.where(fam_mask & correct)[0]
        top = cand[np.argsort(-proba_all[cand, col])[: a.n_easy]]
        print(f"  family {f} ({FAM_LABELS.get(f)}): {len(cand):,} correctly-classified, "
              f"took top {len(top):,} by confidence "
              f"(p range {proba_all[top, col].min():.3f}-{proba_all[top, col].max():.3f})")
        easy_idx.append(top)
    easy_idx = np.concatenate(easy_idx)
    X_easy = X_all[easy_idx]
    y_easy = y_all[easy_idx]
    rows_easy = rows_all[easy_idx]
    print(f"\ntotal easiest-to-classify tokens: {len(easy_idx):,}")

    print(f"running PCA (n_components={a.n_pcs})...")
    pca = PCA(n_components=a.n_pcs, random_state=a.seed)
    proj = pca.fit_transform(X_easy)  # (n_easy_total, n_pcs)
    print(f"explained variance ratio: " +
          ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(pca.explained_variance_ratio_)) +
          f"  (cumulative={pca.explained_variance_ratio_.sum():.3f})")

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))

    for pc in range(a.n_pcs):
        scores = proj[:, pc]
        order = np.argsort(scores)
        lo = order[: a.n_extreme]
        hi = order[-a.n_extreme:][::-1]

        # does this PC just re-derive family identity, or cut across families?
        fam_means = {f: float(scores[y_easy == f].mean()) for f in sorted(set(y_easy.tolist()))}
        print(f"\n=== PC{pc+1} (explained var {pca.explained_variance_ratio_[pc]:.3f}) ===")
        print(f"  per-family mean score: " +
              ", ".join(f"fam{f}={m:+.2f}" for f, m in fam_means.items()))

        need_wids = [str(w) for w in wids[rows_easy[np.concatenate([hi, lo])]]]
        wic = get_words_in_context(INPUT_FEATURES, need_wids, N=10)

        print(f"  --- high end ---")
        for i in hi:
            wid = str(wids[rows_easy[i]])
            e = wic.get(wid, {"before": "?", "word": "?", "after": "?"})
            print(f"    [{scores[i]:+.2f}, fam{y_easy[i]}] ...{e['before']}[[{e['word']}]]{e['after']}...")
        print(f"  --- low end ---")
        for i in lo:
            wid = str(wids[rows_easy[i]])
            e = wic.get(wid, {"before": "?", "word": "?", "after": "?"})
            print(f"    [{scores[i]:+.2f}, fam{y_easy[i]}] ...{e['before']}[[{e['word']}]]{e['after']}...")


if __name__ == "__main__":
    main()
