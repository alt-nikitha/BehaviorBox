"""Instead of PCA (which finds max-variance directions, not necessarily the direction the
classifier uses), project tokens directly onto the TRAINED logreg's weight vector for the
target family. This is the actual discriminative axis, guaranteed to separate the classes
(unlike an arbitrary principal component that may be orthogonal to it).

Reproduces the exact same exclusive high-confidence sample + train/test split + logistic
regression as predict_family_from_embedding_highconf.py / logreg_easiest_fam0_tokens.py
(r>0.85, L1<0.3, medmcqa excluded, seed=42 defaults). On the held-out TEST set, restricts to
true target-family tokens, projects their (scaled) embeddings onto the one-vs-rest weight
vector, and:
  1. prints context for the low/high extremes along that axis (like fam0_only_pca_groups.py's
     binned view, but for the ACTUAL decision direction instead of a PC).
  2. correlates the projection score against simple structural features (token length,
     position in document, function-word flag, adjacent punctuation) to test whether the
     axis is structural rather than semantic.
  3. reports what fraction of total embedding variance the direction explains, as a sanity
     check on whether this is a dominant or a thin/marginal axis.

Usage:
  python logreg_weight_projection.py --n-ex 5000 --n-bins 5 --per-bin 8
"""
import argparse, pickle, re, sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
PUNCT_RE = re.compile(r"[.,;:!?\"'()\[\]]")


def structural_features(word_ids, ctx):
    """Simple structural features per token, derived from word_id + surrounding context.
    Returns dict of name -> np.array (float), same order as word_ids."""
    length = np.array([len(ctx.get(wid, {}).get("word", "")) for wid in word_ids], float)
    pos = np.array([int(wid.rsplit("_", 1)[1]) for wid in word_ids], float)
    is_func = np.array([
        ctx.get(wid, {}).get("word", "").strip().lower() in ENGLISH_STOP_WORDS for wid in word_ids
    ], float)
    prev_punct = np.array([
        bool(PUNCT_RE.search(ctx.get(wid, {}).get("before", "")[-1:])) for wid in word_ids
    ], float)
    next_punct = np.array([
        bool(PUNCT_RE.search(ctx.get(wid, {}).get("after", "")[:1])) for wid in word_ids
    ], float)
    return {
        "token_length": length,
        "doc_position": pos,
        "is_function_word": is_func,
        "preceded_by_punct": prev_punct,
        "followed_by_punct": next_punct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-r", type=float, default=0.85)
    ap.add_argument("--max-l1", type=float, default=0.3)
    ap.add_argument("--per-family", type=int, default=548_236)
    ap.add_argument("--exclude-family", type=int, nargs="*", default=[3])
    ap.add_argument("--target-family", type=int, default=0)
    ap.add_argument("--n-ex", type=int, default=5000, help="how many true target-family "
                     "test tokens to analyze structurally")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", default="fam0_logreg_weights.pkl",
                     help="pickle clf+scaler+classes here for reuse without retraining; "
                          "empty string to skip")
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
    rows_te = rows[idx_te]

    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    print(f"training logistic regression on {len(y_tr):,} rows...")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    classes = list(clf.classes_)
    col = classes.index(a.target_family)
    w = clf.coef_[col]              # the actual one-vs-rest discriminative direction
    w_unit = w / np.linalg.norm(w)

    if a.save_model:
        with open(a.save_model, "wb") as f:
            pickle.dump({
                "clf": clf, "scaler": scaler, "classes": classes,
                "sample_args": dict(min_r=a.min_r, max_l1=a.max_l1, per_family=a.per_family,
                                     exclude_family=a.exclude_family, seed=a.seed),
            }, f)
        print(f"saved trained clf+scaler -> {a.save_model}")

    # --- variance sanity check: how dominant is this direction vs total embedding variance ---
    total_var = np.var(X_te, axis=0).sum()
    proj_all = X_te @ w_unit
    dir_var = np.var(proj_all)
    print(f"\nweight-direction variance / total embedding variance: "
          f"{dir_var:.3f} / {total_var:.3f} = {dir_var/total_var:.5f}")

    # --- restrict to true target-family test tokens, project onto raw (unnormalized) w ---
    true_mask = y_te == a.target_family
    print(f"test tokens with true label = fam{a.target_family} "
          f"({FAM_LABELS.get(a.target_family)}): {true_mask.sum():,}")
    cand = np.where(true_mask)[0]
    if a.n_ex < len(cand):
        rng = np.random.default_rng(a.seed)
        cand = rng.choice(cand, a.n_ex, replace=False)
    scores = X_te[cand] @ w  # same quantity that feeds the sigmoid (minus intercept)
    rows_cand = rows_te[cand]

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    word_ids = [str(wids_all[r]) for r in rows_cand]
    print(f"fetching context for {len(word_ids)} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, word_ids, N=10)

    # --- print low/high extremes along the discriminative axis, binned like the PCA script ---
    order = np.argsort(scores)
    n_ex = a.per_bin * a.n_bins
    lo, hi = order[:n_ex], order[-n_ex:][::-1]
    print(f"\n=== logreg weight direction for fam{a.target_family}, "
          f"range [{scores.min():.2f}, {scores.max():.2f}] ===")
    print(f"  --- LOW (least fam{a.target_family}-like among true members, n={len(lo)}) ---")
    for i in lo:
        wid = word_ids[i]
        e = ctx.get(wid, {"before": "?", "word": "?", "after": "?"})
        print(f"    [{scores[i]:+.2f}] ...{e['before']}[[{e['word']}]]{e['after']}...")
    print(f"  --- HIGH (most fam{a.target_family}-like, n={len(hi)}) ---")
    for i in hi:
        wid = word_ids[i]
        e = ctx.get(wid, {"before": "?", "word": "?", "after": "?"})
        print(f"    [{scores[i]:+.2f}] ...{e['before']}[[{e['word']}]]{e['after']}...")

    # --- correlate projection score against structural features ---
    feats = structural_features(word_ids, ctx)
    print(f"\n=== correlation: logreg weight-direction score vs structural features "
          f"(n={len(scores):,}) ===")
    for name, vals in feats.items():
        r, p = pearsonr(scores, vals)
        print(f"  {name:>20s}: r={r:+.4f}  p={p:.2e}  mean={vals.mean():.3f}")


if __name__ == "__main__":
    main()
