"""Which tokens does the trained logistic regression classify MOST CONFIDENTLY as
fam0 (late-reasoning/68k-jump: arc_challenge, bbh, csqa, gsm8k, mmlu_other,
mmlu_social_sciences, mmlu_stem, naturalqs, winogrande)?

Reproduces the exact same exclusive high-confidence sample + train/test split + logistic
regression used in predict_family_from_embedding_highconf.py (r>0.85, L1<0.3, medmcqa
excluded, seed=42 — same defaults), then on the held-out TEST set, restricts to tokens whose
TRUE label is fam0 and ranks them by the model's predicted probability for fam0. The top of
that ranking is "easiest to classify" — confidently and correctly predicted — which is what
you'd inspect to see what content signal the model actually latched onto.

Usage:
  python logreg_easiest_fam0_tokens.py --top-n 40
"""
import argparse, pickle, sys
from pathlib import Path

import numpy as np
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
    ap.add_argument("--target-family", type=int, default=0, help="which family to inspect")
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-words", type=int, default=10)
    ap.add_argument("--out", default="logreg_easiest_fam0_tokens.txt")
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
    rows_te = rows[idx_te]  # global memmap row index for each test example, same order as X_te

    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    print(f"training logistic regression on {len(y_tr):,} rows...")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    classes = list(clf.classes_)
    tgt_col = classes.index(a.target_family)
    proba = clf.predict_proba(X_te)[:, tgt_col]

    true_mask = y_te == a.target_family
    print(f"test tokens with true label = fam{a.target_family} "
          f"({FAM_LABELS.get(a.target_family)}): {true_mask.sum():,}")

    te_idxs = np.where(true_mask)[0]
    order = te_idxs[np.argsort(-proba[te_idxs])]  # descending confidence
    top = order[: a.top_n]

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    top_word_ids = [str(wids[rows_te[i]]) for i in top]
    print(f"fetching context for {len(top_word_ids)} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, top_word_ids, N=a.context_words)

    lines = [f"Easiest-to-classify fam{a.target_family} ({FAM_LABELS.get(a.target_family)}) "
             f"tokens, by logistic-regression predicted probability", ""]
    for rank, i in enumerate(top, 1):
        wid = str(wids[rows_te[i]])
        p = proba[i]
        c = ctx.get(wid, {"before": "?", "word": "?", "after": "?"})
        lines.append(f"{rank:>3}. p(fam{a.target_family})={p:.4f}  word_id={wid}")
        lines.append(f"     ...{c['before']}[[{c['word']}]]{c['after']}...")
    report = "\n".join(lines)
    print(report)
    Path(a.out).write_text(report)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
