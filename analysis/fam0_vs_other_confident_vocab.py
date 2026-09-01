"""Fully automatic (no hand-built concept lists): find what distinguishes tokens the logreg
confidently and CORRECTLY calls fam0 from tokens it confidently and CORRECTLY calls NOT fam0
(true label fam1 or fam2, predicted correctly with high confidence) -- via differential
vocabulary enrichment (log-odds ratio with smoothing) over their context windows, not
hypothesized concepts.

Group A: true_label==0, predicted==0, ranked by p(fam0) descending.
Group B: true_label!=0, predicted==true_label, ranked by p(true_label) descending.
Both groups are equally confident and equally correct -- so whatever differs in their
vocabulary is plausibly what the classifier's fam0 direction actually tracks, without
assuming in advance what that is.

Loads the saved clf+scaler from logreg_weight_projection.py --save-model (no retraining),
reproduces the same exclusive sample/split, tokenizes context windows (target word + before
+ after) into lowercased word tokens, and ranks terms by add-k-smoothed log-odds ratio between
group A and group B frequencies.

Usage:
  python fam0_vs_other_confident_vocab.py --model fam0_logreg_weights.pkl --n-group 5000 --top-n 40
"""
import argparse, pickle, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

WORD_IDS = CACHE_DIR / "word_ids.pkl"
TOKEN_RE = re.compile(r"\w+|[^\w\s]")  # words OR single punctuation/symbol chars


def tokenize_window(e, include_target=True):
    text = e["before"] + (" " + e["word"] + " " if include_target else " ") + e["after"]
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_vocab_counts(word_ids, ctx):
    counts = Counter()
    total = 0
    term_to_wids = {}
    for wid in word_ids:
        e = ctx.get(wid, {"before": "", "word": "", "after": ""})
        toks = tokenize_window(e)
        counts.update(toks)
        total += len(toks)
        for t in set(toks):
            term_to_wids.setdefault(t, []).append(wid)
    return counts, total, term_to_wids


def print_example_contexts(term, wids_for_term, ctx, n=4):
    for wid in wids_for_term[:n]:
        e = ctx.get(wid, {"before": "?", "word": "?", "after": "?"})
        print(f"        ...{e['before']}[[{e['word']}]]{e['after']}...")


def log_odds_ratio(countA, totalA, countB, totalB, k=0.5):
    vocab = set(countA) | set(countB)
    rows = []
    for term in vocab:
        a, b = countA.get(term, 0), countB.get(term, 0)
        lor = (np.log(a + k) - np.log(totalA - a + k)) - (np.log(b + k) - np.log(totalB - b + k))
        rows.append((term, lor, a, b))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--n-group", type=int, default=5000,
                     help="tokens per group (most-confident-correct of each)")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--n-ctx", type=int, default=4, help="example contexts to print per term")
    ap.add_argument("--min-count", type=int, default=10,
                     help="ignore terms appearing fewer than this many times total (noise)")
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}")

    fam_names, F, members = build_families(0.45)
    R, D, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D, valid, sargs["min_r"], sargs["max_l1"],
                                     exclude=set(sargs["exclude_family"]))
    rows, labels_sel = sample_equal_exclusive(mask, labels, sargs["per_family"], sargs["seed"], fam_names)
    print(f"sampled {len(rows):,} tokens total (reproducing saved split); reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=sargs["seed"], stratify=labels_sel)
    X_te = scaler.transform(X[idx_te])
    y_te = labels_sel[idx_te]
    rows_te = rows[idx_te]

    proba = clf.predict_proba(X_te)  # columns ordered per `classes`
    pred = np.array(classes)[np.argmax(proba, axis=1)]

    # Group A: confidently + correctly fam0
    col0 = classes.index(0)
    maskA = (y_te == 0) & (pred == 0)
    candA = np.where(maskA)[0]
    candA = candA[np.argsort(-proba[candA, col0])][: a.n_group]

    # Group B: confidently + correctly NOT fam0 (true fam1 or fam2, correctly predicted)
    maskB = (y_te != 0) & (pred == y_te)
    candB = np.where(maskB)[0]
    conf_b = proba[candB, [classes.index(t) for t in y_te[candB]]]
    candB = candB[np.argsort(-conf_b)][: a.n_group]
    print(f"group A (confident+correct fam0): {len(candA):,} "
          f"(p range {proba[candA, col0].min():.3f}-{proba[candA, col0].max():.3f})")
    fam_mix_B = {f: int((y_te[candB] == f).sum()) for f in sorted(set(y_te[candB].tolist()))}
    print(f"group B (confident+correct NOT fam0): {len(candB):,}  composition={fam_mix_B}")

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    widsA = [str(wids_all[r]) for r in rows_te[candA]]
    widsB = [str(wids_all[r]) for r in rows_te[candB]]
    print(f"fetching context for {len(widsA) + len(widsB):,} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, widsA + widsB, N=10)

    countA, totalA, termA_wids = build_vocab_counts(widsA, ctx)
    countB, totalB, termB_wids = build_vocab_counts(widsB, ctx)
    print(f"\nvocab sizes: A={len(countA):,} tokens ({totalA:,} total)  "
          f"B={len(countB):,} tokens ({totalB:,} total)")

    rows_lor = log_odds_ratio(countA, totalA, countB, totalB)
    rows_lor = [r for r in rows_lor if (r[2] + r[3]) >= a.min_count]
    rows_lor.sort(key=lambda t: -t[1])

    def doc_id(wid):
        return "_".join(wid.split("_")[:-1])

    def show(rows_side, wids_index, top_n, n_ctx):
        for term, lor, a_, b_ in rows_side[:top_n]:
            wids_for_term = wids_index.get(term, [])
            n_docs = len(set(doc_id(w) for w in wids_for_term))
            print(f"\n  {term!r}  lor={lor:+.3f}  countA={a_:<6} countB={b_:<6} "
                  f"(appears in {n_docs} distinct docs out of {len(wids_for_term)} hits)")
            print_example_contexts(term, wids_for_term, ctx, n=n_ctx)

    print(f"\n=== top {a.top_n} terms enriched in confident-CORRECT fam0 (vs confident-correct "
          f"non-fam0) ===")
    show(rows_lor, termA_wids, a.top_n, a.n_ctx)

    print(f"\n\n=== top {a.top_n} terms enriched in confident-CORRECT non-fam0 (vs confident-"
          f"correct fam0) ===")
    show(rows_lor[::-1], termB_wids, a.top_n, a.n_ctx)


if __name__ == "__main__":
    main()
