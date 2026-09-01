"""What does the trained logreg's fam0 weight vector actually key on? Loads the saved
clf+scaler (fam0_logreg_weights.pkl, from logreg_weight_projection.py --save-model) instead
of retraining, projects a large sample of TRUE fam0 test tokens onto the weight vector, and
correlates the projection score against several NAMED candidate concepts instead of just
eyeballing extremes:

  - token_length            (weak sanity check, already found r~0.30 on a small sample)
  - is_function_word        (sklearn ENGLISH_STOP_WORDS)
  - code_density            (density of code/markup-like characters in the +/-10-word context
                              window -- the low-extreme dump was dominated by HTML/code/LaTeX)
  - is_html_context         (regex for an HTML/XML tag opener in the context)
  - is_formal_register_word / is_casual_register_word (hardcoded lexicons taken directly from
    the earlier PCA-register-axis finding: shall/pursuant/comprising/whom/throughout/
    subsequently/latter/resorted/hence/whether/thereby/although/unlike/regardless vs
    most/if/some/kind/built/took/passed/sold/closed)

Ranks candidates by |r| so the strongest known correlate of the classifier's actual decision
axis can be named directly, rather than inferred from reading token lists.

Usage:
  python logreg_weight_concept_probe.py --n-sample 20000 --model fam0_logreg_weights.pkl
"""
import argparse, pickle, re, sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

from task_shape_content_clusters import CACHE_DIR, INPUT_FEATURES
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

WORD_IDS = CACHE_DIR / "word_ids.pkl"
CODE_CHARS = re.compile(r"[<>;{}\\$=\[\]|`]")
HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[ >]")
FORMAL_WORDS = {"shall", "pursuant", "comprising", "comprised", "whom", "throughout",
                "subsequently", "latter", "resorted", "hence", "whether", "thereby",
                "although", "unlike", "regardless"}
CASUAL_WORDS = {"most", "if", "some", "kind", "built", "took", "passed", "sold", "closed"}


def concept_features(word_ids, ctx):
    length, is_func, code_dens, is_html, is_formal, is_casual = [], [], [], [], [], []
    for wid in word_ids:
        e = ctx.get(wid, {"before": "", "word": "", "after": ""})
        w = e["word"].strip()
        window = e["before"] + e["after"]
        length.append(len(w))
        is_func.append(w.lower() in ENGLISH_STOP_WORDS)
        code_dens.append(len(CODE_CHARS.findall(window)) / max(len(window), 1))
        is_html.append(bool(HTML_TAG.search(window)))
        is_formal.append(w.lower() in FORMAL_WORDS)
        is_casual.append(w.lower() in CASUAL_WORDS)
    return {
        "token_length": np.array(length, float),
        "is_function_word": np.array(is_func, float),
        "code_char_density": np.array(code_dens, float),
        "is_html_context": np.array(is_html, float),
        "is_formal_register_word": np.array(is_formal, float),
        "is_casual_register_word": np.array(is_casual, float),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--target-family", type=int, default=0)
    ap.add_argument("--n-sample", type=int, default=20000,
                     help="true target-family test tokens to score+correlate")
    ap.add_argument("--per-family", type=int, default=None,
                     help="override the saved sample_args --per-family (default: reuse saved)")
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    per_family = a.per_family or sargs["per_family"]
    print(f"loaded {a.model}: sample_args={sargs}")

    fam_names, F, members = build_families(0.45)
    R, D, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D, valid, sargs["min_r"], sargs["max_l1"],
                                     exclude=set(sargs["exclude_family"]))
    rows, labels_sel = sample_equal_exclusive(mask, labels, per_family, sargs["seed"], fam_names)
    print(f"sampled {len(rows):,} tokens total (reproducing saved split); reading embeddings...")
    X = read_embeddings(rows, edim=768)

    idx = np.arange(len(labels_sel))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=sargs["seed"], stratify=labels_sel)
    X_te = scaler.transform(X[idx_te])
    y_te = labels_sel[idx_te]
    rows_te = rows[idx_te]

    col = classes.index(a.target_family)
    w = clf.coef_[col]

    true_mask = y_te == a.target_family
    cand = np.where(true_mask)[0]
    print(f"true fam{a.target_family} test tokens available: {len(cand):,}")
    if a.n_sample < len(cand):
        rng = np.random.default_rng(sargs["seed"])
        cand = rng.choice(cand, a.n_sample, replace=False)
    scores = X_te[cand] @ w
    rows_cand = rows_te[cand]

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    word_ids = [str(wids_all[r]) for r in rows_cand]
    print(f"fetching context for {len(word_ids):,} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, word_ids, N=10)

    feats = concept_features(word_ids, ctx)
    print(f"\n=== |correlation| ranking: logreg weight-direction score vs named concepts "
          f"(n={len(scores):,}) ===")
    results = []
    for name, vals in feats.items():
        r, p = pearsonr(scores, vals)
        results.append((name, r, p, vals.mean()))
    results.sort(key=lambda t: -abs(t[1]))
    for name, r, p, mean in results:
        print(f"  {name:>24s}: r={r:+.4f}  p={p:.2e}  mean={mean:.4f}")


if __name__ == "__main__":
    main()
