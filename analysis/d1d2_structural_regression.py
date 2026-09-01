"""How much of the classifier's full weight-space signal (d1, d2 -- the non-redundant 2D
projection from logreg_three_class_projection.py) is explained by structural descriptors that
DON'T assume English/POS-taggable text? Every earlier proxy that used a POS tagger or function-
word lexicon silently assumed the window was taggable prose, but this corpus is heavily code/
markup/math/foreign-script -- an English POS tagger on `<td>3.40</td>` is just noise. So this
uses only UNIVERSAL, language-agnostic structural features computed straight from characters:

  token_length, mean_word_length, code_char_density, is_html_context, digit_density,
  punct_density, non_ascii_ratio, stopword_ratio (a fixed closed-class word-list membership
  check -- not a tagger, degrades gracefully to ~0 on code rather than emitting garbage tags)

Regresses these jointly against d1 and d2 (across ALL 3 families, not just fam0, since d1/d2
are the complete non-redundant decision surface) and reports R^2 + standardized coefficients,
plus simple per-feature correlations for interpretability.

Loads the saved clf+scaler (no retraining), reproduces the same exclusive sample/split.

Usage:
  python d1d2_structural_regression.py --model fam0_logreg_weights.pkl --n-per-family 5000
"""
import argparse, pickle, re, string, sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LinearRegression
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
WORD_RE = re.compile(r"[A-Za-z']+")
CODE_CHARS = re.compile(r"[<>;{}\\$=\[\]|`]")
HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[ >]")
PUNCT_SET = set(string.punctuation)


def structural_features(word_ids, ctx):
    feats = {k: [] for k in ["token_length", "mean_word_length", "code_char_density",
                              "is_html_context", "digit_density", "punct_density",
                              "non_ascii_ratio", "stopword_ratio"]}
    for wid in word_ids:
        e = ctx.get(wid, {"before": "", "word": "", "after": ""})
        window = e["before"] + e["word"] + e["after"]
        words = WORD_RE.findall(e["before"] + " " + e["word"] + " " + e["after"])
        words_l = [w.lower() for w in words]
        n = max(len(window), 1)
        feats["token_length"].append(len(e["word"].strip()))
        feats["mean_word_length"].append(np.mean([len(w) for w in words_l]) if words_l else 0)
        feats["code_char_density"].append(len(CODE_CHARS.findall(window)) / n)
        feats["is_html_context"].append(float(bool(HTML_TAG.search(window))))
        feats["digit_density"].append(sum(c.isdigit() for c in window) / n)
        feats["punct_density"].append(sum(c in PUNCT_SET for c in window) / n)
        feats["non_ascii_ratio"].append(sum(ord(c) > 127 for c in window) / n)
        feats["stopword_ratio"].append(
            np.mean([w in ENGLISH_STOP_WORDS for w in words_l]) if words_l else 0)
    return {k: np.array(v) for k, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--ref-class", type=int, default=0)
    ap.add_argument("--n-per-family", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}")

    fam_names, F, members = build_families(0.45)
    R, D_, valid = compute_r_dist_all(F)
    mask, labels = exclusive_labels(R, D_, valid, sargs["min_r"], sargs["max_l1"],
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

    S = clf.decision_function(X_te)
    ref_col = classes.index(a.ref_class)
    other_cols = [c for c in range(len(classes)) if c != ref_col]
    other_classes = [classes[c] for c in other_cols]
    Dproj = S[:, other_cols] - S[:, [ref_col]]
    print(f"reference class: fam{a.ref_class}  ->  d1=fam{other_classes[0]}-fam{a.ref_class}, "
          f"d2=fam{other_classes[1]}-fam{a.ref_class}")

    rng = np.random.default_rng(a.seed)
    sel_idx = []
    for c in classes:
        cand = np.where(y_te == c)[0]
        n = min(a.n_per_family, len(cand))
        sel_idx.append(rng.choice(cand, n, replace=False))
    sel_idx = np.concatenate(sel_idx)
    D_sel = Dproj[sel_idx]
    print(f"using {len(sel_idx):,} tokens ({a.n_per_family} cap per family)")

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    word_ids = [str(wids_all[rows_te[i]]) for i in sel_idx]
    print(f"fetching context for {len(word_ids):,} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, word_ids, N=10)

    feats = structural_features(word_ids, ctx)
    feat_names = list(feats.keys())
    Xf = np.stack([feats[k] for k in feat_names], axis=1)
    scaler_f = StandardScaler()
    Xf_std = scaler_f.fit_transform(Xf)

    for label, target in [("d1", D_sel[:, 0]), ("d2", D_sel[:, 1])]:
        print(f"\n=== regressing structural features against {label} (n={len(target):,}) ===")
        reg = LinearRegression().fit(Xf_std, target)
        r2 = reg.score(Xf_std, target)
        print(f"  joint R^2 = {r2:.4f}")
        coefs = sorted(zip(feat_names, reg.coef_), key=lambda t: -abs(t[1]))
        for name, coef in coefs:
            r, p = pearsonr(feats[name], target)
            print(f"    {name:>20s}: std_coef={coef:+.3f}  pairwise_r={r:+.3f} (p={p:.1e})")


if __name__ == "__main__":
    main()
