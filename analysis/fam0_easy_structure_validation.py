"""Rigorous, automatic re-test of the two candidate patterns a prior small-N/eyeballed PCA
pass found within fam0's EASIEST-to-classify tokens (fam0_only_pca_groups.py, ~5000 tokens,
manual inspection of sorted extremes): a formality/register axis and verb-lemma-family
clustering. This script re-derives PCA on a larger sample of true-fam0 test tokens (ranked by
p(fam0), i.e. the literal "easiest to classify" set from the very first analysis in this
session) and checks both hypotheses with actual statistics instead of eyeballing:

  1. Formality/register: correlate each PC's score against FOUR automatic, no-hand-curation
     readability/register proxies computed over each token's +/-10 word context window --
     mean word length, long-word ratio (len>=7, a standard Flesch-style complexity proxy),
     type-token ratio (lexical diversity), and stopword ratio (sklearn ENGLISH_STOP_WORDS).
  2. Verb-lemma-family clustering: Porter-stem (algorithmic, no downloaded corpus needed) each
     target word; for stem-families with >=5 members among the sampled tokens, test whether
     same-family tokens sit closer together in PC space than chance via eta^2 (between-family
     variance / total variance) with a label-permutation null (500 shuffles) for a real p-value.

Loads the saved clf+scaler (no retraining), reproduces the same exclusive sample/split.

Usage:
  python fam0_easy_structure_validation.py --model fam0_logreg_weights.pkl --n-easy 10000
"""
import argparse, pickle, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from nltk.stem import PorterStemmer
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
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
WORD_RE = re.compile(r"[A-Za-z']+")
STEMMER = PorterStemmer()


def register_proxies(word_ids, ctx):
    mean_len, long_ratio, ttr, stop_ratio = [], [], [], []
    for wid in word_ids:
        e = ctx.get(wid, {"before": "", "word": "", "after": ""})
        words = WORD_RE.findall(e["before"] + " " + e["word"] + " " + e["after"])
        words_l = [w.lower() for w in words]
        if not words_l:
            mean_len.append(0); long_ratio.append(0); ttr.append(0); stop_ratio.append(0)
            continue
        lens = [len(w) for w in words_l]
        mean_len.append(np.mean(lens))
        long_ratio.append(np.mean([l >= 7 for l in lens]))
        ttr.append(len(set(words_l)) / len(words_l))
        stop_ratio.append(np.mean([w in ENGLISH_STOP_WORDS for w in words_l]))
    return {
        "mean_word_length": np.array(mean_len),
        "long_word_ratio(len>=7)": np.array(long_ratio),
        "type_token_ratio": np.array(ttr),
        "stopword_ratio": np.array(stop_ratio),
    }


def lemma_eta_squared(pc_scores, stems, min_family=5, n_perm=500, seed=42):
    """eta^2 = between-family SS / total SS, restricted to families with >=min_family members.
    Returns (eta2_real, eta2_null_mean, eta2_null_std, z, families_used, n_tokens_used)."""
    fam_counts = defaultdict(list)
    for i, s in enumerate(stems):
        fam_counts[s].append(i)
    keep_families = {s: idxs for s, idxs in fam_counts.items() if len(idxs) >= min_family}
    idx_all = np.array([i for idxs in keep_families.values() for i in idxs])
    labels = np.array([s for s, idxs in keep_families.items() for _ in idxs])
    if len(idx_all) < 20 or len(keep_families) < 2:
        return None
    y = pc_scores[idx_all]

    def eta2(y, labels):
        grand_mean = y.mean()
        ss_tot = np.sum((y - grand_mean) ** 2)
        ss_between = 0.0
        for lab in set(labels.tolist()):
            m = labels == lab
            ss_between += m.sum() * (y[m].mean() - grand_mean) ** 2
        return ss_between / ss_tot if ss_tot > 0 else 0.0

    real = eta2(y, labels)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        null.append(eta2(y, shuffled))
    null = np.array(null)
    z = (real - null.mean()) / (null.std() + 1e-12)
    return real, null.mean(), null.std(), z, len(keep_families), len(idx_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--target-family", type=int, default=0)
    ap.add_argument("--n-easy", type=int, default=10000)
    ap.add_argument("--n-pcs", type=int, default=10)
    ap.add_argument("--min-lemma-family", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
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

    col = classes.index(a.target_family)
    proba = clf.predict_proba(X_te)[:, col]
    true_mask = y_te == a.target_family
    cand = np.where(true_mask)[0]
    top = cand[np.argsort(-proba[cand])[: a.n_easy]]
    print(f"true fam{a.target_family} tokens available: {len(cand):,}; using top {len(top):,} "
          f"by p(fam{a.target_family}) (range {proba[top].min():.3f}-{proba[top].max():.3f})")

    X_easy = X_te[top]
    rows_easy = rows_te[top]

    print(f"running PCA (n_components={a.n_pcs}) on ONLY these easy fam{a.target_family} tokens...")
    pca = PCA(n_components=a.n_pcs, random_state=a.seed)
    proj = pca.fit_transform(X_easy)
    print("explained variance ratio: " +
          ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(pca.explained_variance_ratio_)) +
          f"  (cumulative={pca.explained_variance_ratio_.sum():.3f})")

    wids_all = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    word_ids = [str(wids_all[r]) for r in rows_easy]
    print(f"fetching context for {len(word_ids):,} tokens...")
    ctx = get_words_in_context(INPUT_FEATURES, word_ids, N=10)

    # --- 1. formality/register proxies vs each PC ---
    proxies = register_proxies(word_ids, ctx)
    print(f"\n=== register-proxy correlations (n={len(word_ids):,}) ===")
    for pc in range(a.n_pcs):
        scores = proj[:, pc]
        rows_out = []
        for name, vals in proxies.items():
            r, p = pearsonr(scores, vals)
            rows_out.append((name, r, p))
        rows_out.sort(key=lambda t: -abs(t[1]))
        best = rows_out[0]
        flag = " <-- notable" if abs(best[1]) > 0.1 else ""
        print(f"  PC{pc+1} (var {pca.explained_variance_ratio_[pc]:.3f}): " +
              ", ".join(f"{n}=r{r:+.3f}(p={p:.1e})" for n, r, p in rows_out) + flag)

    # --- 2. verb-lemma-family clustering vs each PC ---
    target_words = [ctx.get(wid, {"word": ""})["word"].strip().lower() for wid in word_ids]
    stems = [STEMMER.stem(w) if w.isalpha() else f"__nonword_{w}" for w in target_words]
    print(f"\n=== lemma-family (Porter stem) clustering test, min_family={a.min_lemma_family}, "
          f"{a.n_perm} permutations ===")
    for pc in range(a.n_pcs):
        res = lemma_eta_squared(proj[:, pc], stems, a.min_lemma_family, a.n_perm, a.seed)
        if res is None:
            print(f"  PC{pc+1}: not enough qualifying lemma families")
            continue
        real, null_mean, null_std, z, n_fam, n_tok = res
        flag = " <-- SIGNIFICANT" if z > 3 else ""
        print(f"  PC{pc+1}: eta^2={real:.4f}  null={null_mean:.4f}+/-{null_std:.4f}  z={z:+.2f}  "
              f"({n_fam} lemma families, {n_tok} tokens){flag}")


if __name__ == "__main__":
    main()
