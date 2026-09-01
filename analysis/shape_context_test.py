"""Is the *context* around shape-matched tokens different, even if the tokens aren't?

Two tests, shape group (top-N by task-curve correlation) vs a random control:

  1. DOMAIN  — per-row source-domain labels (domains.pkl, aligned to memmap rows) for
     the FULL group vs the whole corpus. Enrichment = group share / corpus share.
     No text reads needed, so this runs over all 200k tokens.
  2. CONTEXT WORDS — log-odds-ratio with an informative Dirichlet prior (Monroe et al.
     2008) over the +/-10-word windows, shape group vs control. z>1.96 is significant.
     Uses a text sample (parquet reads), default 6k tokens per side.

Usage:
  python shape_context_test.py --task medmcqa --top-n 200000 --sample 6000
"""
import argparse, collections, json, math, pickle, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sae.data_utils import get_words_in_context

CACHE_DIR = Path("/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen")
DOMAINS = CACHE_DIR / "domains.pkl"
WORD_IDS = CACHE_DIR / "word_ids.pkl"
INPUT_FEATURES = "/data/user_data/nsrikant/bbox_data/output/olmo_256000_unseen/input_features"
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def groups(task, top_n, seed):
    z = np.load(CACHE_DIR / f"taskcorr_{task}.npz")
    r, valid = z["r"], z["valid"]
    rows = np.argpartition(-r, top_n)[:top_n]
    ctrl = np.random.default_rng(seed).choice(np.where(valid)[0], top_n, replace=False)
    print(f"shape group r {r[rows].min():.3f}..{r[rows].max():.3f}   control r "
          f"{r[ctrl].min():.3f}..{r[ctrl].max():.3f}")
    return rows, ctrl


def domain_test(rows, ctrl, show):
    dom = pickle.load(open(DOMAINS, "rb"))
    names, corpus = np.unique(dom, return_counts=True)
    idx = {n: i for i, n in enumerate(names)}
    corpus_share = corpus / corpus.sum()

    def share(sel):
        c = np.zeros(len(names))
        u, n = np.unique(dom[sel], return_counts=True)
        for x, k in zip(u, n):
            c[idx[x]] = k
        return c / c.sum(), c

    g_share, g_cnt = share(rows)
    c_share, _ = share(ctrl)
    g_enr = (g_share + 1e-12) / (corpus_share + 1e-12)
    c_enr = (c_share + 1e-12) / (corpus_share + 1e-12)

    print(f"\n=== DOMAIN (all {len(rows):,} tokens per group, {len(names)} domains) ===")
    print(f"enrichment sd: shape={g_enr.std():.3f}  control={c_enr.std():.3f}")
    print(f"{'domain':<40}{'corpus%':>9}{'group%':>9}{'enr':>7}{'ctrlEnr':>9}")
    for i in np.argsort(-g_enr)[:show]:
        print(f"{names[i][:39]:<40}{corpus_share[i]*100:>8.2f}%{g_share[i]*100:>8.2f}%"
              f"{g_enr[i]:>7.2f}{c_enr[i]:>9.2f}")
    print("  ... least enriched ...")
    for i in np.argsort(g_enr)[:show][::-1]:
        print(f"{names[i][:39]:<40}{corpus_share[i]*100:>8.2f}%{g_share[i]*100:>8.2f}%"
              f"{g_enr[i]:>7.2f}{c_enr[i]:>9.2f}")


def ctx_counts(rows, wids, sample, rng):
    sel = rng.choice(rows, min(sample, len(rows)), replace=False)
    wic = get_words_in_context(INPUT_FEATURES, [str(w) for w in wids[sel]], N=10)
    c = collections.Counter()
    for e in wic.values():
        c.update(w.lower() for w in TOKEN_RE.findall(e["before"] + " " + e["after"]))
    return c, len(wic)


def logodds(a, b, show):
    """Monroe et al. informative-Dirichlet log-odds z-scores: a vs b."""
    prior = a + b
    n_a, n_b, n_p = sum(a.values()), sum(b.values()), sum(prior.values())
    out = []
    for w, p in prior.items():
        if p < 20:
            continue
        ya, yb = a[w], b[w]
        d = (math.log((ya + p) / (n_a + n_p - ya - p))
             - math.log((yb + p) / (n_b + n_p - yb - p)))
        var = 1.0 / (ya + p) + 1.0 / (yb + p)
        out.append((d / math.sqrt(var), w, ya, yb))
    out.sort(reverse=True)
    print(f"\n=== CONTEXT WORDS (log-odds z, shape vs control; |z|>1.96 significant) ===")
    print("most shape-group:  " + ", ".join(f"{w}({z:+.1f})" for z, w, _, _ in out[:show]))
    print("most control:      " + ", ".join(f"{w}({z:+.1f})" for z, w, _, _ in out[-show:][::-1]))
    zs = np.array([o[0] for o in out])
    print(f"vocab compared={len(out):,}  |z|>1.96: {(np.abs(zs)>1.96).sum()} "
          f"({100*(np.abs(zs)>1.96).mean():.1f}%)  max|z|={np.abs(zs).max():.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--top-n", type=int, default=200_000)
    ap.add_argument("--sample", type=int, default=6000, help="tokens per side for context text")
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows, ctrl = groups(a.task, a.top_n, a.seed)
    domain_test(rows, ctrl, a.show)

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    rng = np.random.default_rng(a.seed)
    print(f"\nreading contexts ({a.sample:,} per side) ...")
    ca, na = ctx_counts(rows, wids, a.sample, rng)
    cb, nb = ctx_counts(ctrl, wids, a.sample, rng)
    print(f"resolved {na:,} shape / {nb:,} control contexts; "
          f"{sum(ca.values()):,} vs {sum(cb.values()):,} context words")
    logodds(ca, cb, a.show)


if __name__ == "__main__":
    main()
