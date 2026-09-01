"""Rank-based shape matching: Kendall tau between each token's checkpoint ordering
and the task's, over all 36.7M tokens.

Why tau and not r: Pearson on 11 points with one dominant monotone trend saturates
near 0.95 for almost any pair, so the ranking is uninformative. Tau depends only on
the ordering, which is invariant to ANY monotone transform -- so z-norming (affine)
leaves it unchanged and raw-vs-znormed is the same number.

Reports:
  * float16 tie rate in the memmap (if orderings are decided by rounding, stop and
    use the raw logprobs instead).
  * tau distribution over the corpus, vs the exact permutation null.
  * exact-ordering matches (expected ~36.7M/11! = 0.9 by chance).
  * tie-collapsed strict check: checkpoints within 2 binomial SE of each other are
    merged into blocks, and only the surviving strict inequalities are required.
  * domain enrichment of the top-tau group.

Usage:
  python kendall_shape_match.py --task medmcqa --n-eval 4183
"""
import argparse, pickle, time
from pathlib import Path

import numpy as np

from task_shape_content_clusters import CACHE_DIR, OUT_DIR, STEPS, open_memmap
from sample_by_task_centroids import DEFAULT_STEM, build_task_centroids

DOMAINS = CACHE_DIR / "domains.pkl"
WORD_IDS = CACHE_DIR / "word_ids.pkl"
IU, JU = np.triu_indices(len(STEPS), k=1)          # the 55 checkpoint pairs


def raw_task_perf(task):
    import glob, json, os
    fp = os.path.join(Path(__file__).resolve().parent, f"precomputed_data_{task}",
                      f"{DEFAULT_STEM}.json")
    perf = json.load(open(fp))["overall_performance"]
    return np.array([p for p in perf if p is not None], dtype=np.float64)


def tau_scan(task, sign_t, chunk=2_000_000):
    """Kendall tau-b of every row's checkpoint ordering vs the task's, plus the
    float16 tie count per row. Cached."""
    cache = CACHE_DIR / f"kendall_{task}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"loaded cached tau from {cache}")
        return z["tau"], z["ties"], z["strict"]
    mm, edim, n_out = open_memmap()
    N = mm.shape[0]
    tau = np.empty(N, np.float32)
    ties = np.empty(N, np.int16)
    strict = np.zeros(N, bool)
    n_pairs = len(IU)
    # strict check uses only pairs the eval can actually resolve (sign_t != 0)
    keep = sign_t != 0
    t = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -n_out:], dtype=np.float32)
        d = b[:, JU] - b[:, IU]                       # (rows, 55)
        sd = np.sign(d)
        ties[s:e] = (sd == 0).sum(1)
        conc = (sd * sign_t)                          # +1 concordant, -1 discordant
        # tau-b denominator: sqrt of non-tied pairs on each side
        n_x = (sd != 0).sum(1)
        n_y = (sign_t != 0).sum()
        tau[s:e] = conc.sum(1) / (np.sqrt(n_x * n_y) + 1e-9)
        strict[s:e] = (conc[:, keep] > 0).all(1)
        print(f"  {e:,}/{N:,} ({time.time()-t:.0f}s)", flush=True)
    np.savez(cache, tau=tau, ties=ties, strict=strict)
    return tau, ties, strict


def perm_null(sign_t, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    n_y = (sign_t != 0).sum()
    out = np.empty(n)
    for i in range(n):
        p = rng.permutation(len(STEPS)).astype(np.float64)
        sd = np.sign(p[JU] - p[IU])
        out[i] = (sd * sign_t).sum() / np.sqrt(len(IU) * n_y)
    return out


def domain_enr(sel, tag):
    dom = pickle.load(open(DOMAINS, "rb"))
    names, cnt = np.unique(dom, return_counts=True)
    cs = cnt / cnt.sum()
    idx = {n: i for i, n in enumerate(names)}
    g = np.zeros(len(names))
    u, n = np.unique(dom[sel], return_counts=True)
    for x, k in zip(u, n):
        g[idx[x]] = k
    enr = (g / g.sum() + 1e-12) / (cs + 1e-12)
    o = np.argsort(-enr)
    print(f"\n--- domains, {tag} (n={len(sel):,}), enr sd={enr.std():.3f}")
    print("   top: " + ", ".join(f"{names[i][:32]} {enr[i]:.2f}" for i in o[:5]))
    print("   bot: " + ", ".join(f"{names[i][:32]} {enr[i]:.2f}" for i in o[-4:]))
    for h in ["common_crawl_health", "s2pdf_health",
              "common_crawl_science_math_and_technology", "finemath-3plus"]:
        print(f"      {h:<42} enr={enr[idx[h]]:.2f}  rank {list(o).index(idx[h])+1}/{len(names)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n-eval", type=int, default=4183, help="eval set size, for binomial SE")
    ap.add_argument("--top-n", type=int, default=200_000)
    a = ap.parse_args()

    perf = raw_task_perf(a.task)
    print(f"{a.task} raw: " + ", ".join(f"{s}k:{v:.3f}" for s, v in zip(STEPS, perf)))
    order = np.argsort(perf)
    print("ordering (ascending): " + " < ".join(f"{STEPS[i]}k" for i in order))

    d_t = perf[JU] - perf[IU]
    sign_t = np.sign(d_t)
    # collapse pairs the eval cannot resolve: |diff| < 2 * binomial SE
    se = np.sqrt(perf * (1 - perf) / a.n_eval)
    thr = 2 * np.sqrt(se[IU]**2 + se[JU]**2)
    unresolved = np.abs(d_t) < thr
    sign_t[unresolved] = 0
    print(f"\ncheckpoint pairs: {len(IU)} total, {unresolved.sum()} unresolved at 2 SE "
          f"(n_eval={a.n_eval:,}), {int((sign_t!=0).sum())} strict constraints remain")

    tau, ties, strict = tau_scan(a.task, sign_t)
    fin = np.isfinite(tau)
    print(f"\nfloat16 ties: mean {ties.mean():.2f} of 55 pairs per token; "
          f"tokens with 0 ties: {100*(ties==0).mean():.1f}%")

    null = perm_null(sign_t)
    print(f"\ntau null (random orderings): mean={null.mean():.3f} sd={null.std():.3f} "
          f"p99={np.percentile(null,99):.3f} max={null.max():.3f}")
    print(f"tau corpus: median={np.median(tau[fin]):.3f} p99={np.percentile(tau[fin],99):.3f} "
          f"max={tau[fin].max():.3f}")
    for t in (0.6, 0.8, 0.9):
        obs = (tau[fin] > t).sum()
        exp = (null > t).mean() * fin.sum()
        print(f"  tau>{t}: {obs:,} observed vs {exp:,.0f} expected by chance "
              f"({obs/max(exp,1e-9):.1f}x)")
    print(f"\nstrict ordering match (all {int((sign_t!=0).sum())} resolvable pairs correct): "
          f"{strict.sum():,} tokens ({100*strict.mean():.4f}%)")
    print(f"  chance for a random ordering: {(perm_null(sign_t)==perm_null(sign_t).max()).mean():.2e} "
          f"-- see tau null above for calibration")

    top = np.argpartition(-tau, a.top_n)[:a.top_n]
    domain_enr(top, f"top-{a.top_n//1000}k by tau")
    if strict.sum() > 200:
        domain_enr(np.where(strict)[0], "strict ordering match")


if __name__ == "__main__":
    main()
