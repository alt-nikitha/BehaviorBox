"""Does the corpus contain ANY token with the task curve's dip, or only monotone rises?

The r-ranking is dominated by the gross "low then high" shape, so a token can score
r=0.99 while completely missing the mid-training dip. This decomposes the task curve:

    task_z  =  a * PC1(corpus curves)  +  residual

PC1 is the corpus's dominant shape (the monotone rise). The residual is what makes the
task curve task-shaped -- for medmcqa, the fall to chance at 4k-16k. We then score all
36.7M tokens against the RESIDUAL, not the raw curve, and look at what comes back.

Also reports:
  * PCA spectrum of token curves -- how many distinct shapes the corpus actually has.
  * count of tokens that are genuinely U-shaped (fall then recover) by a direct test.
  * a plot of the best residual matches, individually (no cluster averaging).

Usage:
  python find_dip_tokens.py --task medmcqa --plot-top 12
"""
import argparse, pickle, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task_shape_content_clusters import CACHE_DIR, OUT_DIR, STEPS, open_memmap, task_curve

WORD_IDS = CACHE_DIR / "word_ids.pkl"


def sample_curves(n, seed=0):
    """Z-normed curves for a random sample of valid rows."""
    mm, edim, n_out = open_memmap()
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(mm.shape[0], n, replace=False))
    T = np.empty((n, n_out), np.float32)
    for s in range(0, n, 20000):
        e = min(s + 20000, n)
        p = np.asarray(mm[rows[s:e], -n_out:], dtype=np.float32)
        T[s:e] = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-9)
    return T[np.isfinite(T).all(1)]


def scan(target, tag, chunk=2_000_000):
    """Correlate every row's z-curve with `target`; also count U-shaped tokens."""
    cache = CACHE_DIR / f"residcorr_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["r"], z["u"]
    t_z = (target - target.mean()) / target.std()
    mm, edim, n_out = open_memmap()
    N = mm.shape[0]
    r = np.empty(N, np.float32)
    u = np.zeros(N, bool)
    t = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -n_out:], dtype=np.float32)
        mu = b.mean(1, keepdims=True); sd = b.std(1, keepdims=True)
        z = (b - mu) / (sd + 1e-9)
        r[s:e] = (z @ t_z) / n_out
        # U-shape: early level clearly above the 4k-16k trough, and late above early
        early = z[:, 0]
        trough = z[:, 2:5].min(1)
        late = z[:, -1]
        u[s:e] = (early - trough > 0.5) & (late - early > 0.5)
        r[s:e][sd[:, 0] <= 1e-6] = -np.inf
        print(f"  {e:,}/{N:,} ({time.time()-t:.0f}s)", flush=True)
    np.savez(cache, r=r, u=u)
    return r, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--plot-top", type=int, default=12)
    a = ap.parse_args()

    c = task_curve(a.task)
    T = sample_curves(a.sample)
    print(f"PCA on {len(T):,} sampled token curves")
    X = T - T.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = S**2 / (S**2).sum()
    print("explained variance: " + ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(ev[:5])))
    print(f"PC1 alone accounts for {ev[0]*100:.1f}% of all trajectory variation")

    pc1 = Vt[0] / np.linalg.norm(Vt[0])
    if np.corrcoef(pc1, np.arange(len(pc1)))[0, 1] < 0:
        pc1 = -pc1
    proj = float(c @ pc1)
    resid = c - proj * pc1
    print(f"\ntask curve: {proj/np.linalg.norm(c)*100:.0f}% of its norm is PC1; "
          f"residual norm={np.linalg.norm(resid):.2f}")
    print("residual (the task-specific part): " +
          ", ".join(f"{s}k:{v:+.2f}" for s, v in zip(STEPS, resid)))

    r_raw, _ = scan(c, a.task + "_raw") if not (CACHE_DIR / f"taskcorr_{a.task}.npz").exists() \
        else (np.load(CACHE_DIR / f"taskcorr_{a.task}.npz")["r"], None)
    r_res, u = scan(resid, a.task + "_resid")

    fin = np.isfinite(r_res)
    print(f"\n=== matching the RESIDUAL over all {fin.sum():,} valid tokens ===")
    print(f"r median={np.median(r_res[fin]):.3f}  p99={np.percentile(r_res[fin],99):.3f}  "
          f"max={r_res[fin].max():.3f}")
    for thr in (0.5, 0.7, 0.9):
        print(f"  tokens with residual r>{thr}: {(r_res[fin]>thr).sum():,} "
              f"({100*(r_res[fin]>thr).mean():.4f}%)")
    print(f"U-shaped tokens (fall >0.5sd then recover >0.5sd): {u.sum():,} "
          f"({100*u.sum()/fin.sum():.3f}%)")

    # how do the best raw-r tokens do on the residual?
    top_raw = np.argpartition(-r_raw, 200000)[:200000]
    print(f"\ntop-200k by RAW r: residual r median={np.median(r_res[top_raw]):.3f}, "
          f"U-shaped fraction={100*u[top_raw].mean():.3f}% "
          f"(corpus baseline {100*u[fin].mean():.3f}%)")

    # plot best residual matches individually
    best = np.argpartition(-r_res, a.plot_top)[:a.plot_top]
    best = best[np.argsort(-r_res[best])]
    mm, edim, n_out = open_memmap()
    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    cols = 4; rows_n = int(np.ceil(a.plot_top / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 2.7 * rows_n), sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax in axes[a.plot_top:]:
        ax.axis("off")
    for i, row in enumerate(best):
        p = np.asarray(mm[row, -n_out:], dtype=np.float32)
        z = (p - p.mean()) / (p.std() + 1e-9)
        ax = axes[i]
        ax.plot(STEPS, z, color="#4a7fb5", lw=1.8, label="token")
        ax.plot(STEPS, c, color="black", ls="--", lw=1.4, label="task")
        ax.set_xscale("log"); ax.set_xticks([1, 8, 68, 256])
        ax.set_xticklabels(["1k", "8k", "68k", "256k"], fontsize=7)
        ax.set_title(f"{wids[row]}  residual r={r_res[row]:.2f}", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"Tokens best matching the {a.task} RESIDUAL (the dip), individually",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96])
    out = OUT_DIR / f"{a.task}_residual_matches.png"
    fig.savefig(out, dpi=115)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
