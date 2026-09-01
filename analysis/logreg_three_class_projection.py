"""Project test-set embeddings onto the FULL weight space of the trained multinomial logreg,
reduced to its non-redundant form. Softmax/argmax is invariant to adding a constant vector to
all class weights, so only the DIFFERENCES between logits matter -- with 3 classes that's
exactly 2 free dimensions, not 3. Using fam0 as the reference class:

    d1 = (w_fam1 - w_fam0) . x + (b_fam1 - b_fam0)
    d2 = (w_fam2 - w_fam0) . x + (b_fam2 - b_fam0)

d1=d2=0 is the fam0 decision score by construction, so a single (d1, d2) scatter is the
complete, non-redundant picture of the 3-way decision surface (no information lost, unlike
picking 2 of 3 raw pairwise projections). The decision regions are simple half-planes:
  fam0 wins iff d1<=0 and d2<=0
  fam1 wins iff d1>=0 and d1>=d2
  fam2 wins iff d2>=0 and d2>=d1
Shaded as background; TRUE-label points scattered on top, so you can see whether classes
occupy distinct corners or whether it's a smeared 1-D axis (per every prior finding here).

Loads the saved clf+scaler (no retraining), reproduces the same exclusive sample/split.

Usage:
  python logreg_three_class_projection.py --model fam0_logreg_weights.pkl --n-per-class 3000
"""
import argparse, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_family_subset import build_families
from predict_family_from_embedding_highconf import (
    compute_r_dist_all, exclusive_labels, sample_equal_exclusive, read_embeddings, FAM_LABELS,
)

OUT_DIR = Path(__file__).resolve().parent

# dataviz reference palette, categorical slots 1-3 (validated all-pairs CVD-safe for 3-series scatter)
COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}  # blue / orange / aqua
REGION_COLORS = {0: "#cde2fb", 1: "#f7d3bf", 2: "#bfe9d8"}  # light tints of the same 3 hues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fam0_logreg_weights.pkl")
    ap.add_argument("--n-per-class", type=int, default=3000)
    ap.add_argument("--ref-class", type=int, default=0, help="reference class subtracted out")
    ap.add_argument("--out", default="logreg_three_class_projection.png")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    with open(a.model, "rb") as f:
        saved = pickle.load(f)
    clf, scaler, classes, sargs = saved["clf"], saved["scaler"], saved["classes"], saved["sample_args"]
    print(f"loaded {a.model}: sample_args={sargs}, classes={classes}")
    print(f"coef_ shape: {clf.coef_.shape}")

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

    S = clf.decision_function(X_te)  # (N, 3) = X @ coef_.T + intercept_, columns ordered per `classes`
    ref_col = classes.index(a.ref_class)
    other_cols = [c for c in range(len(classes)) if c != ref_col]
    D = S[:, other_cols] - S[:, [ref_col]]  # (N, 2): d1, d2 relative to reference class
    other_classes = [classes[c] for c in other_cols]
    print(f"reference class: fam{a.ref_class}  ->  d1=fam{other_classes[0]}-fam{a.ref_class}, "
          f"d2=fam{other_classes[1]}-fam{a.ref_class}")

    rng = np.random.default_rng(a.seed)
    sel_idx = []
    for c in classes:
        cand = np.where(y_te == c)[0]
        n = min(a.n_per_class, len(cand))
        sel_idx.append(rng.choice(cand, n, replace=False))
    sel_idx = np.concatenate(sel_idx)
    D_sel, y_sel = D[sel_idx], y_te[sel_idx]
    print(f"plotting {len(sel_idx):,} points ({a.n_per_class} cap per class)")

    fig, ax = plt.subplots(figsize=(7.5, 7))

    # decision-region background: argmax(0, d1, d2) -> ref/other1/other2
    lim = np.abs(D_sel).max() * 1.1
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 400), np.linspace(-lim, lim, 400))
    scores_grid = np.stack([np.zeros_like(gx), gx, gy], axis=-1)  # [ref, other1, other2]
    winner_idx = np.argmax(scores_grid, axis=-1)  # 0=ref, 1=other1, 2=other2
    winner_class = np.array([a.ref_class] + other_classes)[winner_idx]
    for c in classes:
        ax.contourf(gx, gy, (winner_class == c).astype(float), levels=[0.5, 1.5],
                    colors=[REGION_COLORS[c]], alpha=0.6, zorder=0)

    for c in classes:
        m = y_sel == c
        ax.scatter(D_sel[m, 0], D_sel[m, 1], s=7, alpha=0.35, linewidths=0,
                   color=COLORS[c], label=f"true {FAM_LABELS.get(c, str(c))}")

    ax.axhline(0, color="#898781", lw=0.8, zorder=1)
    ax.axvline(0, color="#898781", lw=0.8, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"d1 = (w_fam{other_classes[0]} - w_fam{a.ref_class}) . x   "
                  f"[shaded: predicted region]")
    ax.set_ylabel(f"d2 = (w_fam{other_classes[1]} - w_fam{a.ref_class}) . x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(markerscale=3, fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(f"Non-redundant 2D projection onto the full weight space "
                 f"(reference: fam{a.ref_class})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / a.out, dpi=150)
    print(f"wrote {OUT_DIR / a.out}")

    # confusion sanity check within the plotted sample
    pred_sel = np.array([a.ref_class] + other_classes)[
        np.argmax(np.concatenate([np.zeros((len(D_sel), 1)), D_sel], axis=1), axis=1)]
    for c in classes:
        m = y_sel == c
        acc = (pred_sel[m] == c).mean()
        print(f"  true fam{c}: n={m.sum()}  predicted-correctly={acc:.3f}")


if __name__ == "__main__":
    main()
