"""Classify per-sample probability trajectories in cached SAE data.

Reads a cached preprocessed_data.dat (N, input_dim + num_models) produced by
create_cached_data.py, extracts the trailing `num_models` prob columns as each
sample's trajectory, and bins samples into shape categories. Writes per-category
counts and a figure of mean normalized curves per category.

Example:
    python analyze_trajectory_distribution.py \
        --cached_data_dir /home/nsrikant/.cache/n_moreearly_olmo3/olmo_256000_unseen/ofw=0.8 \
        --num_models 7 \
        --out_dir ./trend_analysis_olmo

    python analyze_trajectory_distribution.py \
        --cached_data_dir /home/nsrikant/.cache/n_moreearly_amber/amber_300_unseen/ofw=0.8 \
        --num_models 7 \
        --out_dir ./trend_analysis_amber
"""

import click
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DTYPES = {
    "np16": np.float16,
    "fp16": np.float16,
    "fp32": np.float32,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bf16": np.float32,  # bfloat16 not native to numpy; read as fp32 for analysis
    "bfloat16": np.float32,
}

CATEGORIES = [
    "flat",
    "monotonic_increase",
    "monotonic_decrease",
    "rise_then_plateau",
    "fall_then_plateau",
    "inverted_u",
    "u_shape",
    "noisy_other",
]


def classify(
    probs: np.ndarray,
    flat_range_thresh: float,
    mono_frac: float = 0.8,
    stagnation_ratio: float = 0.25,
) -> np.ndarray:
    """Assign a category index (into CATEGORIES) to each row of `probs`.

    Args:
        probs: (N, M) trajectory per sample.
        flat_range_thresh: if max-min of a trajectory is below this, call it flat.
            Should be set relative to the dataset's trajectory-range distribution
            (e.g. 10th percentile).
        mono_frac: fraction of diffs needed (same sign) to call it monotonic.
        stagnation_ratio: late-half range / full range below this => plateau tail.
    """
    N, M = probs.shape
    diffs = np.diff(probs, axis=1)
    traj_range = probs.max(axis=1) - probs.min(axis=1)
    total_change = probs[:, -1] - probs[:, 0]
    frac_pos = (diffs > 0).mean(axis=1)
    frac_neg = (diffs < 0).mean(axis=1)
    argmax_mid = (probs.argmax(axis=1) > 0) & (probs.argmax(axis=1) < M - 1)
    argmin_mid = (probs.argmin(axis=1) > 0) & (probs.argmin(axis=1) < M - 1)

    half = max(1, M // 2)
    late = probs[:, -half:]
    early = probs[:, :half]
    late_range = late.max(axis=1) - late.min(axis=1)
    early_range = early.max(axis=1) - early.min(axis=1)

    eps = 1e-12
    late_ratio = late_range / (traj_range + eps)
    early_ratio = early_range / (traj_range + eps)

    cat = np.full(N, CATEGORIES.index("noisy_other"), dtype=np.int32)

    is_flat = traj_range < flat_range_thresh
    late_plateau = (late_ratio < stagnation_ratio) & ~is_flat
    early_plateau = (early_ratio < stagnation_ratio) & ~is_flat

    mono_inc = (frac_pos >= mono_frac) & (total_change > 0) & ~is_flat
    mono_dec = (frac_neg >= mono_frac) & (total_change < 0) & ~is_flat

    rise_plateau = late_plateau & (total_change > 0) & ~mono_inc
    fall_plateau = late_plateau & (total_change < 0) & ~mono_dec

    # inverted-u: peak in interior, ends lower than peak, not already classified
    inv_u = argmax_mid & ~is_flat
    u = argmin_mid & ~is_flat

    # assign in priority order (later assignments overwrite prior "noisy_other")
    cat[inv_u] = CATEGORIES.index("inverted_u")
    cat[u] = CATEGORIES.index("u_shape")
    cat[mono_inc] = CATEGORIES.index("monotonic_increase")
    cat[mono_dec] = CATEGORIES.index("monotonic_decrease")
    cat[rise_plateau] = CATEGORIES.index("rise_then_plateau")
    cat[fall_plateau] = CATEGORIES.index("fall_then_plateau")
    cat[is_flat] = CATEGORIES.index("flat")
    return cat


def load_probs(cached_data_dir: str, num_models: int) -> np.ndarray:
    info_path = os.path.join(cached_data_dir, "cached_data_info.json")
    data_path = os.path.join(cached_data_dir, "preprocessed_data.dat")
    with open(info_path) as f:
        info = json.load(f)
    shape = tuple(info["shape"])
    dtype = DTYPES[info["dtype"]]
    assert shape[1] > num_models, f"shape {shape} has fewer cols than num_models={num_models}"
    data = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
    # Only materialize the trailing num_models columns to keep memory bounded.
    probs = np.asarray(data[:, -num_models:], dtype=np.float32)
    return probs


def summarize(cat: np.ndarray) -> pd.DataFrame:
    counts = np.bincount(cat, minlength=len(CATEGORIES))
    total = counts.sum()
    rows = [
        {"category": name, "count": int(counts[i]), "fraction": counts[i] / total}
        for i, name in enumerate(CATEGORIES)
    ]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def plot_domain_distribution(
    cat: np.ndarray,
    domains: np.ndarray,
    out_path: str,
    counts_csv_path: str,
    top_k_legend: int = 15,
):
    """Stacked horizontal bar of per-category domain composition (row-normalized)."""
    domains = np.asarray(domains)
    uniq_domains, dom_idx = np.unique(domains, return_inverse=True)
    n_cats = len(CATEGORIES)
    n_doms = len(uniq_domains)

    counts = np.zeros((n_cats, n_doms), dtype=np.int64)
    flat_idx = cat * n_doms + dom_idx
    np.add.at(counts.ravel(), flat_idx, 1)

    df = pd.DataFrame(counts, index=CATEGORIES, columns=uniq_domains)
    df.to_csv(counts_csv_path)

    row_totals = counts.sum(axis=1, keepdims=True).clip(min=1)
    frac = counts / row_totals

    # Order domains by overall frequency so the legend / colors are stable
    overall_order = np.argsort(-counts.sum(axis=0))
    uniq_domains = uniq_domains[overall_order]
    frac = frac[:, overall_order]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(uniq_domains))]

    fig, ax = plt.subplots(figsize=(12, 0.6 * n_cats + 2))
    y = np.arange(n_cats)
    left = np.zeros(n_cats)
    for j, dom in enumerate(uniq_domains):
        ax.barh(y, frac[:, j], left=left, color=colors[j],
                label=dom if j < top_k_legend else None,
                edgecolor="white", linewidth=0.3)
        left += frac[:, j]

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={counts[i].sum()})" for i, c in enumerate(CATEGORIES)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of samples in category")
    ax.set_title(f"Domain composition per trajectory category (top {top_k_legend} domains in legend)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_category_means(probs: np.ndarray, cat: np.ndarray, out_path: str):
    n_cats = len(CATEGORIES)
    ncols = 4
    nrows = (n_cats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
    axes = axes.flatten()

    for i, name in enumerate(CATEGORIES):
        ax = axes[i]
        mask = cat == i
        n = mask.sum()
        if n == 0:
            ax.set_title(f"{name} (n=0)")
            ax.axis("off")
            continue
        # Normalize each trajectory to [0,1] for shape comparison
        sub = probs[mask]
        lo = sub.min(axis=1, keepdims=True)
        hi = sub.max(axis=1, keepdims=True)
        norm = (sub - lo) / np.maximum(hi - lo, 1e-12)
        mean_curve = norm.mean(axis=0)
        # Show a few individual curves in light grey
        sample_idx = np.random.choice(n, size=min(50, n), replace=False)
        for j in sample_idx:
            ax.plot(norm[j], color="grey", alpha=0.15, linewidth=0.8)
        ax.plot(mean_curve, color="C0", linewidth=2.0)
        ax.set_title(f"{name} (n={n}, {n / len(cat):.1%})")
        ax.set_xlabel("model index")
        ax.set_ylim(-0.05, 1.05)

    for j in range(n_cats, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@click.command()
@click.option("--cached_data_dir", type=click.Path(exists=True), required=True,
              help="Directory containing preprocessed_data.dat and cached_data_info.json")
@click.option("--num_models", type=int, required=True,
              help="Number of model/prob columns at the end of each row")
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--flat_range_percentile", type=float, default=10.0,
              help="Trajectories with range below this percentile of dataset ranges are 'flat'")
@click.option("--mono_frac", type=float, default=0.8,
              help="Fraction of same-sign diffs required for 'monotonic'")
@click.option("--stagnation_ratio", type=float, default=0.25,
              help="late-half range / full range below this => plateau tail")
@click.option("--sample_cap", type=int, default=None,
              help="Optional random subsample size for speed")
@click.option("--seed", type=int, default=0)
def main(
    cached_data_dir: str,
    num_models: int,
    out_dir: str,
    flat_range_percentile: float,
    mono_frac: float,
    stagnation_ratio: float,
    sample_cap: int,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading probs from {cached_data_dir} ...", flush=True)
    probs = load_probs(cached_data_dir, num_models)
    print(f"Probs shape: {probs.shape}", flush=True)

    if sample_cap is not None and sample_cap < len(probs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(probs), size=sample_cap, replace=False)
        probs = probs[idx]
        print(f"Subsampled to {probs.shape}", flush=True)

    traj_range = probs.max(axis=1) - probs.min(axis=1)
    flat_thresh = float(np.percentile(traj_range, flat_range_percentile))
    print(f"Flat-range threshold ({flat_range_percentile}th pct): {flat_thresh:.6g}", flush=True)

    cat = classify(
        probs,
        flat_range_thresh=flat_thresh,
        mono_frac=mono_frac,
        stagnation_ratio=stagnation_ratio,
    )

    df = summarize(cat)
    print("\nCategory distribution:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "category_counts.csv"), index=False)

    range_stats = pd.Series(traj_range).describe(
        percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    )
    print("\nPer-sample trajectory range stats:")
    print(range_stats.to_string())
    range_stats.to_csv(os.path.join(out_dir, "range_stats.csv"))

    plot_path = os.path.join(out_dir, "category_mean_curves.png")
    plot_category_means(probs, cat, plot_path)
    print(f"\nWrote {plot_path}", flush=True)

    domains_pkl = os.path.join(os.path.dirname(cached_data_dir.rstrip("/")), "domains.pkl")
    if os.path.exists(domains_pkl):
        import pickle
        with open(domains_pkl, "rb") as f:
            domains = pickle.load(f)
        if sample_cap is not None and len(domains) != len(probs):
            domains = np.asarray(domains)[idx]
        domain_plot_path = os.path.join(out_dir, "category_domain_distribution.png")
        domain_csv_path = os.path.join(out_dir, "category_domain_counts.csv")
        plot_domain_distribution(cat, np.asarray(domains), domain_plot_path, domain_csv_path)
        print(f"Wrote {domain_plot_path} and {domain_csv_path}", flush=True)
    else:
        print(f"[skip] no domains.pkl at {domains_pkl}", flush=True)

    # Also save a histogram of trajectory ranges
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(traj_range, bins=80, log=True)
    ax.axvline(flat_thresh, color="red", linestyle="--", label=f"flat thresh ({flat_range_percentile}th pct)")
    ax.set_xlabel("trajectory range (max - min)")
    ax.set_ylabel("count (log)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "range_histogram.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
