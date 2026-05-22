"""Cross-task correlation heatmap + PCA, mirroring the Task Grouping tab in feature_explorer.py.

For a given dataset (e.g. "Amber-300-12000-k25-0.8"), builds a feature x task
correlation matrix across all precomputed_data_* folders, then:
  1. Plots a task-task cosine-similarity heatmap, reordered by hierarchical clustering
     so groups of similar tasks sit next to each other.
  2. Projects each task's correlation vector to 2D via PCA and colors by cluster.

Usage:
    python plot_task_correlation_heatmaps.py \
        --dataset Amber-300-3000-k25-0.8-late-checkpoints \
        --corr partial_pearson_corr \
        --min-corr 0.0 \
        --out-dir ./task_grouping_plots

    python plot_task_correlation_heatmaps.py \
        --dataset OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints \
        --corr partial_diff_spearman_corr \
        --min-corr 0.0 \
        --out-dir ./task_grouping_plots --max-prob-std 0.1

    python plot_task_correlation_heatmaps.py \
        --dataset OLMo3-7b-256k-3000-k25-0.8-early-checkpoints \
        --corr partial_spearman_corr \
        --min-corr 0.0 \
        --out-dir ./task_grouping_plots --max-prob-std 0.1


    python plot_task_correlation_heatmaps.py \
        --dataset OLMo3-7b-256k-3000-k25-0.8-early-checkpoints \
        --corr partial_pearson_corr \
        --min-corr 0.0 \
        --out-dir ./task_grouping_plots --max-prob-std 0.1

    python plot_task_correlation_heatmaps.py \
        --dataset OLMo3-7b-256k-3000-k25-0.8-early-checkpoints \
        --corr partial_diff_pearson_corr \
        --min-corr 0.0 \
        --out-dir ./task_grouping_plots --max-prob-std 0.1
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
valid_tasks = ["arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa", "medmcqa", "mmlu_stem","mmlu_social_sciences","mmlu_other", "blimp", "coqa","gsm8k", "lambada", "naturalqs"]

def build_feature_task_matrix(dataset_name, corr_key):
    """Feature x task correlation matrix across all precomputed_data_* folders.

    Returns (matrix, std_matrix, descriptions). std_matrix holds the mean of
    std_probs across checkpoints (i.e. average sample-prob std) per feature/task.
    """
    pattern = os.path.join(ANALYSIS_DIR, "precomputed_data_*")
    task_corrs = {}
    task_stds = {}
    descriptions = {}
    for d in sorted(glob.glob(pattern)):
        tname = os.path.basename(d).replace("precomputed_data_", "")
        if tname not in valid_tasks:
            continue
        fpath = os.path.join(d, f"{dataset_name}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            tdata = json.load(f)
        tc, ts = {}, {}
        for feat in tdata["features"]:
            c = feat.get(corr_key)
            if c is not None:
                tc[feat["feature_id"]] = c
            sp = feat.get("std_probs")
            if sp:
                arr = np.asarray(sp, dtype=float)
                arr = arr[~np.isnan(arr)]
                if arr.size:
                    ts[feat["feature_id"]] = float(arr.mean())
            if feat["feature_id"] not in descriptions and feat.get("description"):
                descriptions[feat["feature_id"]] = feat["description"]
        task_corrs[tname] = tc
        task_stds[tname] = ts

    if not task_corrs:
        return None, None, {}

    all_tasks = sorted(task_corrs.keys())
    all_features = sorted(set().union(*(tc.keys() for tc in task_corrs.values())))
    matrix = pd.DataFrame(index=all_features, columns=all_tasks, dtype=float)
    std_matrix = pd.DataFrame(index=all_features, columns=all_tasks, dtype=float)
    for tname in all_tasks:
        for fid, c in task_corrs[tname].items():
            matrix.loc[fid, tname] = c
        for fid, s in task_stds[tname].items():
            std_matrix.loc[fid, tname] = s
    return matrix, std_matrix, descriptions


def print_top_specificity_per_task(spec_matrix, descriptions, top_k=10, raw_matrix=None):
    """Print the top-K features per task, ranked by specificity score.

    spec_matrix: feature x task DataFrame of S_{i,j}.
    descriptions: optional {feature_id: description}.
    raw_matrix: optional raw Corr(Feature_i, Task_j) to display alongside S for context.
    """
    print(f"\n=== Top {top_k} features per task by specificity (S_ij) ===")
    for task in spec_matrix.columns:
        col = spec_matrix[task].dropna()
        if col.empty:
            continue
        top = col.sort_values(ascending=False).head(top_k)
        print(f"\n[{task}]")
        for fid, s in top.items():
            raw = raw_matrix.loc[fid, task] if raw_matrix is not None else None
            raw_str = f"  raw_r={raw:+.3f}" if raw is not None else ""
            desc = descriptions.get(fid, "")
            desc = (desc[:80] + "…") if len(desc) > 80 else desc
            print(f"  feat {fid:>6}  S={s:+.3f}{raw_str}  {desc}")


def task_specificity(matrix):
    """S_{i,j} = Corr(Feature_i, Task_j) - median_j Corr(Feature_i, Task_j).

    matrix: feature x task correlation DataFrame.
    Returns a DataFrame of the same shape with each row centered on its median.
    """
    median_per_feature = matrix.median(axis=1, skipna=True)
    return matrix.sub(median_per_feature, axis=0)


def compute_cosine_similarity(matrix, min_abs_corr, strict=True):
    """Task-task cosine similarity over features strong in both tasks (strict) or either (loose)."""
    tasks = list(matrix.columns)
    n = len(tasks)
    sim = pd.DataFrame(np.zeros((n, n)), index=tasks, columns=tasks)
    shared_counts = pd.DataFrame(np.zeros((n, n), dtype=int), index=tasks, columns=tasks)
    for i in range(n):
        for j in range(i, n):
            v1 = matrix[tasks[i]].values
            v2 = matrix[tasks[j]].values
            both = ~np.isnan(v1) & ~np.isnan(v2)
            if strict:
                strong = (np.abs(v1) >= min_abs_corr) & (np.abs(v2) >= min_abs_corr)
            else:
                strong = (np.abs(v1) >= min_abs_corr) | (np.abs(v2) >= min_abs_corr)
            mask = both & strong
            shared_counts.iloc[i, j] = int(mask.sum())
            shared_counts.iloc[j, i] = int(mask.sum())
            if mask.sum() == 0:
                continue
            a, b = v1[mask], v2[mask]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            s = float(np.dot(a, b) / norm) if norm > 0 else 0.0
            sim.iloc[i, j] = s
            sim.iloc[j, i] = s
    return sim, shared_counts


def cluster_tasks(cos_sim, max_k=10):
    """Hierarchical clustering (Ward). Returns (linkage Z, labels, ordered_task_names, k)."""
    sim_vals = cos_sim.values.astype(float)
    dist = np.clip(1.0 - sim_vals, 0, 2)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    best_k, best_score = 2, -1.0
    upper_k = min(max_k, len(cos_sim) - 1)
    for k in range(2, upper_k + 1):
        labels_k = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels_k)) < 2:
            continue
        sc = silhouette_score(dist, labels_k, metric="precomputed")
        if sc > best_score:
            best_score, best_k = sc, k

    labels = fcluster(Z, t=best_k, criterion="maxclust")
    leaf_order = leaves_list(Z)
    ordered_tasks = [cos_sim.index[i] for i in leaf_order]
    return Z, labels, ordered_tasks, best_k, best_score


def plot_clustered_heatmap(cos_sim, ordered_tasks, labels, out_path, title):
    """Heatmap with tasks reordered by clustering; cluster boundaries shown."""
    reordered = cos_sim.loc[ordered_tasks, ordered_tasks]
    task_to_label = dict(zip(cos_sim.index, labels))
    ordered_labels = [task_to_label[t] for t in ordered_tasks]

    n = len(ordered_tasks)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * n + 3), max(7, 0.4 * n + 2)))
    sns.heatmap(
        reordered,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        cbar_kws={"label": "Cosine similarity"},
        ax=ax,
    )
    # Cluster separator lines
    for i in range(1, n):
        if ordered_labels[i] != ordered_labels[i - 1]:
            ax.axhline(i, color="black", linewidth=1.5)
            ax.axvline(i, color="black", linewidth=1.5)

    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_prob_std_distribution(std_matrix, out_path, title, threshold=None):
    """Histograms of per-feature mean(std_probs): aggregated across tasks and per-task.

    For each feature/task we have mean_t std_probs (the average sample-prob std across
    checkpoints). We plot:
      (a) the per-feature aggregate (max over tasks of that quantity), and
      (b) the flat distribution of every (feature, task) entry.
    """
    per_feature_max = std_matrix.max(axis=1, skipna=True).dropna().values
    flat_entries = std_matrix.values[~np.isnan(std_matrix.values)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, sub in zip(
        axes,
        [per_feature_max, flat_entries],
        [f"per feature (max across tasks)  n={len(per_feature_max)}",
         f"per (feature, task) entry  n={len(flat_entries)}"],
    ):
        ax.hist(data, bins=50, color="#2196F3", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("mean(std_probs) across checkpoints")
        ax.set_ylabel("count")
        ax.set_title(sub)
        ax.grid(True, alpha=0.3)
        if threshold is not None:
            ax.axvline(threshold, color="red", linestyle="--",
                       label=f"threshold={threshold}")
            kept = (data <= threshold).sum()
            ax.text(0.98, 0.95, f"≤ thresh: {kept} ({100*kept/max(len(data),1):.1f}%)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
            ax.legend(loc="upper left")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pca(cos_sim, labels, out_path, title):
    """PCA of task correlation-similarity vectors, colored by cluster."""
    task_names = list(cos_sim.index)
    X = cos_sim.values.astype(float)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    palette = [
        "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FFC107",
        "#00BCD4", "#E91E63", "#8BC34A", "#FF9800", "#607D8B",
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    for ci in sorted(set(labels)):
        mask = np.array(labels) == ci
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=140,
            color=palette[(ci - 1) % len(palette)],
            label=f"Cluster {ci}",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        for j, t in enumerate(task_names):
            if mask[j]:
                ax.annotate(t, (coords[j, 0], coords[j, 1]),
                            xytext=(5, 5), textcoords="offset points",
                            fontsize=9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Amber-300-12000-k25-0.8",
                   help="Dataset file stem (without .json) under precomputed_data_*/.")
    p.add_argument("--corr", default="partial_pearson_corr",
                   choices=["partial_diff_spearman_corr", "partial_spearman_corr", "partial_pearson_corr", "partial_diff_pearson_corr",
                            "pearson_corr", "spearman_corr", "partial_diff_spearman_corr"])
    p.add_argument("--min-corr", type=float, default=0.3,
                   help="Per-task |corr| threshold for a feature to count.")
    p.add_argument("--loose", action="store_true",
                   help="Use 'strong in at least one task' filter (default is strict: both tasks).")
    p.add_argument("--specificity", action="store_true",
                   help="Replace raw correlations with task-specificity scores: "
                        "S_{i,j} = Corr(Feature_i, Task_j) - median_j Corr(Feature_i, All Tasks).")
    p.add_argument("--top-k", type=int, default=10,
                   help="Print top-K features per task ranked by specificity score.")
    p.add_argument("--max-k", type=int, default=10,
                   help="Max clusters to search over for silhouette.")
    p.add_argument("--max-prob-std", type=float, default=None,
                   help="Per-task filter: drop feature/task entries where the max of "
                        "std_probs across checkpoints exceeds this threshold (e.g. 0.2). "
                        "Keeps only features whose per-sample probabilities are tightly "
                        "concentrated for that task.")
    p.add_argument("--out-dir", default=os.path.join(ANALYSIS_DIR, "task_grouping_plots"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Dataset: {args.dataset}  |  corr: {args.corr}  |  min|corr|: {args.min_corr}")
    matrix, std_matrix, descriptions = build_feature_task_matrix(args.dataset, args.corr)
    if matrix is None:
        raise SystemExit(f"No precomputed_data_*/{args.dataset}.json files found.")
    print(f"  feature x task matrix: {matrix.shape[0]} features x {matrix.shape[1]} tasks")
    dist_path = os.path.join(args.out_dir,
                             f"prob_std_dist__{args.dataset}.png")
    plot_prob_std_distribution(
        std_matrix, dist_path,
        title=f"Distribution of max(std_probs) across checkpoints ({args.dataset})",
        threshold=args.max_prob_std,
    )
    if args.max_prob_std is not None:
        before = matrix.notna().sum().sum()
        mask = std_matrix > args.max_prob_std
        matrix = matrix.mask(mask)
        after = matrix.notna().sum().sum()
        print(f"  applied max_prob_std={args.max_prob_std}: kept {after}/{before} entries "
              f"({100.0 * after / max(before, 1):.1f}%)")
    raw_matrix = matrix.copy()
    spec_matrix = task_specificity(matrix)
    if args.top_k > 0:
        print_top_specificity_per_task(spec_matrix, descriptions,
                                       top_k=args.top_k, raw_matrix=raw_matrix)
    if args.specificity:
        matrix = spec_matrix
        print("  applied task-specificity transform (row-median-centered)")

    cos_sim, shared_counts = compute_cosine_similarity(
        matrix, args.min_corr, strict=not args.loose)
    filt = "strict (both tasks)" if not args.loose else "loose (either task)"
    print(f"  filter: {filt}")
    tri = shared_counts.values[np.triu_indices_from(shared_counts.values, k=1)]
    print(f"  shared-feature count per pair: min={tri.min()}, median={int(np.median(tri))}, max={tri.max()}")
    Z, labels, ordered_tasks, k, sil = cluster_tasks(cos_sim, max_k=args.max_k)
    print(f"  best k = {k}  (silhouette = {sil:.3f})")
    for ci in sorted(set(labels)):
        members = [t for t, L in zip(cos_sim.index, labels) if L == ci]
        print(f"    cluster {ci}: {members}")

    stem = f"{args.dataset}__{args.corr}__min{args.min_corr}"
    if args.specificity:
        stem += "__specificity"
    if args.max_prob_std is not None:
        stem += f"__maxstd{args.max_prob_std}"
    heatmap_path = os.path.join(args.out_dir, f"heatmap__{stem}.png")
    pca_path = os.path.join(args.out_dir, f"pca__{stem}.png")

    title_hm = f"Task-task cosine similarity ({args.dataset}, {args.corr}, min|r|={args.min_corr})"
    title_pca = f"PCA of task similarity vectors ({args.dataset}, {args.corr})"

    plot_clustered_heatmap(cos_sim, ordered_tasks, labels, heatmap_path, title_hm)
    plot_pca(cos_sim, labels, pca_path, title_pca)


if __name__ == "__main__":
    main()
