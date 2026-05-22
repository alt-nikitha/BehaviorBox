"""
Per-task boxplots of per-word NLL bucketed by max-|corr| feature correlation.

Buckets (by best-correlation value, ties broken by max |corr|):
    [-1.00, -0.75),  [-0.75, -0.50),  [-0.50, 0.50],  (0.50, 0.75],  (0.75, 1.00]

Usage:
    python plot_loss_boxplots.py \
        --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/task_mapped_results \
        --losses /data/user_data/nsrikant/bbox_data/causal_data/amber/token_losses.npy \
        --output_dir /home/nsrikant/BehaviorBoxNew/causal_train/boxplots_amber

    python plot_loss_boxplots.py \
        --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/task_mapped_results \
        --losses /data/user_data/nsrikant/bbox_data/causal_data/olmo/token_losses.npy \
        --output_dir /home/nsrikant/BehaviorBoxNew/causal_train/boxplots_olmo
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

BUCKETS = [
    ("[-1,-0.75)", -1.0, -0.75),
    ("[-0.75,-0.5)", -0.75, -0.5),
    ("[-0.5,0.5]", -0.5, 0.5),
    ("(0.5,0.75]", 0.5, 0.75),
    ("(0.75,1]", 0.75, 1.0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_mapped_dir", required=True)
    p.add_argument("--losses", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def process_task(npz_path, losses):
    d = np.load(npz_path)
    w_idx = d["w_idx"]
    best_c = d["best_c"]
    loss = np.asarray(losses[w_idx], dtype=np.float32)
    finite = np.isfinite(loss)
    loss = loss[finite]
    best_c = best_c[finite]
    masks = [
        (best_c >= -1.0) & (best_c < -0.75),
        (best_c >= -0.75) & (best_c < -0.5),
        (best_c >= -0.5) & (best_c <= 0.5),
        (best_c > 0.5) & (best_c <= 0.75),
        (best_c > 0.75) & (best_c <= 1.0),
    ]
    return [loss[m] for m in masks]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    losses = np.load(args.losses)
    print(f"Loaded {len(losses)} losses "
          f"({np.isfinite(losses).sum()} finite)")

    files = sorted(glob.glob(os.path.join(args.task_mapped_dir, "*_bestc.npz")))
    print(f"Found {len(files)} tasks")

    task_data = []
    for path in tqdm(files, desc="Tasks"):
        task = os.path.basename(path).replace("_bestc.npz", "")
        buckets = process_task(path, losses)
        counts = [len(b) for b in buckets]
        if sum(counts) == 0:
            print(f"  [skip] {task}: no tokens")
            continue
        task_data.append((task, buckets, counts))

    # Combined figure: one subplot per task
    n = len(task_data)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    summary_rows = []
    for ax, (task, buckets, counts) in zip(axes.flat, task_data):
        labels = [f"{lab}\nn={counts[i]}" for i, (lab, _, _) in enumerate(BUCKETS)]
        ax.boxplot(buckets, labels=labels, showfliers=False)
        ax.set_ylabel("Per-word NLL")
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)
        medians = [float(np.median(b)) if len(b) else float("nan") for b in buckets]
        summary_rows.append((task, counts, medians))

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Per-word NLL by max-|corr| bucket", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(args.output_dir, "all_tasks_boxplots.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print("\nSummary (median NLL per bucket):")
    header = f"{'task':<22}" + "".join(f" {lab:>14}" for lab, _, _ in BUCKETS)
    print(header)
    print("-" * len(header))
    for task, counts, medians in summary_rows:
        row = f"{task:<22}"
        for c, m in zip(counts, medians):
            row += f" {m:>7.3f}(n={c})"[:15].rjust(15)
        print(row)

    print(f"\nWrote combined figure ({len(summary_rows)} tasks) to {out}")


if __name__ == "__main__":
    main()
