"""
Filter tasks where the |corr| > 0.75 positive and negative tails are
roughly balanced in count and baseline loss, so the two upweighting
experiments (+ve vs -ve) are fairly comparable.

Per-token correlation = correlation of the feature with max |c| active
on that token (same convention as plot_loss_boxplots.py).

Usage:
    python filter_balanced_tasks.py \
        --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/task_mapped_results \
        --losses /data/user_data/nsrikant/bbox_data/causal_data/amber/token_losses.npy \
        --output /home/nsrikant/BehaviorBoxNew/causal_train/balanced_tasks_amber.csv

    python filter_balanced_tasks.py \
        --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/task_mapped_results \
        --losses /data/user_data/nsrikant/bbox_data/causal_data/olmo/token_losses.npy \
        --output /home/nsrikant/BehaviorBoxNew/causal_train/balanced_tasks_olmo.csv
"""

import argparse
import csv
import glob
import os

import numpy as np
import orjson
from tqdm import tqdm

POS_THR = 0.75
NEG_THR = -0.75


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_mapped_dir", required=True)
    p.add_argument("--losses", required=True)
    p.add_argument("--output", required=True, help="CSV summary path")
    p.add_argument("--count_tol", type=float, default=0.5,
                   help="Max relative count diff |P-N|/max(P,N) to pass")
    p.add_argument("--loss_tol", type=float, default=float("inf"),
                   help="Max relative mean-loss diff to pass")
    p.add_argument("--min_count", type=int, default=200,
                   help="Both tails must have at least this many tokens")
    return p.parse_args()


def collect_tails(jsonl_path, losses):
    """Return (pos_losses, neg_losses, pos_abs_corrs, neg_abs_corrs)."""
    pos_losses, neg_losses = [], []
    pos_abs, neg_abs = [], []
    with open(jsonl_path, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            loss = losses[obj["w_idx"]]
            if not np.isfinite(loss):
                continue
            best_c, best_abs = 0.0, -1.0
            for ft in obj["feats"]:
                ac = abs(ft["c"])
                if ac > best_abs:
                    best_abs = ac
                    best_c = ft["c"]
            if best_c > POS_THR:
                pos_losses.append(loss)
                pos_abs.append(best_abs)
            elif best_c < NEG_THR:
                neg_losses.append(loss)
                neg_abs.append(best_abs)
    return (np.asarray(pos_losses, dtype=np.float32),
            np.asarray(neg_losses, dtype=np.float32),
            np.asarray(pos_abs, dtype=np.float32),
            np.asarray(neg_abs, dtype=np.float32))


def rel_diff(a, b):
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 else 0.0


def main():
    args = parse_args()
    losses = np.load(args.losses)
    print(f"Loaded {len(losses)} losses ({np.isfinite(losses).sum()} finite)")

    files = sorted(glob.glob(os.path.join(args.task_mapped_dir, "*_mapped.jsonl")))
    print(f"Found {len(files)} tasks\n")

    rows = []
    for path in tqdm(files, desc="Tasks"):
        task = os.path.basename(path).replace("_mapped.jsonl", "")
        pl, nl, pa, na = collect_tails(path, losses)
        n_pos, n_neg = len(pl), len(nl)

        row = {
            "task": task,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "mean_loss_pos": float(pl.mean()) if n_pos else float("nan"),
            "mean_loss_neg": float(nl.mean()) if n_neg else float("nan"),
            "median_loss_pos": float(np.median(pl)) if n_pos else float("nan"),
            "median_loss_neg": float(np.median(nl)) if n_neg else float("nan"),
            "mean_absr_pos": float(pa.mean()) if n_pos else float("nan"),
            "mean_absr_neg": float(na.mean()) if n_neg else float("nan"),
        }

        if n_pos >= args.min_count and n_neg >= args.min_count:
            row["count_rel_diff"] = rel_diff(n_pos, n_neg)
            row["loss_rel_diff"] = rel_diff(row["mean_loss_pos"],
                                            row["mean_loss_neg"])
            row["absr_rel_diff"] = rel_diff(row["mean_absr_pos"],
                                            row["mean_absr_neg"])
            row["passes"] = (row["count_rel_diff"] <= args.count_tol
                             and row["loss_rel_diff"] <= args.loss_tol)
        else:
            row["count_rel_diff"] = float("nan")
            row["loss_rel_diff"] = float("nan")
            row["absr_rel_diff"] = float("nan")
            row["passes"] = False

        rows.append(row)

    # Print table
    hdr = (f"{'task':<24} {'n_pos':>7} {'n_neg':>7} "
           f"{'cnt_Δ':>6} {'μL_pos':>7} {'μL_neg':>7} {'L_Δ':>6} "
           f"{'μ|r|_pos':>8} {'μ|r|_neg':>8} pass")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "✓" if r["passes"] else " "
        print(f"{r['task']:<24} {r['n_pos']:>7d} {r['n_neg']:>7d} "
              f"{r['count_rel_diff']:>6.2f} "
              f"{r['mean_loss_pos']:>7.3f} {r['mean_loss_neg']:>7.3f} "
              f"{r['loss_rel_diff']:>6.2f} "
              f"{r['mean_absr_pos']:>8.3f} {r['mean_absr_neg']:>8.3f}  {flag}")

    n_pass = sum(r["passes"] for r in rows)
    print(f"\n{n_pass}/{len(rows)} tasks pass "
          f"(count_tol={args.count_tol}, loss_tol={args.loss_tol}, "
          f"min_count={args.min_count})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
