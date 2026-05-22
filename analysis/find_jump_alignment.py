#!/usr/bin/env python3
"""Difference-based change-point analysis.

For each feature: compute median prob across samples per checkpoint, then find
the largest single-checkpoint jump (max |p[t+1] - p[t]|). Same for each task's
overall_performance. Group features by which checkpoint transition the jump
occurs in, and report co-occurrence with task jumps.
"""

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/nsrikant/BehaviorBoxNew/analysis")
DEFAULT_MODEL = "OLMo3-7b-256k-3000-k10-0.8-late-checkpoints"
DEFAULT_SAE_FOLDER = Path(
    "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_skip1_seed=42_ofw=0.8_N=3000_k=25_lp=None_znorm_odlw=auto"
)
COLUMN_PREFIX = "olmo3-"


# DEFAULT_MODEL = "Amber-300-3000-k25-0.8-late-checkpoints"
# DEFAULT_SAE_FOLDER = Path(
#     "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.8_varfilt=0.2_N=3000_k=10_lp=None_znorm_odlw=auto"
# )
# COLUMN_PREFIX = "amber-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sae-folder", type=Path, default=DEFAULT_SAE_FOLDER)
    ap.add_argument("--column-prefix", default=COLUMN_PREFIX)
    ap.add_argument("--min-feature-jump", type=float, default=0.0,
                    help="ignore features whose max |Δ median prob| is below this")
    ap.add_argument("--min-task-jump", type=float, default=0.0,
                    help="ignore tasks whose max |Δ performance| is below this")
    ap.add_argument("--out-json", type=Path,
                    default=ROOT / f"jump_alignment_{COLUMN_PREFIX[:-1]}.json")
    ap.add_argument("--out-txt", type=Path,
                    default=ROOT / f"jump_alignment_{COLUMN_PREFIX[:-1]}.txt")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / f"precomputed_data_*/{args.model}.json")))
    if not files:
        raise SystemExit(f"No precomputed JSONs for model {args.model}")

    t0 = time.time()
    act_csv = args.sae_folder / "top-50_activations.csv"
    print(f"Loading activations CSV: {act_csv}")
    act_df = pd.read_csv(act_csv)
    act_df["feature"] = act_df["feature"].astype(str)

    print(f"Loading {len(files)} task JSONs...")
    task_data_by_name = {}
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        task_data_by_name[d["task_name"]] = d
    task_names = sorted(task_data_by_name)

    first_ckpts = task_data_by_name[task_names[0]]["checkpoints"]
    ckpt_cols = [args.column_prefix + cp for cp in first_ckpts]
    available = set(act_df.columns)
    col_mask = np.array([c in available for c in ckpt_cols])
    present_cols = [c for c, ok in zip(ckpt_cols, col_mask) if ok]
    present_ckpts = [cp for cp, ok in zip(first_ckpts, col_mask) if ok]
    if len(present_cols) < 3:
        raise SystemExit(f"Need >=3 matching checkpoint columns; have {present_cols}")
    n_ck = len(present_cols)
    print(f"Using {n_ck} checkpoints: {present_ckpts}")

    # Transitions are labeled by the index of the *post-jump* checkpoint.
    transition_labels = [f"{present_ckpts[i]}->{present_ckpts[i+1]}"
                         for i in range(n_ck - 1)]

    # ---- Task jumps ----
    task_jumps = {}
    for name in task_names:
        d = task_data_by_name[name]
        if d.get("checkpoints") != first_ckpts:
            continue
        perf = np.array(d.get("overall_performance", []), dtype=float)
        if perf.shape[0] != col_mask.shape[0]:
            continue
        perf = perf[col_mask]
        deltas = np.diff(perf)
        if deltas.size == 0:
            continue
        i = int(np.argmax(np.abs(deltas)))
        if abs(deltas[i]) < args.min_task_jump:
            continue
        task_jumps[name] = {
            "transition_idx": i,
            "transition": transition_labels[i],
            "delta": float(deltas[i]),
            "perf_before": float(perf[i]),
            "perf_after": float(perf[i + 1]),
            "all_deltas": [float(x) for x in deltas],
        }
    print(f"Task jumps computed for {len(task_jumps)} tasks in {time.time()-t0:.1f}s")

    # ---- Feature jumps (median probability across samples) ----
    print("Computing per-feature median probability jumps...")
    col_idx = [act_df.columns.get_loc(c) for c in present_cols]
    arr_all = act_df.iloc[:, col_idx].to_numpy(dtype=float)
    feat_arr = act_df["feature"].to_numpy()

    feature_jumps = {}
    for fid in np.unique(feat_arr):
        idx = np.where(feat_arr == fid)[0]
        if idx.size == 0:
            continue
        probs = np.exp(arr_all[idx])  # samples x checkpoints
        med = np.median(probs, axis=0)
        deltas = np.diff(med)
        if deltas.size == 0:
            continue
        i = int(np.argmax(np.abs(deltas)))
        if abs(deltas[i]) < args.min_feature_jump:
            continue
        feature_jumps[str(fid)] = {
            "transition_idx": i,
            "transition": transition_labels[i],
            "delta": float(deltas[i]),
            "median_before": float(med[i]),
            "median_after": float(med[i + 1]),
            "n_samples": int(idx.size),
            "all_medians": [float(x) for x in med],
        }
    print(f"Feature jumps: {len(feature_jumps)} features in {time.time()-t0:.1f}s")

    # ---- Group features by transition; co-occurrence with tasks ----
    feats_by_trans = defaultdict(list)
    for fid, info in feature_jumps.items():
        feats_by_trans[info["transition_idx"]].append((fid, info))
    tasks_by_trans = defaultdict(list)
    for name, info in task_jumps.items():
        tasks_by_trans[info["transition_idx"]].append((name, info))

    by_transition = {}
    for i, label in enumerate(transition_labels):
        feats = sorted(feats_by_trans.get(i, []),
                       key=lambda x: -abs(x[1]["delta"]))
        tasks = sorted(tasks_by_trans.get(i, []),
                       key=lambda x: -abs(x[1]["delta"]))
        by_transition[label] = {
            "transition_idx": i,
            "n_features": len(feats),
            "n_tasks": len(tasks),
            "tasks": [{"task": n, **info} for n, info in tasks],
            "features": [{"feature_id": fid, **info} for fid, info in feats],
        }

    results = {
        "checkpoints": present_ckpts,
        "transitions": transition_labels,
        "task_jumps": task_jumps,
        "feature_jumps": feature_jumps,
        "by_transition": by_transition,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    # ---- TXT report ----
    lines = []
    lines.append("=" * 100)
    lines.append(f"Checkpoints ({n_ck}): {present_ckpts}")
    lines.append(f"Transitions: {transition_labels}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("TASK JUMPS (max |Δ overall_performance| per task)")
    lines.append("-" * 100)
    for name in sorted(task_jumps, key=lambda n: -abs(task_jumps[n]["delta"])):
        info = task_jumps[name]
        lines.append(f"  {info['delta']:+.4f}  @ {info['transition']:<25}  "
                     f"({info['perf_before']:.4f} -> {info['perf_after']:.4f})  {name}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("FEATURES & TASKS GROUPED BY JUMP TRANSITION")
    for i, label in enumerate(transition_labels):
        bucket = by_transition[label]
        lines.append("")
        lines.append("-" * 100)
        lines.append(f"Transition {i}: {label}   "
                     f"(n_tasks={bucket['n_tasks']}, n_features={bucket['n_features']})")
        if bucket["tasks"]:
            lines.append("  Tasks jumping here:")
            for t in bucket["tasks"]:
                lines.append(f"    {t['delta']:+.4f}  {t['task']}")
        else:
            lines.append("  (no tasks jump here)")
        if bucket["features"]:
            lines.append("  Features jumping here (by |Δ median prob|):")
            for f in bucket["features"]:
                lines.append(
                    f"    {f['delta']:+.4f}  f{f['feature_id']:<6}  "
                    f"med {f['median_before']:.4f} -> {f['median_after']:.4f}  "
                    f"n={f['n_samples']}"
                )
    lines.append("")

    # ---- Co-occurrence summary ----
    lines.append("=" * 100)
    lines.append("CO-OCCURRENCE SUMMARY")
    lines.append("-" * 100)
    lines.append(f"{'transition':<28} {'n_tasks':>8} {'n_features':>11}")
    for i, label in enumerate(transition_labels):
        b = by_transition[label]
        lines.append(f"{label:<28} {b['n_tasks']:>8} {b['n_features']:>11}")
    lines.append("")

    with open(args.out_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_txt}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
