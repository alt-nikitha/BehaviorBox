"""Extract top-K OLMO features for a given task by z-norm distance,
with current description + top samples, for renaming review.

Usage:
  python extract_top_features_for_review.py --task arc_challenge [--top-k 20] [--metric area|mse]
"""

import argparse
import json
import sys

sys.path.insert(0, "/home/nsrikant/BehaviorBoxNew/analysis")

from task_shape_groups_olmo import (
    load_native_data, z_normalize, METRIC_FNS, DEFAULT_MODEL_ID,
)
from sample_curves import sample_distance_to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="Task name (e.g. arc_challenge).")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--metric", choices=["area", "mse"], default="area")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--max-samples", type=int, default=15,
                    help="Number of samples (closest by z-dist) to keep per feature.")
    args = ap.parse_args()

    ckpts, task_perf, feat_meta = load_native_data(args.model_id)
    if args.task not in task_perf:
        raise SystemExit(f"task {args.task!r} not in available tasks: {sorted(task_perf)}")

    dist_fn = METRIC_FNS[args.metric]
    task_z = z_normalize(task_perf[args.task])
    if task_z is None:
        raise SystemExit(f"task {args.task!r} has no valid z-normalized curve")

    rows = []
    for fid, meta in feat_meta.items():
        if not meta.get("description"):
            continue
        fz = z_normalize(meta.get("median_probs"))
        if fz is None:
            continue
        d = dist_fn(task_z, fz)
        if d != d:
            continue
        rows.append((fid, float(d), meta))

    rows.sort(key=lambda r: r[1])
    top = rows[: args.top_k]

    out = []
    for fid, d, meta in top:
        samples = meta.get("samples", [])
        enriched = []
        for s in samples:
            sd = sample_distance_to(task_z, s.get("word_id"), mode=args.metric)
            enriched.append((s, sd))
        enriched.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else float("inf")))
        sample_rows = []
        for s, sd in enriched[: args.max_samples]:
            sample_rows.append({
                "before": str(s.get("before", "")),
                "word": str(s.get("word", "")),
                "after": str(s.get("after", "")),
                "act": float(s.get("activation", 0) or 0),
                "cos": float(s.get("cos_sim", 0) or 0),
                "z_dist": (None if sd is None else float(sd)),
            })
        out.append({
            "fid": fid,
            "dist": d,
            "metric": args.metric,
            "description": meta.get("description", ""),
            "samples": sample_rows,
        })

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
