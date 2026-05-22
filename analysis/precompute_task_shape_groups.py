"""Precompute task-shape feature assignments and per-sample distances for the
Streamlit viewer (`task_shape_groups.py`). Run once per (dataset, metric);
the viewer reads the resulting pickle and only renders.

What gets computed:
  • Task centroids — z-normalized overall_performance per task.
  • Feature curves — z-normalized median_probs, deduped by feature_id.
  • Feature meta — description, median std, n_tasks the feature appears in.
  • Feature samples — taken from the feature's first occurrence (samples are
    task-independent for a given feature_id, so no pooling is necessary).
  • Assignments — for each feature, its single nearest task centroid in two
    flavors: positive (raw z-curve) and negative (flipped z-curve).
  • Per-sample distances — each sample's z-normalized output trajectory vs each
    task centroid, via `sample_curves.sample_distance_to`. None where the
    sample's word_id can't be resolved against the memmap.

Usage:
    python precompute_task_shape_groups.py --dataset <model_stem> --metric area
    python precompute_task_shape_groups.py --dataset <model_stem> --metric mse

Output: precomputed_task_shape/<dataset>__<metric>.pkl
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
OUT_DIR = ANALYSIS_DIR / "precomputed_task_shape"

VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


def normalize_curve(values):
    """Z-normalize a curve. Returns (valid_idxs, z_values) or None if degenerate."""
    valid = [(i, p) for i, p in enumerate(values or []) if p is not None]
    if len(valid) < 3:
        return None
    idxs = [i for i, _ in valid]
    vals = np.array([p for _, p in valid], dtype=float)
    sd = vals.std()
    if sd < 1e-9:
        return None
    return idxs, (vals - vals.mean()) / sd


def _shared_diff(a, b):
    idx_a, v_a = a
    idx_b, v_b = b
    shared = sorted(set(idx_a) & set(idx_b))
    if len(shared) < 3:
        return None
    da = dict(zip(idx_a, v_a))
    db = dict(zip(idx_b, v_b))
    return np.array([da[k] for k in shared]) - np.array([db[k] for k in shared])


def mse_distance(a, b):
    d = _shared_diff(a, b)
    return float("nan") if d is None else float(np.mean(d ** 2))


def area_distance(a, b):
    d = _shared_diff(a, b)
    return float("nan") if d is None else float(np.trapezoid(np.abs(d)))


METRICS = {"mse": mse_distance, "area": area_distance}


def load_dataset(dataset_name):
    out = {}
    for d in sorted(glob.glob(str(ANALYSIS_DIR / "precomputed_data_*"))):
        tname = Path(d).name.replace("precomputed_data_", "")
        if tname not in VALID_TASKS:
            continue
        fpath = Path(d) / f"{dataset_name}.json"
        if fpath.exists():
            with open(fpath) as f:
                out[tname] = json.load(f)
    return out


def build_task_centroids(tsg_data):
    task_curves = {}
    task_ckpts = []
    for tname, td in tsg_data.items():
        norm = normalize_curve(td.get("overall_performance"))
        if norm is not None:
            task_curves[tname] = norm
            if not task_ckpts:
                task_ckpts = list(td.get("checkpoints") or [])
    return task_curves, task_ckpts


def build_features(tsg_data):
    """Returns (feature_curves, feature_meta, feature_samples). Feature samples
    are taken from the first task in which the feature appears."""
    feature_curves = {}
    feature_meta = {}
    feature_samples = {}
    for tname, td in tsg_data.items():
        for feat in td.get("features", []):
            fid = feat["feature_id"]
            if fid in feature_curves:
                feature_meta[fid]["n_tasks"] += 1
                continue
            norm = normalize_curve(feat.get("median_probs"))
            if norm is None:
                continue
            feature_curves[fid] = norm
            sps = [s for s in (feat.get("std_probs") or []) if s is not None]
            feature_meta[fid] = {
                "desc": feat.get("description", ""),
                "median_std": float(np.median(sps)) if sps else float("nan"),
                "n_tasks": 1,
            }
            feature_samples[fid] = [dict(s) for s in (feat.get("samples") or [])]
    return feature_curves, feature_meta, feature_samples


def compute_sample_task_distances(feature_samples, task_curves, metric, cache_dir=None):
    """Populate each sample with `_task_dists`: {tname: float | None}.
    Silently skips if `sample_curves` / its memmap cache isn't available."""
    try:
        from sample_curves import sample_distance_to, DEFAULT_CACHE
    except Exception as e:
        print(f"  [warn] could not import sample_curves ({e}); skipping sample distances")
        for slist in feature_samples.values():
            for s in slist:
                s["_task_dists"] = {}
        return

    cdir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    print(f"  sample memmap cache: {cdir}")

    n_total = sum(len(v) for v in feature_samples.values())
    done = 0
    for fid, slist in feature_samples.items():
        for s in slist:
            wid = s.get("word_id")
            d_by_task = {}
            if wid:
                for tname, tcurve in task_curves.items():
                    try:
                        d_by_task[tname] = sample_distance_to(tcurve, wid, mode=metric,
                                                              cache_dir=cdir)
                    except Exception:
                        d_by_task[tname] = None
            s["_task_dists"] = d_by_task
            done += 1
            if done % 5000 == 0:
                print(f"    {done}/{n_total} samples")


def assign_features(feature_curves, task_curves, distance_fn):
    """Assign each feature to its nearest task centroid; both positive (raw
    z-curve) and negative (flipped z-curve)."""
    pos = {t: [] for t in task_curves}
    neg = {t: [] for t in task_curves}
    for fid, fcurve in feature_curves.items():
        flipped = (fcurve[0], -fcurve[1])
        best_pos = (None, float("inf"))
        best_neg = (None, float("inf"))
        for tname, tcurve in task_curves.items():
            d = distance_fn(tcurve, fcurve)
            if not np.isnan(d) and d < best_pos[1]:
                best_pos = (tname, d)
            d = distance_fn(tcurve, flipped)
            if not np.isnan(d) and d < best_neg[1]:
                best_neg = (tname, d)
        if best_pos[0] is not None:
            pos[best_pos[0]].append((fid, best_pos[1]))
        if best_neg[0] is not None:
            neg[best_neg[0]].append((fid, best_neg[1]))
    for t in pos:
        pos[t].sort(key=lambda x: x[1])
    for t in neg:
        neg[t].sort(key=lambda x: x[1])
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Model JSON filename stem under precomputed_data_*/.")
    ap.add_argument("--metric", required=True, choices=["mse", "area"])
    ap.add_argument("--out", type=Path, default=None,
                    help="Output pickle path. Default: precomputed_task_shape/<dataset>__<metric>.pkl")
    ap.add_argument("--cache-dir", default=None,
                    help="Override the sample_curves memmap cache dir (e.g. "
                         "/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen/ofw=0.8 "
                         "for the 0.8 dataset). Defaults to sample_curves.DEFAULT_CACHE.")
    args = ap.parse_args()

    out_path = args.out or (OUT_DIR / f"{args.dataset}__{args.metric}.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset!r}...")
    tsg_data = load_dataset(args.dataset)
    if not tsg_data:
        print(f"No task JSONs found for dataset {args.dataset!r}.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(tsg_data)} tasks loaded")

    task_curves, task_ckpts = build_task_centroids(tsg_data)
    print(f"  {len(task_curves)} valid task centroids")

    feature_curves, feature_meta, feature_samples = build_features(tsg_data)
    print(f"  {len(feature_curves)} unique features, "
          f"{sum(len(v) for v in feature_samples.values())} total samples")

    print(f"  computing per-sample distances ({args.metric})...")
    compute_sample_task_distances(feature_samples, task_curves, args.metric,
                                  cache_dir=args.cache_dir)

    print(f"  computing nearest-task assignments ({args.metric})...")
    distance_fn = METRICS[args.metric]
    pos, neg = assign_features(feature_curves, task_curves, distance_fn)

    payload = {
        "dataset_name": args.dataset,
        "metric": args.metric,
        "task_curves": task_curves,
        "task_ckpts": task_ckpts,
        "feature_curves": feature_curves,
        "feature_meta": feature_meta,
        "feature_samples": feature_samples,
        "assignments_pos": pos,
        "assignments_neg": neg,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
