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
    python precompute_task_shape_groups.py --dataset olmo-gsm8k --metric area
    python precompute_task_shape_groups.py --dataset <model_stem> --metric mse

Output: precomputed_task_shape/<dataset>__<metric>.pkl
"""

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

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
    diff = np.array([da[k] for k in shared]) - np.array([db[k] for k in shared])
    return shared, diff


def _shared_values(a, b):
    """Like _shared_diff but returns the two aligned value vectors (va, vb) over
    the shared checkpoints, rather than their difference. Returns None if fewer
    than three checkpoints overlap."""
    idx_a, v_a = a
    idx_b, v_b = b
    shared = sorted(set(idx_a) & set(idx_b))
    if len(shared) < 3:
        return None
    da = dict(zip(idx_a, v_a))
    db = dict(zip(idx_b, v_b))
    va = np.array([da[k] for k in shared])
    vb = np.array([db[k] for k in shared])
    return va, vb


def _parse_step(name):
    """Extract a numeric training step from a checkpoint name.

    Handles plain numbers and step-encoded names like 'stage1-step1000' (the
    OLMo convention). Returns None if no step can be parsed."""
    if isinstance(name, (int, float)):
        return float(name)
    s = str(name)
    m = re.search(r"step(\d+)", s)
    if m:
        return float(m.group(1))
    nums = re.findall(r"\d+", s)
    return float(nums[-1]) if nums else None


def _ckpt_x(shared, ckpts):
    """Map shared positional indices to numeric checkpoint coordinates for area
    integration. Returns None if ckpts is missing, indices out of range, or
    steps can't be parsed — caller falls back to unit spacing.

    Checkpoint names are step-encoded strings (e.g. 'stage1-step1000'), so we
    parse the step number rather than float()-ing the whole name. We then take
    log(step): the raw steps are roughly geometric (1k, 2k, 4k, ... 256k), so on
    a *linear* x-axis the final three intervals (~50k wide each) would own ~half
    the integrated area regardless of where the curve actually moves. log spacing
    makes the intervals roughly uniform, so every checkpoint contributes
    comparably and the metric reflects curve *shape* rather than late-step
    magnitude."""
    if not ckpts:
        return None
    try:
        x = np.array([_parse_step(ckpts[i]) for i in shared], dtype=float)
    except (TypeError, ValueError, IndexError):
        return None
    if np.any(np.isnan(x)) or np.any(x <= 0) or not np.all(np.diff(x) > 0):
        return None
    return np.log(x)


def _difference(curve):
    """First-difference a normalized curve (idxs, vals), indexed at the later
    endpoint of each consecutive pair. Returns None if too short.

    Differencing the z-curve makes the metric compare *when each curve changes*
    rather than its overall (near-monotone) level, so generic rising trends no
    longer look alike — surfacing features whose jumps align in time with a
    task's jumps."""
    idxs, vals = curve
    if len(vals) < 2:
        return None
    return idxs[1:], np.diff(vals)


def mse_distance(a, b, ckpts=None, diff=False):
    if diff:
        a, b = _difference(a), _difference(b)
        if a is None or b is None:
            return float("nan")
    res = _shared_diff(a, b)
    return float("nan") if res is None else float(np.mean(res[1] ** 2))


def area_distance(a, b, ckpts=None, diff=False):
    if diff:
        a, b = _difference(a), _difference(b)
        if a is None or b is None:
            return float("nan")
    res = _shared_diff(a, b)
    if res is None:
        return float("nan")
    shared, d = res
    x = _ckpt_x(shared, ckpts)
    if x is not None:
        return float(np.trapezoid(np.abs(d), x=x))
    return float(np.trapezoid(np.abs(d)))


def _spearman_rho(a, b):
    """Spearman rank correlation between two curves over their shared
    checkpoints, or None if degenerate (< 3 shared points or zero variance)."""
    res = _shared_values(a, b)
    if res is None:
        return None
    va, vb = res
    if np.std(va) < 1e-12 or np.std(vb) < 1e-12:
        return None
    rho, _ = stats.spearmanr(va, vb)
    return None if rho != rho else float(rho)


def spearman_distance(a, b, ckpts=None, diff=False):
    """Distance = 1 - Spearman rank correlation between the two curves over their
    shared checkpoints, so the value lies in [0, 2] with lower = better match
    (matching the distance semantics the rest of the pipeline assumes: nearest-
    task assignment minimizes it, specificity subtracts it). A flipped feature
    curve negates rho, so d^- = 1 + rho is handled correctly by the caller.

    Spearman (rank-based) is the standard training-dynamics measure of whether a
    feature trajectory co-moves monotonically with task performance, and unlike
    area it is invariant to checkpoint spacing, so `ckpts` is unused. With diff=
    True it correlates consecutive-checkpoint *changes* instead of levels."""
    if diff:
        a, b = _difference(a), _difference(b)
        if a is None or b is None:
            return float("nan")
    rho = _spearman_rho(a, b)
    return float("nan") if rho is None else float(1.0 - rho)


# Blend weight for spearman_combo: combined_rho = w*rho_level + (1-w)*rho_diff.
# Smaller w leans on jump-alignment (diff); larger w leans on global ordering
# (level). main() overrides this from --combo-weight.
COMBO_WEIGHT = 0.25


def spearman_combo_distance(a, b, ckpts=None, diff=False):
    """Distance = 1 - [w * rho_level + (1-w) * rho_diff], combining two Spearman
    correlations so that BOTH constraints are enforced:

      * rho_level: Spearman of the raw (z-normed) curves — rewards features whose
        checkpoint *ordering* matches the task's (global co-movement).
      * rho_diff:  Spearman of consecutive-checkpoint differences — rewards
        features whose *jumps* land at the same checkpoints as the task's
        (local timing alignment).

    w = COMBO_WEIGHT. Because most curves are near-monotone, rho_level is close
    to saturated and acts as an ordering gate/tiebreaker while rho_diff does the
    discriminating. Flipping the feature curve negates both terms, so the
    pipeline's d^- (flipped) branch is handled correctly. Spacing-invariant, so
    `ckpts` and the `diff` flag are unused (both components are computed here)."""
    rho_level = _spearman_rho(a, b)
    da, db = _difference(a), _difference(b)
    rho_diff = None if (da is None or db is None) else _spearman_rho(da, db)
    if rho_level is None or rho_diff is None:
        return float("nan")
    w = COMBO_WEIGHT
    rho = w * rho_level + (1.0 - w) * rho_diff
    return float(1.0 - rho)


METRICS = {"mse": mse_distance, "area": area_distance,
           "spearman": spearman_distance,
           "spearman_combo": spearman_combo_distance}


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


def load_feature_densities(tsg_data):
    """Map feature_id -> activation density (fraction of words on which the
    feature fires), read from the SAE's feature_densities.npy. The array is
    indexed by integer feature id. Returns {} if the file can't be found.

    Density is our frequency proxy: high-density features are the ubiquitous
    function-word / punctuation latents whose curves are the smoothest and thus
    spuriously win shape-match against any task's dominant transition."""
    folder = None
    for td in tsg_data.values():
        if td.get("folder"):
            folder = td["folder"]
            break
    if not folder:
        return {}
    dpath = Path(folder) / "feature_densities.npy"
    if not dpath.exists():
        print(f"  [warn] no feature_densities.npy at {dpath}; frequency filter disabled")
        return {}
    dens = np.load(dpath)
    return {str(i): float(dens[i]) for i in range(len(dens))}


def build_features(tsg_data, densities=None):
    """Returns (feature_curves, feature_meta, feature_samples). Feature samples
    are taken from the first task in which the feature appears. When `densities`
    is provided, each feature's activation density is recorded in its meta."""
    densities = densities or {}
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
                "density": densities.get(str(fid), float("nan")),
                "n_tasks": 1,
            }
            feature_samples[fid] = [dict(s) for s in (feat.get("samples") or [])]
    return feature_curves, feature_meta, feature_samples


def apply_frequency_filter(feature_curves, feature_meta, feature_samples,
                           max_density_pct):
    """Drop the highest-density (most frequent) features before scoring.

    `max_density_pct` is a percentile in (0, 1]: features whose density is above
    that percentile of the *kept-feature* density distribution are removed. This
    strips the ubiquitous function-word/punctuation latents that otherwise
    dominate every task's shape-match ranking. Features with unknown density are
    kept. Returns the filtered (curves, meta, samples) and the count dropped."""
    if not max_density_pct or max_density_pct >= 1.0:
        return feature_curves, feature_meta, feature_samples, 0
    dens = {fid: feature_meta[fid].get("density", float("nan"))
            for fid in feature_curves}
    known = np.array([v for v in dens.values() if v == v], dtype=float)
    if known.size == 0:
        print("  [warn] no known densities; frequency filter skipped")
        return feature_curves, feature_meta, feature_samples, 0
    cutoff = float(np.quantile(known, max_density_pct))
    keep = {fid for fid, v in dens.items() if not (v == v) or v <= cutoff}
    dropped = len(feature_curves) - len(keep)
    fc = {fid: c for fid, c in feature_curves.items() if fid in keep}
    fm = {fid: m for fid, m in feature_meta.items() if fid in keep}
    fs = {fid: s for fid, s in feature_samples.items() if fid in keep}
    print(f"  frequency filter: density cutoff={cutoff:.5f} "
          f"(pct={max_density_pct}); dropped {dropped}, kept {len(fc)}")
    return fc, fm, fs, dropped


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


def compute_dist_matrix(feature_curves, task_curves, distance_fn, ckpts=None,
                        diff=False):
    """Full d-score matrix. Returns {fid: {tname: (d_pos, d_neg)}}.
    When `ckpts` is provided, area_distance integrates over actual checkpoint
    spacing rather than treating adjacent checkpoints as unit-spaced.
    When `diff` is set, curves are first-differenced before comparison so the
    metric reflects when each curve jumps rather than its overall level."""
    matrix = {}
    for fid, fcurve in feature_curves.items():
        flipped = (fcurve[0], -fcurve[1])
        row = {}
        for tname, tcurve in task_curves.items():
            row[tname] = (
                distance_fn(tcurve, fcurve, ckpts=ckpts, diff=diff),
                distance_fn(tcurve, flipped, ckpts=ckpts, diff=diff),
            )
        matrix[fid] = row
    return matrix


def assign_features(dist_matrix, task_names):
    """Nearest-task assignment (closest centroid wins), pos and neg."""
    pos = {t: [] for t in task_names}
    neg = {t: [] for t in task_names}
    for fid, row in dist_matrix.items():
        best_pos = (None, float("inf"))
        best_neg = (None, float("inf"))
        for t in task_names:
            d_p, d_n = row[t]
            if not np.isnan(d_p) and d_p < best_pos[1]:
                best_pos = (t, d_p)
            if not np.isnan(d_n) and d_n < best_neg[1]:
                best_neg = (t, d_n)
        if best_pos[0] is not None:
            pos[best_pos[0]].append((fid, best_pos[1]))
        if best_neg[0] is not None:
            neg[best_neg[0]].append((fid, best_neg[1]))
    for t in task_names:
        pos[t].sort(key=lambda x: x[1])
        neg[t].sort(key=lambda x: x[1])
    return pos, neg


def compute_specificity(dist_matrix, task_names):
    """Per task, rank features by specificity = median(d_others) - d_task.
    Higher = feature fits this task more than the rest. Uses median (not mean)
    of others so one similar task can't tank the score.
    Returns (spec_pos, spec_neg) where each is {tname: [(fid, specificity,
    d_task, median_others), ...]} sorted descending by specificity."""
    spec_pos = {t: [] for t in task_names}
    spec_neg = {t: [] for t in task_names}
    for fid, row in dist_matrix.items():
        for t in task_names:
            d_p_t, d_n_t = row[t]
            others_p = [row[o][0] for o in task_names if o != t and not np.isnan(row[o][0])]
            others_n = [row[o][1] for o in task_names if o != t and not np.isnan(row[o][1])]
            if others_p and not np.isnan(d_p_t):
                med = float(np.median(others_p))
                spec_pos[t].append((fid, med - float(d_p_t), float(d_p_t), med))
            if others_n and not np.isnan(d_n_t):
                med = float(np.median(others_n))
                spec_neg[t].append((fid, med - float(d_n_t), float(d_n_t), med))
    for t in task_names:
        spec_pos[t].sort(key=lambda x: -x[1])
        spec_neg[t].sort(key=lambda x: -x[1])
    return spec_pos, spec_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Model JSON filename stem under precomputed_data_*/.")
    ap.add_argument("--metric", default="spearman_combo",
                    choices=["mse", "area", "spearman", "spearman_combo"],
                    help="Curve-similarity metric. Default spearman_combo: "
                         "blends Spearman of levels (ordering agrees) with "
                         "Spearman of diffs (jumps align).")
    ap.add_argument("--combo-weight", type=float, default=0.25,
                    help="spearman_combo blend weight w in [0,1]: "
                         "combined_rho = w*rho_level + (1-w)*rho_diff. Smaller w "
                         "leans on jump-alignment; larger on global ordering. "
                         "Default 0.25.")
    ap.add_argument("--diff", dest="diff", action="store_true", default=True,
                    help="For area/mse/spearman: compare differenced z-curves "
                         "(emphasize when curves jump rather than their overall "
                         "near-monotone level). Default on. Ignored by "
                         "spearman_combo, which computes both components itself.")
    ap.add_argument("--raw", dest="diff", action="store_false",
                    help="Compare raw z-curves instead of differenced.")
    ap.add_argument("--max-density-pct", type=float, default=0.9,
                    help="Frequency filter: drop features whose activation "
                         "density is above this percentile (0-1) of the density "
                         "distribution, removing ubiquitous function-word / "
                         "punctuation latents before ranking. Set to 1.0 to "
                         "disable. Default 0.9 (drop top decile).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output pickle path. Default: precomputed_task_shape/<dataset>__<metric>.pkl")
    ap.add_argument("--cache-dir", default=None,
                    help="Override the sample_curves memmap cache dir (e.g. "
                         "/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen/ofw=0.8_znorm"
                         "for the 0.8 dataset). Defaults to sample_curves.DEFAULT_CACHE.")
    args = ap.parse_args()

    global COMBO_WEIGHT
    COMBO_WEIGHT = args.combo_weight

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

    densities = load_feature_densities(tsg_data)
    feature_curves, feature_meta, feature_samples = build_features(
        tsg_data, densities=densities)
    print(f"  {len(feature_curves)} unique features, "
          f"{sum(len(v) for v in feature_samples.values())} total samples")

    feature_curves, feature_meta, feature_samples, n_dropped = (
        apply_frequency_filter(feature_curves, feature_meta, feature_samples,
                               args.max_density_pct))

    print(f"  computing per-sample distances ({args.metric})...")
    compute_sample_task_distances(feature_samples, task_curves, args.metric,
                                  cache_dir=args.cache_dir)

    distance_fn = METRICS[args.metric]
    mode = "differenced" if args.diff else "raw"
    print(f"  building feature × task distance matrix ({args.metric}, {mode} z-curves)...")
    dist_matrix = compute_dist_matrix(feature_curves, task_curves, distance_fn,
                                      ckpts=task_ckpts, diff=args.diff)
    task_names = sorted(task_curves.keys())

    print("  computing nearest-task assignments...")
    pos, neg = assign_features(dist_matrix, task_names)

    print("  computing specificity rankings (median_d_others - d_task)...")
    spec_pos, spec_neg = compute_specificity(dist_matrix, task_names)

    payload = {
        "dataset_name": args.dataset,
        "metric": args.metric,
        "differenced": args.diff,
        "combo_weight": (args.combo_weight if args.metric == "spearman_combo"
                         else None),
        "max_density_pct": args.max_density_pct,
        "n_features_dropped_by_freq": n_dropped,
        "task_curves": task_curves,
        "task_ckpts": task_ckpts,
        "feature_curves": feature_curves,
        "feature_meta": feature_meta,
        "feature_samples": feature_samples,
        "assignments_pos": pos,
        "assignments_neg": neg,
        "dist_matrix": dist_matrix,
        "specificity_pos": spec_pos,
        "specificity_neg": spec_neg,
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
