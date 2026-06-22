"""Check whether a feature's probability jumps line up, in time, with a task's
performance jumps across training checkpoints.

Two notions of "jumps at the same spots":

  1. Spearman-of-diffs (ordering of jumps): z-normalize both curves, take
     consecutive-checkpoint differences, and Spearman-correlate the two diff
     vectors. High positive => the intervals where the feature rises most are
     the same intervals where the task rises most (their jump *orderings* agree).

  2. Argmax-jump coincidence (single biggest jump): the checkpoint interval of
     each curve's largest positive step; "hit" if they're the same interval
     (optionally within +/- a tolerance of one interval).

For one task it prints the features whose jumps best align, by Spearman-of-diffs,
alongside whether their biggest jump coincides with the task's.

Usage:
    python jump_alignment.py --task gsm8k
    python jump_alignment.py --task gsm8k --top 25 --tol 1
    python jump_alignment.py --task all          # summary line per task
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STEM = "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm"
VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


def znorm(vals):
    """Z-normalize a curve, dropping None entries. Returns (idxs, zvals) or None."""
    pairs = [(i, v) for i, v in enumerate(vals or []) if v is not None]
    if len(pairs) < 3:
        return None
    idxs = [i for i, _ in pairs]
    x = np.array([v for _, v in pairs], dtype=float)
    if x.std() < 1e-12:
        return None
    return idxs, (x - x.mean()) / x.std()


def diffs(curve):
    """Consecutive differences of a (idxs, zvals) curve, labeled by the interval's
    left checkpoint index. Returns (interval_idxs, dvals)."""
    idxs, z = curve
    return idxs[1:], np.diff(z)


def spearman_of_diffs(fcur, tcur):
    """Spearman correlation of the two curves' consecutive diffs over shared
    intervals. Returns rho or None."""
    fi, fd = diffs(fcur)
    ti, td = diffs(tcur)
    shared = sorted(set(fi) & set(ti))
    if len(shared) < 3:
        return None
    fmap, tmap = dict(zip(fi, fd)), dict(zip(ti, td))
    a = np.array([fmap[k] for k in shared])
    b = np.array([tmap[k] for k in shared])
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    rho, _ = stats.spearmanr(a, b)
    return None if rho != rho else float(rho)


def biggest_jump_interval(curve):
    """Left-checkpoint index of the largest positive consecutive step."""
    iv, dv = diffs(curve)
    return iv[int(np.argmax(dv))]


def load_task(stem, task):
    fp = os.path.join(ANALYSIS_DIR, f"precomputed_data_{task}", f"{stem}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def analyze_task(stem, task, top, tol, ckpt_names):
    data = load_task(stem, task)
    if data is None:
        print(f"[{task}] no data")
        return None
    tcur = znorm(data.get("overall_performance"))
    if tcur is None:
        print(f"[{task}] degenerate task curve")
        return None
    t_jump = biggest_jump_interval(tcur)

    rows = []
    seen = set()
    for feat in data.get("features", []):
        fid = feat["feature_id"]
        if fid in seen:
            continue
        seen.add(fid)
        fcur = znorm(feat.get("median_probs"))
        if fcur is None:
            continue
        rho = spearman_of_diffs(fcur, tcur)
        if rho is None:
            continue
        f_jump = biggest_jump_interval(fcur)
        hit = abs(f_jump - t_jump) <= tol
        rows.append((rho, hit, f_jump, fid, feat.get("description", "")))

    if not rows:
        print(f"[{task}] no comparable features")
        return None

    rows.sort(key=lambda r: -r[0])
    n = len(rows)
    coincide_rate = np.mean([r[1] for r in rows])

    def ckpt(i):
        return ckpt_names[i] if ckpt_names and i < len(ckpt_names) else f"idx{i}"

    print(f"\n[{task}]  features compared: {n}  |  task's biggest jump: "
          f"{ckpt(t_jump)}->{ckpt(t_jump+1)}")
    print(f"  overall jump-coincidence rate (|Δ|<= {tol} interval): "
          f"{coincide_rate:.2f}")
    print(f"  top {top} features by Spearman-of-diffs:")
    print(f"    {'rho':>5}  {'jump@':>14}  {'hit':>3}  feat  description")
    for rho, hit, fj, fid, desc in rows[:top]:
        js = f"{ckpt(fj)}->{ckpt(fj+1)}"
        print(f"    {rho:+.2f}  {js:>14}  {'Y' if hit else ' ':>3}  "
              f"f{fid}: {desc[:60]}")

    return {"task": task, "n": n, "task_jump": t_jump,
            "coincide_rate": float(coincide_rate),
            "mean_top_rho": float(np.mean([r[0] for r in rows[:top]]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--task", default="gsm8k",
                    help="Task name, or 'all' for a one-line summary per task.")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--tol", type=int, default=0,
                    help="Jump-coincidence tolerance in #intervals (0 = exact).")
    args = ap.parse_args()

    # checkpoint names (for readable jump labels) from any available task json
    ckpt_names = None
    for d in glob.glob(os.path.join(ANALYSIS_DIR, "precomputed_data_*")):
        fp = os.path.join(d, f"{args.stem}.json")
        if os.path.exists(fp):
            ckpt_names = json.load(open(fp)).get("checkpoints")
            if ckpt_names:
                break

    tasks = sorted(VALID_TASKS) if args.task == "all" else [args.task]
    summaries = []
    for t in tasks:
        s = analyze_task(args.stem, t, args.top, args.tol, ckpt_names)
        if s:
            summaries.append(s)

    if args.task == "all" and summaries:
        print(f"\n{'='*60}\nSUMMARY (jump-coincidence rate per task)\n{'='*60}")
        for s in sorted(summaries, key=lambda x: -x["coincide_rate"]):
            print(f"  {s['task']:22s} coincide={s['coincide_rate']:.2f}  "
                  f"mean_top_rho={s['mean_top_rho']:+.2f}  (n={s['n']})")


if __name__ == "__main__":
    main()
