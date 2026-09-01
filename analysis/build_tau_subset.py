"""Build an SAE-training subset from tokens whose checkpoint ORDERING matches a task's,
by Kendall tau threshold.

Selection is rank-based (see kendall_shape_match.py): tau depends only on the ordering of
the 11 checkpoint probabilities, so it is invariant to any monotone transform and has an
exact permutation null -- unlike Pearson r, which saturates near 0.95 for nearly every
token and cannot discriminate.

The default threshold tau>=0.8 sits far out in the null (null sd ~0.25, p99 ~0.56), giving
a false-discovery rate under 0.5% against random orderings.

Emits one line per token: {"word_id", "task", "tau_<task>"..., "strict", "weight"}.
Tokens passing for several tasks appear ONCE with task="both"/"multi".

Usage:
  python build_tau_subset.py --tasks gsm8k blimp --tau-min 0.8
  python build_tau_subset.py --tasks gsm8k blimp --tau-min 0.85 --cap-per-task 1000000
"""
import argparse, json, pickle
from pathlib import Path

import numpy as np

from task_shape_content_clusters import CACHE_DIR

WORD_IDS = CACHE_DIR / "word_ids.pkl"
OUT_DEFAULT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--tau-min", type=float, default=0.8)
    ap.add_argument("--cap-per-task", type=int, default=None,
                    help="if a task exceeds this, keep its highest-tau tokens only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    taus, stricts = {}, {}
    for t in a.tasks:
        fp = CACHE_DIR / f"kendall_{t}.npz"
        if not fp.exists():
            raise SystemExit(f"missing {fp}; run kendall_shape_match.py --task {t} first")
        z = np.load(fp)
        taus[t], stricts[t] = z["tau"], z["strict"]

    sel = {}
    for t in a.tasks:
        idx = np.where(taus[t] >= a.tau_min)[0]
        n_all = len(idx)
        if a.cap_per_task and n_all > a.cap_per_task:
            idx = idx[np.argsort(-taus[t][idx])[:a.cap_per_task]]
        sel[t] = idx
        print(f"{t:<10} tau>={a.tau_min}: {n_all:,} tokens"
              + (f" -> capped to {len(idx):,}" if len(idx) != n_all else "")
              + f"  (tau {taus[t][idx].min():.3f}..{taus[t][idx].max():.3f}, "
                f"{stricts[t][idx].sum():,} strict)")

    union = np.unique(np.concatenate([sel[t] for t in a.tasks]))
    print(f"\nunion: {len(union):,} unique tokens "
          f"({100*len(union)/len(taus[a.tasks[0]]):.2f}% of corpus)")
    if len(a.tasks) > 1:
        for i, t1 in enumerate(a.tasks):
            for t2 in a.tasks[i+1:]:
                ov = len(np.intersect1d(sel[t1], sel[t2]))
                print(f"  overlap {t1} & {t2}: {ov:,} "
                      f"({100*ov/min(len(sel[t1]),len(sel[t2])):.1f}% of the smaller set)")

    member = {t: np.isin(union, sel[t], assume_unique=False) for t in a.tasks}
    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(union))

    stem = "_".join(a.tasks)
    out = Path(a.out) if a.out else OUT_DEFAULT / f"sae_sample_by_tau_{stem}_t{a.tau_min}.jsonl"
    with open(out, "w") as f:
        for i in order:
            row = union[i]
            hit = [t for t in a.tasks if member[t][i]]
            rec = {"word_id": str(wids[row]),
                   "task": hit[0] if len(hit) == 1 else "multi",
                   "weight": 1.0,
                   "strict": bool(any(stricts[t][row] for t in hit))}
            for t in hit:
                rec[f"tau_{t}"] = round(float(taus[t][row]), 4)
            f.write(json.dumps(rec) + "\n")
    print(f"\nWROTE {len(union):,} tokens -> {out}")
    n_multi = sum(1 for t in a.tasks for _ in [0]) and \
        int(np.sum(np.sum([member[t] for t in a.tasks], axis=0) > 1))
    print(f"  multi-task tokens: {n_multi:,}; single-task: {len(union)-n_multi:,}")


if __name__ == "__main__":
    main()
