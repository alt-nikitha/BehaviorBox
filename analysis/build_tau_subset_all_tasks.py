"""Uniformly-sampled SAE subset across ALL tasks with a complete performance curve,
selected by Kendall tau between each token's checkpoint ordering and the task's.

One memmap pass covers every task at once: in the pairwise-sign representation
(s_ij = sign(p_j - p_i)), the Kendall numerator is a plain dot product, so
    tau(tokens, tasks) = S_tokens @ S_tasks.T / n_pairs
is a single matmul per chunk. Caches the full (N_tokens, N_tasks) tau matrix.

Sampling is uniform by RANK, not by threshold: each task contributes its top
`--per-task` tokens by tau. A pure threshold would be wildly unbalanced (at tau>=0.8
gsm8k has 2.0M tokens and medmcqa 115k), which starves the sparse shape families the
SAE most needs density for.

Usage:
  python build_tau_subset_all_tasks.py --per-task 100000
  python build_tau_subset_all_tasks.py --per-task 100000 --tau-min 0.7
  python build_tau_subset_all_tasks.py --tau-only          # just compute + report
"""
import argparse, json, pickle, time
from pathlib import Path

import numpy as np

from task_shape_content_clusters import CACHE_DIR, STEPS, open_memmap
from sample_by_task_centroids import DEFAULT_STEM, build_task_centroids
from kendall_shape_match import raw_task_perf

WORD_IDS = CACHE_DIR / "word_ids.pkl"
TAU_MATRIX = CACHE_DIR / "kendall_all_tasks.npz"
IU, JU = np.triu_indices(len(STEPS), k=1)


def task_sign_matrix(dedup=True):
    """(n_pairs, n_groups) sign matrix over tasks with a complete curve.

    Different tasks can induce the SAME checkpoint ordering (lambada/naturalqs and
    mmlu_other/mmlu_social_sciences do). Those are one shape family, not two, so with
    dedup=True they are merged into a single column -- otherwise "uniform per task"
    would silently give those families double weight."""
    cents = build_task_centroids(DEFAULT_STEM, len(STEPS))
    tasks = sorted(cents)
    sign_of = {t: np.sign(raw_task_perf(t)[JU] - raw_task_perf(t)[IU]) for t in tasks}
    if not dedup:
        return tasks, np.stack([sign_of[t] for t in tasks], axis=1).astype(np.float32)
    groups = {}
    for t in tasks:
        groups.setdefault(tuple(sign_of[t].tolist()), []).append(t)
    names = ["+".join(v) for v in groups.values()]
    S = np.stack([np.array(k, dtype=np.float32) for k in groups], axis=1)
    if len(names) < len(tasks):
        print(f"merged {len(tasks)} tasks into {len(names)} distinct orderings: "
              + ", ".join(n for n in names if "+" in n))
    return names, S


def compute_tau(tasks, S_task, chunk=1_000_000):
    if TAU_MATRIX.exists():
        z = np.load(TAU_MATRIX, allow_pickle=True)
        if list(z["tasks"]) == tasks:
            print(f"loaded cached tau matrix from {TAU_MATRIX}")
            return z["tau"]
        print("cached tau matrix has different tasks; recomputing")
    mm, edim, n_out = open_memmap()
    N = mm.shape[0]
    T = len(tasks)
    tau = np.zeros((N, T), np.float16)
    n_y = (S_task != 0).sum(0).astype(np.float32)      # per-task non-tied pairs
    t0 = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -n_out:], dtype=np.float32)
        S = np.sign(b[:, JU] - b[:, IU])               # (rows, 55)
        num = S @ S_task                               # (rows, T) concordant - discordant
        n_x = (S != 0).sum(1).astype(np.float32)[:, None]
        tau[s:e] = (num / (np.sqrt(n_x * n_y[None, :]) + 1e-9)).astype(np.float16)
        print(f"  {e:,}/{N:,} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(TAU_MATRIX, tau=tau, tasks=np.array(tasks))
    print(f"cached -> {TAU_MATRIX}")
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=0,
                    help="tokens per family; 0 = auto (the smallest family's pool, so "
                         "that family is taken whole and the rest are matched to it)")
    ap.add_argument("--tau-min", type=float, default=0.7,
                    help="only tokens with tau >= this are eligible")
    ap.add_argument("--sample-mode", choices=["random", "top"], default="random",
                    help="random: draw uniformly from the eligible pool (keeps the tau "
                         "distribution above the threshold representative). "
                         "top: take the highest-tau tokens (biases toward the extreme).")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="families to drop entirely (matched against any task name in "
                         "a merged family, e.g. --exclude medmcqa)")
    ap.add_argument("--tau-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tasks, S_task = task_sign_matrix()
    print(f"{len(tasks)} tasks with complete curves: {', '.join(tasks)}\n")
    tau = compute_tau(tasks, S_task)

    if a.exclude:
        keep = [i for i, t in enumerate(tasks)
                if not any(x in t.split("+") for x in a.exclude)]
        dropped = [t for i, t in enumerate(tasks) if i not in keep]
        print(f"excluding: {', '.join(dropped)}")
        tau = tau[:, keep]
        tasks = [tasks[i] for i in keep]

    pools = {t: np.where(tau[:, i].astype(np.float32) >= a.tau_min)[0]
             for i, t in enumerate(tasks)}
    per_task = a.per_task or min(len(v) for v in pools.values())
    smallest = min(pools, key=lambda t: len(pools[t]))
    print(f"\ntau >= {a.tau_min}; per-family n = {per_task:,}"
          + (f" (auto: the {smallest} pool, taken whole)" if not a.per_task else ""))

    rng = np.random.default_rng(a.seed)
    print(f"\n{'family':<34}{'pool >= tau_min':>16}{'selected':>10}{'mean tau':>10}{'tau floor':>11}")
    sel = {}
    for i, t in enumerate(tasks):
        col = tau[:, i].astype(np.float32)
        pool = pools[t]
        if len(pool) <= per_task:
            idx = pool
        elif a.sample_mode == "random":
            idx = rng.choice(pool, per_task, replace=False)
        else:
            idx = pool[np.argsort(-col[pool])[:per_task]]
        sel[t] = idx
        print(f"{t:<34}{len(pool):>16,}{len(idx):>10,}"
              f"{col[idx].mean():>10.3f}{col[idx].min():>11.3f}")
    if a.tau_only:
        return

    union = np.unique(np.concatenate([sel[t] for t in tasks]))
    member = {t: np.isin(union, sel[t]) for t in tasks}
    n_hits = np.sum([member[t] for t in tasks], axis=0)
    print(f"\nunion: {len(union):,} unique tokens "
          f"({100*len(union)/tau.shape[0]:.2f}% of corpus); "
          f"sum of per-task = {sum(len(v) for v in sel.values()):,}")
    print(f"tokens claimed by 1 task: {(n_hits==1).sum():,}; "
          f"by 2+: {(n_hits>1).sum():,}; max tasks for one token: {n_hits.max()}")

    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    order = rng.permutation(len(union))
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / \
        f"sae_sample_by_tau_alltasks_t{a.tau_min}_n{per_task}.jsonl"
    tau_u = tau[union].astype(np.float32)
    with open(out, "w") as f:
        for i in order:
            hit = [t for t in tasks if member[t][i]]
            rec = {"word_id": str(wids[union[i]]),
                   "task": hit[0] if len(hit) == 1 else "multi",
                   "n_tasks": len(hit), "weight": 1.0}
            for t in hit:
                rec[f"tau_{t}"] = round(float(tau_u[i, tasks.index(t)]), 4)
            f.write(json.dumps(rec) + "\n")
    print(f"\nWROTE {len(union):,} tokens -> {out}")


if __name__ == "__main__":
    main()
