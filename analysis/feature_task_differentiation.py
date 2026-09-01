"""What differentiates the tasks in terms of SAE features?

Each token in the training subset belongs to one or more task SHAPE families (it passed
tau>=0.7 for them). Each live SAE feature has a set of top-activating tokens. So for every
(feature, task) we can ask: does this feature fire on task T's tokens more than the
background rate? Enrichment = (feature's fraction of T-tokens) / (corpus fraction of
T-tokens). A feature with high enrichment for exactly one task is what "differentiates"
that task.

The headline number is the OPPOSITE of task-specificity: how many features are shared
across many tasks vs specific to one. Given the pairwise-sign SAE groups by trajectory
shape (+ content), and the tau families overlap heavily, the prior from this whole line
of work is that features are mostly shared -- this quantifies whether that holds.

Outputs:
  * per-task: the most enriched features with their content labels
  * specificity histogram: for each feature, how peaked is its task distribution
  * a task x task feature-overlap matrix (Jaccard of each task's top-feature set)

Usage:
  python feature_task_differentiation.py --sae <folder> --subset <jsonl>
"""
import argparse, collections, glob, json, os
import numpy as np
import pandas as pd

DEF_SAE = ("/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/"
           "n_only_early_and_late_olmo3_seed=42_ofw=0.5_psign_subset="
           "sae_sample_by_tau_alltasks_t0.7_n1856450_N=3000_k=25_lp=None")
DEF_SUBSET = ("/home/nsrikant/BehaviorBoxNew/analysis/"
              "sae_sample_by_tau_alltasks_t0.7_n1856450.jsonl")


def load_membership(subset):
    """word_id -> set(task family names); plus background counts."""
    mem = {}
    bg = collections.Counter()
    n = 0
    with open(subset) as f:
        for line in f:
            r = json.loads(line)
            tasks = [k[4:] for k in r if k.startswith("tau_")]
            mem[r["word_id"]] = tasks
            bg.update(tasks)
            n += 1
    return mem, bg, n


def load_labels(sae):
    for sub in ("feature_labels_validated", "feature_labels"):
        fps = glob.glob(os.path.join(sae, sub, "*.json"))
        if fps:
            d = json.load(open(fps[0]))
            return {str(k): v.get("Description", "").split("<SEP>")[0].strip()
                    for k, v in d.items()}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", default=DEF_SAE)
    ap.add_argument("--subset", default=DEF_SUBSET)
    ap.add_argument("--top-k", type=int, default=50,
                    help="top-activating tokens per feature to attribute")
    ap.add_argument("--min-tokens", type=int, default=20,
                    help="skip features with fewer resolved tokens")
    ap.add_argument("--per-task", type=int, default=12)
    a = ap.parse_args()

    print("loading membership ...", flush=True)
    mem, bg, n_tok = load_membership(a.subset)
    tasks = sorted(bg)
    bg_frac = {t: bg[t] / n_tok for t in tasks}
    print(f"{n_tok:,} tokens, {len(tasks)} task families")
    print("background token share per family:")
    for t in tasks:
        print(f"  {t:<34} {100*bg_frac[t]:5.2f}%")

    labels = load_labels(a.sae)
    df = pd.read_csv(os.path.join(a.sae, "top-50_activations.csv"),
                     usecols=["feature", "act_value", "word_id"])
    df = df[df["act_value"] > 0]

    # per feature: task-membership fraction among its top-k tokens
    ti = {t: i for i, t in enumerate(tasks)}
    feat_frac = {}          # fid -> (n_tasks,) fraction vector
    feat_ntok = {}
    for fid, g in df.groupby("feature"):
        g = g.nlargest(a.top_k, "act_value")
        cnt = np.zeros(len(tasks))
        m = 0
        for w in g["word_id"].astype(str):
            ts = mem.get(w)
            if ts is None:
                continue
            m += 1
            for t in ts:
                cnt[ti[t]] += 1
        if m >= a.min_tokens:
            feat_frac[int(fid)] = cnt / m
            feat_ntok[int(fid)] = m
    print(f"\n{len(feat_frac)} live features with >= {a.min_tokens} resolved tokens")

    bgv = np.array([bg_frac[t] for t in tasks])
    F = np.array([feat_frac[f] for f in sorted(feat_frac)])
    fids = sorted(feat_frac)
    enr = (F + 1e-9) / (bgv + 1e-9)          # (n_feat, n_tasks)

    # ---- specificity: is each feature peaked on one task or spread? ----
    # normalized entropy of each feature's enrichment profile (0 = one task, 1 = uniform)
    p = enr / enr.sum(1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(1) / np.log(len(tasks))
    print("\n=== feature specificity (normalized entropy of task-enrichment) ===")
    print(f"  median={np.median(ent):.3f}  (0 = task-specific, 1 = shared across all)")
    for lo, hi, lab in [(0, .5, "specific  (<0.5)"), (.5, .8, "moderate  (0.5-0.8)"),
                        (.8, 1.01, "shared    (>0.8)")]:
        m = (ent >= lo) & (ent < hi)
        print(f"  {lab}: {m.sum():>4} features ({100*m.mean():4.1f}%)")

    # ---- per-task top features ----
    print("\n=== most task-differentiating features per family ===")
    for j, t in enumerate(tasks):
        order = np.argsort(-enr[:, j])[:a.per_task]
        print(f"\n--- {t}  (bg {100*bg_frac[t]:.1f}%) ---")
        for idx in order:
            fid = fids[idx]
            print(f"  f{fid:<5} enr={enr[idx,j]:4.1f}  frac={100*F[idx,j]:4.0f}%  "
                  f"spec={ent[idx]:.2f}  n={feat_ntok[fid]:<3} "
                  f"{labels.get(str(fid),'')[:70]}")

    # ---- task x task overlap: do tasks share their top features? ----
    K = 100
    topsets = {t: set(np.argsort(-enr[:, j])[:K]) for j, t in enumerate(tasks)}
    print(f"\n=== Jaccard overlap of each task's top-{K} feature sets ===")
    print("       " + "".join(f"{t[:6]:>7}" for t in tasks))
    for a_i, ta in enumerate(tasks):
        row = []
        for b_i, tb in enumerate(tasks):
            j = len(topsets[ta] & topsets[tb]) / len(topsets[ta] | topsets[tb])
            row.append(f"{j:>7.2f}")
        print(f"{ta[:6]:<6} " + "".join(row))

    out = os.path.join(os.path.dirname(a.subset), "feature_task_enrichment.csv")
    pd.DataFrame(enr, index=fids, columns=tasks).to_csv(out)
    print(f"\nwrote per-feature enrichment matrix -> {out}")


if __name__ == "__main__":
    main()
