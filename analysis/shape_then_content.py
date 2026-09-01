"""Hacky shape-first pass: bucket tokens by trajectory shape, then look for
content structure *inside* the bucket.

Pipeline:
  1. KMeans on the 768-d Longformer embeddings of ALL cached tokens -> background
     content clusters (this is the null distribution over content).
  2. Define a shape group:
       --group task:<name>   tokens whose z-curve best correlates with a task curve
       --group jumpdrop:J,D  tokens in a (biggest-jump, biggest-drop) cell
  3. For each content cluster, compare the group's share against the background
     share. Enrichment > 1 means that content is over-represented in the shape.

Coherent clusters are NOT the result -- content clusters are coherent globally.
The result is whether the enrichment column departs from 1.0 beyond what a random
group of the same size does (printed as a control).

Usage:
  python shape_then_content.py --group task:medmcqa --top-n 5000 --k 40
  python shape_then_content.py --group jumpdrop:68,256 --k 40
"""
import argparse, json, pickle, re, collections
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sample_by_task_centroids import DEFAULT_STEM, build_task_centroids

SAE = ("/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/"
       "n_only_early_and_late_olmo3_seed=42_ofw=0.5_N=3000_k=25_lp=None_pznorm=0.01")
STEPS = [1, 2, 4, 8, 16, 32, 68, 103, 154, 205, 256]
COLS = [f"olmo3-stage1-step{s}000_zscore" for s in STEPS]


def load(sae):
    df = pd.read_csv(sae + "/top-50_activations.csv", usecols=["word_id"] + COLS)
    df = df.drop_duplicates("word_id").dropna(subset=COLS)
    emb = pickle.load(open(sae + "/topk_feature_word_embeddings.pkl", "rb"))
    df = df[df["word_id"].isin(emb)]
    wid = df["word_id"].values
    E = np.stack([emb[w] for w in wid]).astype(np.float32)
    T = df[COLS].values.astype(np.float64)
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-9)
    ctx = json.load(open(sae + "/top-50_words_in_context.json"))
    words = np.array([ctx.get(str(w), {}).get("word", "?").strip() for w in wid])
    return wid, E, T, words


def shape_group(spec, T, words, top_n):
    """Return a boolean mask over tokens for the requested shape bucket."""
    kind, _, arg = spec.partition(":")
    if kind == "task":
        cents = build_task_centroids(DEFAULT_STEM, len(STEPS))
        if arg not in cents:
            raise SystemExit(f"no curve for {arg}; have: {sorted(cents)}")
        c = cents[arg].astype(np.float64)
        c = (c - c.mean()) / c.std()
        r = (T @ c) / len(c)                      # both sides are z-normed -> pearson
        idx = np.argsort(-r)[:top_n]
        mask = np.zeros(len(T), bool); mask[idx] = True
        print(f"shape group task:{arg}  n={mask.sum()}  r range "
              f"{r[idx].min():.3f}..{r[idx].max():.3f}")
        return mask
    if kind == "jumpdrop":
        j, d = (int(x) for x in arg.split(","))
        dif = np.diff(T, axis=1)
        jump = np.array(STEPS[1:])[dif.argmax(1)]
        drop = np.array(STEPS[1:])[dif.argmin(1)]
        mask = (jump == j) & (drop == d)
        print(f"shape group jumpdrop:{j},{d}  n={mask.sum()}")
        if mask.sum() < 50:
            raise SystemExit("cell too small; try another (jump,drop)")
        return mask
    raise SystemExit("--group must be task:<name> or jumpdrop:J,D")


def distinctive(labels, c, words, bg_counts, n=8):
    """Words over-represented in cluster c relative to the whole token set."""
    cnt = collections.Counter(w for w in words[labels == c]
                              if w and re.search(r"[A-Za-z0-9]", w))
    tot = sum(cnt.values()) or 1
    bg_tot = sum(bg_counts.values())
    scored = [(w * 0 or w, (k / tot) / ((bg_counts[w] + 1) / bg_tot))
              for w, k in cnt.items() if k >= 3]
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:n]] or list(cnt)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="task:<name> | jumpdrop:J,D")
    ap.add_argument("--top-n", type=int, default=5000, help="group size for task mode")
    ap.add_argument("--k", type=int, default=40, help="number of content clusters")
    ap.add_argument("--sae", default=SAE)
    ap.add_argument("--show", type=int, default=12, help="clusters to print per end")
    a = ap.parse_args()

    wid, E, T, words = load(a.sae)
    print(f"{len(wid)} tokens with embeddings + curves")

    km = KMeans(n_clusters=a.k, n_init=4, random_state=0).fit(E)
    labels = km.labels_
    bg_share = np.bincount(labels, minlength=a.k) / len(labels)
    bg_counts = collections.Counter(w for w in words if w)

    mask = shape_group(a.group, T, words, a.top_n)
    grp_share = np.bincount(labels[mask], minlength=a.k) / mask.sum()

    rng = np.random.default_rng(0)
    ctrl = rng.choice(len(labels), mask.sum(), replace=False)
    ctrl_share = np.bincount(labels[ctrl], minlength=a.k) / mask.sum()

    enr = (grp_share + 1e-9) / (bg_share + 1e-9)
    ctrl_enr = (ctrl_share + 1e-9) / (bg_share + 1e-9)
    print(f"\nenrichment spread: group sd={enr.std():.3f}  random-control sd="
          f"{ctrl_enr.std():.3f}  (group must clearly exceed the control)")
    print(f"max group enrichment={enr.max():.2f}  max control={ctrl_enr.max():.2f}\n")

    order = np.argsort(-enr)
    for tag, ids in (("OVER-represented", order[:a.show]),
                     ("UNDER-represented", order[::-1][:a.show])):
        print(f"--- {tag} content clusters ---")
        for c in ids:
            n_in = int((labels[mask] == c).sum())
            print(f"  C{c:<3} enr={enr[c]:5.2f}  n={n_in:<5} "
                  f"({bg_share[c]*100:4.1f}% of corpus)  "
                  + ", ".join(distinctive(labels, c, words, bg_counts)))
        print()


if __name__ == "__main__":
    main()
