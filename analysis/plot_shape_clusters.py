"""Preview how tokens cluster by trajectory vs content, to inspect the shape/content
decoupling before choosing an SAE stratification axis.

Uses the SAE folder's cached per-word Longformer embeddings + top-50 z-trajectories +
word contexts (no memmap needed). Writes a PNG grid of cluster mean z-trajectories.

Modes:
  content     KMeans on 768-d Longformer embeddings (semantic clusters; curves come out
              near-identical -> content and trajectory are decoupled).
  trajectory  KMeans on 11-d z-curves, K auto-selected by silhouette (distinct shapes,
              mixed content).
  jumpdrop    group by (biggest-jump interval, biggest-drop interval); plot largest cells.

Usage:
  python plot_shape_clusters.py --mode trajectory
  python plot_shape_clusters.py --mode content --k 16
  python plot_shape_clusters.py --mode jumpdrop
"""
import argparse, json, pickle, re, math, collections, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

SAE = "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/n_only_early_and_late_olmo3_seed=42_ofw=0.5_N=3000_k=25_lp=None_pznorm=0.01"
STEPS = [1, 2, 4, 8, 16, 32, 68, 103, 154, 205, 256]
COLS = [f"olmo3-stage1-step{s}000_zscore" for s in STEPS]


def load(sae):
    df = pd.read_csv(sae + "/top-50_activations.csv", usecols=["word_id"] + COLS)
    df = df.drop_duplicates("word_id").dropna(subset=COLS)
    emb = pickle.load(open(sae + "/topk_feature_word_embeddings.pkl", "rb"))
    df = df[df["word_id"].isin(emb)]
    wid = df["word_id"].values
    E = np.stack([emb[w] for w in wid]).astype(np.float32)
    T = df[COLS].values.astype(float); T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-9)
    ctx = json.load(open(sae + "/top-50_words_in_context.json"))
    wof = lambda w: ctx.get(str(w), {}).get("word", "?").strip()
    return wid, E, T, wof


def words(wid, mask, wof, n=5):
    out = []
    for w in wid[mask]:
        x = wof(w)
        if x and x not in out and re.search(r"[A-Za-z0-9]", x):
            out.append(x)
        if len(out) >= n:
            break
    return out


def grid(labels, order, T, wid, wof, title, out, color):
    K = len(order); cols = min(4, K); rows = math.ceil(K / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax in axes[K:]:
        ax.axis("off")
    for i, c in enumerate(order):
        ax = axes[i]; m = labels == c; mean = T[m].mean(0); sd = T[m].std(0)
        ax.axhline(0, color="#ccc", lw=.6)
        ax.plot(STEPS, mean, color=color, lw=2); ax.fill_between(STEPS, mean - sd, mean + sd, color=color, alpha=.12)
        ax.set_xscale("log"); ax.set_xticks([1, 8, 68, 256]); ax.set_xticklabels(["1k", "8k", "68k", "256k"], fontsize=7)
        ax.set_ylim(-2.3, 2.3)
        ax.set_title(f"C{c} (n={m.sum()}): " + ", ".join(words(wid, m, wof)), fontsize=8)
    fig.suptitle(title, fontsize=12); fig.supxlabel("checkpoint (log)"); fig.supylabel("z-normed prob")
    fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig(out, dpi=110); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["content", "trajectory", "jumpdrop"], default="trajectory")
    ap.add_argument("--sae", default=SAE)
    ap.add_argument("--k", type=int, default=None, help="fixed K (content); trajectory auto-selects if unset.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    wid, E, T, wof = load(a.sae)
    out = a.out or f"/home/nsrikant/BehaviorBoxNew/analysis/{a.mode}_clusters_preview.png"

    if a.mode == "content":
        K = a.k or 16
        lab = KMeans(K, n_init=4, random_state=0).fit_predict(StandardScaler().fit_transform(E))
        grid(lab, list(range(K)), T, wid, wof, f"Content clusters (Longformer emb, K={K})", out, "#c0392b")
    elif a.mode == "trajectory":
        sub = np.random.default_rng(0).choice(len(T), min(8000, len(T)), replace=False)
        best = None
        for K in ([a.k] if a.k else [3, 4, 5, 6, 8, 10, 12, 16]):
            km = KMeans(K, n_init=4, random_state=0).fit(T)
            s = silhouette_score(T[sub], km.labels_[sub])
            print(f"K={K:2d} silhouette={s:+.3f}")
            if best is None or s > best[1]:
                best = (K, s, km)
        K, s, km = best; print(f"auto-K={K} (sil={s:.3f})")
        order = sorted(range(K), key=lambda c: -(km.labels_ == c).sum())
        grid(km.labels_, order, T, wid, wof, f"Trajectory-shape clusters (11-d z-curves, K={K})", out, "#1f6feb")
    else:  # jumpdrop
        d = np.diff(T, 1); jump = d.argmax(1); drop = d.argmin(1)
        lbl = [f"{STEPS[i+1]}k" for i in range(10)]
        cell = jump * 10 + drop
        top = [c for c, _ in collections.Counter(cell.tolist()).most_common(16)]
        # relabel to 0..15 for plotting
        remap = {c: i for i, c in enumerate(top)}
        lab = np.array([remap.get(c, -1) for c in cell])
        titles = {remap[c]: f"jump@{lbl[c//10]} drop@{lbl[c%10]}" for c in top}
        K = len(top); cols = 4; rows = math.ceil(K / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), sharex=True, sharey=True); axes = np.array(axes).reshape(-1)
        for i in range(K):
            ax = axes[i]; m = lab == i; mean = T[m].mean(0)
            ax.axhline(0, color="#ccc", lw=.6); ax.plot(STEPS, mean, color="#8e44ad", lw=2)
            ax.fill_between(STEPS, mean - T[m].std(0), mean + T[m].std(0), color="#8e44ad", alpha=.12)
            ax.set_xscale("log"); ax.set_xticks([1, 8, 68, 256]); ax.set_xticklabels(["1k", "8k", "68k", "256k"], fontsize=7); ax.set_ylim(-2.3, 2.3)
            ax.set_title(f"{titles[i]} n={m.sum()}\n" + ", ".join(words(wid, m, wof, 4)), fontsize=7.5)
        fig.suptitle("Clusters by (biggest-jump, biggest-drop) interval — largest 16 cells", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(out, dpi=110); print("wrote", out)


if __name__ == "__main__":
    main()
