"""Compare the two content-only family SAEs (fam1 68k-jump vs fam2 gradual).

Both SAEs reconstruct the SAME 768-d Longformer embedding space (ofw=0 zeroes the prob
block), so their decoder feature vectors are directly comparable. For each live fam1
feature we find its nearest fam2 feature by decoder cosine. If content is independent of
shape family, nearly every feature has a near-twin (cosine ~1) in the other SAE.

Outputs:
  * distribution of best-match cosine (are the feature sets the same?)
  * HTML: matched pairs side-by-side (top-10 samples each) + any fam1 features with NO
    good fam2 match (the family-specific ones, if they exist)

Usage:
  python compare_family_saes.py
"""
import argparse, html, json, os
import numpy as np
import pandas as pd
import torch

D = "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late"
F1 = f"{D}/n_only_early_and_late_olmo3_68k_fam_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_fam1_68kjump_n3000000_N=3000_k=50_lp=None"
F2 = f"{D}/n_only_early_and_late_olmo3_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_fam2_gradual_n3000000_N=3000_k=50_lp=None"
EDIM = 768


def load(sae):
    sd = torch.load(f"{sae}/sae.pt", map_location="cpu")
    W = sd["dec.weight"][:EDIM].T.numpy()          # (n_feat, 768) per-feature decoder
    df = pd.read_csv(f"{sae}/top-50_activations.csv", usecols=["feature", "act_value", "word_id"])
    df = df[df.act_value > 0]
    live = sorted(df.feature.unique().tolist())
    ctx = json.load(open(f"{sae}/top-50_words_in_context.json"))
    return W, set(live), df, ctx


def top_samples(df, ctx, fid, n=10):
    g = df[df.feature == fid].nlargest(n, "act_value")
    out = []
    for w in g.word_id.astype(str):
        e = ctx.get(w, {})
        out.append((e.get("before", "")[-40:], e.get("word", ""), e.get("after", "")[:35]))
    return out


def word_summary(df, ctx, fid, n=10):
    import collections, re
    g = df[df.feature == fid].nlargest(n, "act_value")
    ws = [ctx.get(str(w), {}).get("word", "").strip() for w in g.word_id]
    return collections.Counter(w for w in ws if w).most_common(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos-twin", type=float, default=0.9, help="cosine for a 'same feature'")
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()

    W1, live1, df1, ctx1 = load(F1)
    W2, live2, df2, ctx2 = load(F2)
    print(f"fam1: {len(live1)} live features   fam2: {len(live2)} live features")

    # normalize decoder rows; restrict to live features
    l1 = np.array(sorted(live1)); l2 = np.array(sorted(live2))
    A = W1[l1]; B = W2[l2]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    S = A @ B.T                                    # (n1, n2) cosine
    best = S.max(1); best_j = S.argmax(1)

    print(f"\n=== best fam2-match cosine for each live fam1 feature ===")
    for lo, hi, lab in [(0.9, 1.01, "near-twin (>=0.9)"), (0.7, 0.9, "similar (0.7-0.9)"),
                        (0.0, 0.7, "no match (<0.7)")]:
        m = (best >= lo) & (best < hi)
        print(f"  {lab}: {m.sum():>4} / {len(best)}  ({100*m.mean():4.1f}%)")
    print(f"  median best cosine = {np.median(best):.3f}")

    esc = html.escape
    def block(sae_tag, df, ctx, fid, cos=None):
        rows = "".join(
            f"<div class='s'><span class='c'>{esc(b)}</span>"
            f"<span class='w'>{esc(w)}</span><span class='c'>{esc(a_)}</span></div>"
            for b, w, a_ in top_samples(df, ctx, fid))
        head = f"{sae_tag} f{fid}" + (f" · cos={cos:.2f}" if cos is not None else "")
        summ = ", ".join(f"{repr(w)}×{c}" for w, c in word_summary(df, ctx, fid))
        return f"<div class='feat'><div class='fh'>{esc(head)}</div><div class='sm'>{esc(summ)}</div>{rows}</div>"

    parts = ["<title>fam1 vs fam2 content SAE features</title>", """<style>
body{font:13px/1.5 -apple-system,sans-serif;margin:0;padding:20px;background:#fafafa}
h2{font-size:16px;margin:18px 0 6px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:10px}
.feat{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:8px 10px}
.fh{font-weight:600;font-size:12px}.sm{color:#888;font-size:11px;margin-bottom:4px}
.s{font-family:ui-monospace,monospace;font-size:11px;white-space:pre-wrap;word-break:break-word;border-top:1px solid #f2f2f2;padding:1px 0}
.w{background:#ffe9a8;font-weight:600;padding:0 2px}.c{color:#888}</style>"""]

    # matched pairs (highest-cosine twins), then the least-matched fam1 features
    order_twin = np.argsort(-best)[:a.show]
    order_uniq = np.argsort(best)[:a.show]
    parts.append(f"<h2>Matched pairs — most similar fam1↔fam2 features (top {a.show})</h2>")
    for i in order_twin:
        f1 = int(l1[i]); f2 = int(l2[best_j[i]])
        parts.append("<div class='pair'>" + block("fam1", df1, ctx1, f1)
                     + block("fam2", df2, ctx2, f2, best[i]) + "</div>")
    parts.append(f"<h2>Least-matched fam1 features (candidate family-specific, top {a.show})</h2>")
    for i in order_uniq:
        f1 = int(l1[i]); f2 = int(l2[best_j[i]])
        parts.append("<div class='pair'>" + block("fam1", df1, ctx1, f1)
                     + block("fam2 (nearest)", df2, ctx2, f2, best[i]) + "</div>")

    out = "/home/nsrikant/BehaviorBoxNew/analysis/compare_family_saes.html"
    open(out, "w").write("\n".join(parts))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
