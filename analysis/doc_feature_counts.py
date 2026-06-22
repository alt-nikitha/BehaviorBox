"""Show that some documents feed many more SAE features than others.

For each document (the `<doc_id>` prefix of every feature sample's word_id),
count how many DISTINCT features draw a top-sample from it, then report the
distribution. Deduplicates features across tasks so a feature is counted once.

Usage:
    python doc_feature_counts.py
    python doc_feature_counts.py --stem OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_TASKS = {
    "arc_challenge", "bbh", "hellaswag", "piqa", "winogrande", "csqa",
    "medmcqa", "mmlu_stem", "mmlu_social_sciences", "mmlu_other",
    "blimp", "coqa", "gsm8k", "lambada", "naturalqs",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stem",
        default="OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm",
        help="precomputed_data_<task>/<stem>.json filename stem.",
    )
    ap.add_argument("--top", type=int, default=20, help="How many top docs to list.")
    args = ap.parse_args()

    doc_feats = defaultdict(set)   # doc_id -> set(feature_id)
    seen = set()                   # dedup features across tasks
    files = glob.glob(os.path.join(ANALYSIS_DIR, f"precomputed_data_*/{args.stem}.json"))
    if not files:
        raise SystemExit(f"No precomputed_data_*/{args.stem}.json found.")

    for fp in files:
        tname = os.path.basename(os.path.dirname(fp)).replace("precomputed_data_", "")
        if tname not in VALID_TASKS:
            continue
        with open(fp) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            fid = feat["feature_id"]
            if fid in seen:
                continue
            seen.add(fid)
            for s in feat.get("samples", []):
                wid = str(s.get("word_id", ""))
                if "_" in wid:
                    doc_feats[wid.rsplit("_", 1)[0]].add(fid)

    counts = np.array([len(v) for v in doc_feats.values()])
    print(f"features (deduped): {len(seen)} | documents: {len(doc_feats)}")
    print(f"features-per-doc:  min={counts.min()}  median={int(np.median(counts))}  "
          f"mean={counts.mean():.2f}  max={counts.max()}")
    print(f"std={counts.std():.2f}  (spread is what shows docs differ)\n")

    print("distribution (features-per-doc percentiles):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:<2d} = {np.percentile(counts, p):.0f}")

    print("\nconcentration:")
    srt = np.sort(counts)[::-1]
    tot = counts.sum()
    for frac in [0.01, 0.05, 0.10]:
        n = max(1, int(len(counts) * frac))
        print(f"  top {int(frac*100):>2d}% of docs hold {100*srt[:n].sum()/tot:.1f}% "
              f"of all doc-feature incidences")

    print(f"\ntop {args.top} documents by # distinct features:")
    for doc, fs in sorted(doc_feats.items(), key=lambda kv: -len(kv[1]))[:args.top]:
        print(f"  doc {doc:>8}: {len(fs)} features")

    # ASCII histogram so the spread is obvious at a glance
    print("\nhistogram (features-per-doc):")
    hist, edges = np.histogram(counts, bins=12)
    width = max(hist)
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(50 * h / width) if width else ""
        print(f"  [{lo:5.0f},{hi:5.0f})  {h:6d}  {bar}")


if __name__ == "__main__":
    main()
