"""
Build a flat per-word NLL array aligned with word_ids.pkl from the existing
output_features parquets (which already store per-word logprobs).

Output: float32 array `token_losses.npy` of length len(word_ids), where
losses[i] = -logprob for word_ids[i]. NaN if a word_id has no parquet entry.

Usage:
    python save_token_losses.py \
        --parquet_dir /data/user_data/nsrikant/bbox_data/output/amber_300_unseen/output_features/amber-ckpt_300 \
        --activation_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/raw_activations \
        --output /data/user_data/nsrikant/bbox_data/causal_data/amber/token_losses.npy

    python save_token_losses.py \
        --parquet_dir /data/user_data/nsrikant/bbox_data/output/olmo_256000_unseen/output_features/olmo3-stage1-step256000 \
        --activation_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/raw_activations \
        --output /data/user_data/nsrikant/bbox_data/causal_data/olmo/token_losses.npy
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", required=True,
                   help="Dir containing <prefix>-<start>_<end>.parquet files with "
                        "columns [word_id, doc_id, logprobs]")
    p.add_argument("--activation_dir", required=True,
                   help="Dir containing word_ids.pkl (the target indexing)")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    with open(os.path.join(args.activation_dir, "word_ids.pkl"), "rb") as f:
        word_ids = pickle.load(f)
    word_ids = np.asarray([str(w) for w in word_ids])
    n_words = len(word_ids)
    print(f"Loaded {n_words} word_ids")

    # Build word_id -> global_index lookup
    wid_to_gi = {wid: i for i, wid in enumerate(word_ids)}

    parquets = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    print(f"Found {len(parquets)} parquet files")

    out = np.full(n_words, np.nan, dtype=np.float32)
    n_unmatched = 0
    n_matched = 0

    for pq in tqdm(parquets, desc="Parquets"):
        df = pd.read_parquet(pq, columns=["word_id", "logprobs"])
        wids = df["word_id"].to_numpy()
        losses = -df["logprobs"].to_numpy(dtype=np.float32)

        # Vectorized lookup: map each row's word_id to global index
        gis = np.fromiter(
            (wid_to_gi.get(w, -1) for w in wids),
            dtype=np.int64, count=len(wids),
        )
        mask = gis >= 0
        out[gis[mask]] = losses[mask]
        n_matched += int(mask.sum())
        n_unmatched += int((~mask).sum())

    n_filled = int(np.isfinite(out).sum())
    print(f"Filled {n_filled}/{n_words} words ({100*n_filled/n_words:.1f}%)")
    print(f"  matched rows: {n_matched}, unmatched rows: {n_unmatched}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, out)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
