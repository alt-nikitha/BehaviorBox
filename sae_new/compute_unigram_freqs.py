import os
import pickle
from collections import Counter
import argparse
import pandas as pd

def find_word_id_files(cache_root):
    for root, _, files in os.walk(cache_root):
        if 'word_ids.pkl' in files:
            yield os.path.join(root, 'word_ids.pkl')

def count_unigrams(cache_root, word_id_to_word):
    ctr = Counter()
    for p in find_word_id_files(cache_root):
        with open(p, 'rb') as f:
            ids = pickle.load(f)
        # Look up actual words from the mapping
        words = [word_id_to_word.get(str(wid), '') for wid in ids]
        words = [w for w in words if w]  # Filter out missing/empty
        ctr.update(words)
    return ctr

def build_word_mapping(example_data_dir):
    import dask.dataframe as dd
    input_feat_dir = os.path.join(example_data_dir, "input_features")
    df = dd.read_parquet(input_feat_dir, columns=["word_id","word"]).compute()
    df["word_id"] = df["word_id"].astype(str)
    return dict(zip(df["word_id"], df["word"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache_root', help="root where cached dirs live (e.g. cache/<model_string>)")
    ap.add_argument('--example-data-dir', help="path to one original data_dir to map word_id->word", required=True)
    ap.add_argument('--out', help="output CSV path", default='unigram_freqs.csv')
    args = ap.parse_args()

    # Build word_id -> word mapping first
    mapping = build_word_mapping(args.example_data_dir)
    
    # Count unigrams using the mapping
    counts = count_unigrams(args.cache_root, mapping)
    
    rows = []
    for word, cnt in counts.most_common():
        rows.append({'word': word, 'count': cnt})

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == '__main__':
    main()

# python compute_unigram_freqs.py /home/nsrikant/.cache/n_moreearly_olmo3_7b/olmo_validation_texts --example-data-dir /data/user_data/nsrikant/bbox_data/output/olmo_validation_texts --out /home/nsrikant/.cache/n_moreearly_olmo3_7b/olmo_validation_texts/unigram_freqs.csv