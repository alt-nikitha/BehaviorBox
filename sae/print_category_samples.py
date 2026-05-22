"""Print example data samples that fall in given trajectory categories.

Uses the same classifier as analyze_trajectory_distribution.py, looks up the
word text / doc / domain for a handful of sample indices per category by joining
against input_features/*.parquet via file_to_doc.csv.

Example:
    python print_category_samples.py \
    --cached_data_dir /home/nsrikant/.cache/n_moreearly_olmo3/olmo_256000_unseen/ofw=0.8 \
    --num_models 7 \
    --n_per_cat 20 \
    --out_file trend_analysis_olmo/category_samples.txt
"""

import click
import json
import os
import pickle

import numpy as np
import pandas as pd

from analyze_trajectory_distribution import (
    CATEGORIES,
    DTYPES,
    classify,
    load_probs,
)


def _doc_id_from_word_id(word_id: str) -> str:
    return word_id.rsplit("_", 1)[0]


def _decode_bpe(tok: str) -> str:
    """GPT-2-style BPE: 'Ġ' encodes a leading space, 'Ċ' a newline."""
    return tok.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def _context_window(words: list[str], idx: int, radius: int = 6) -> str:
    lo = max(0, idx - radius)
    hi = min(len(words), idx + radius + 1)
    pre = "".join(_decode_bpe(w) for w in words[lo:idx])
    tgt = _decode_bpe(words[idx])
    post = "".join(_decode_bpe(w) for w in words[idx + 1:hi])
    return f"{pre}<<{tgt}>>{post}".strip()


@click.command()
@click.option("--cached_data_dir", type=click.Path(exists=True), required=True)
@click.option("--num_models", type=int, required=True)
@click.option("--categories", multiple=True,
              default=("monotonic_decrease", "fall_then_plateau", "monotonic_increase"),
              help="Category names to sample from")
@click.option("--n_per_cat", type=int, default=10)
@click.option("--flat_range_percentile", type=float, default=10.0)
@click.option("--mono_frac", type=float, default=0.8)
@click.option("--stagnation_ratio", type=float, default=0.25)
@click.option("--seed", type=int, default=0)
@click.option("--context_radius", type=int, default=6)
@click.option("--out_file", type=click.Path(), default=None,
              help="If given, write the per-category sample output to this file")
def main(
    cached_data_dir: str,
    num_models: int,
    categories: tuple[str, ...],
    n_per_cat: int,
    flat_range_percentile: float,
    mono_frac: float,
    stagnation_ratio: float,
    seed: int,
    context_radius: int,
    out_file: str,
):
    out_fp = open(out_file, "w") if out_file else None

    def emit(line: str = ""):
        print(line)
        if out_fp is not None:
            out_fp.write(line + "\n")
            out_fp.flush()
    info = json.load(open(os.path.join(cached_data_dir, "cached_data_info.json")))
    model_names = info["model_names"]
    data_dir = info["data_dir"]  # e.g. /data/.../olmo_256000_unseen
    input_feature_dir = os.path.join(data_dir, "input_features")
    file_to_doc = pd.read_csv(os.path.join(input_feature_dir, "file_to_doc.csv"))
    doc_to_file = dict(zip(file_to_doc["doc_id"].astype(str), file_to_doc["file"]))

    word_ids_path = os.path.join(os.path.dirname(cached_data_dir.rstrip("/")), "word_ids.pkl")
    print(f"Loading word_ids from {word_ids_path} ...", flush=True)
    word_ids = pickle.load(open(word_ids_path, "rb"))

    print(f"Loading probs from {cached_data_dir} ...", flush=True)
    probs = load_probs(cached_data_dir, num_models)
    print(f"Probs shape: {probs.shape}", flush=True)
    assert len(word_ids) == len(probs), (len(word_ids), len(probs))

    traj_range = probs.max(axis=1) - probs.min(axis=1)
    flat_thresh = float(np.percentile(traj_range, flat_range_percentile))
    cat = classify(
        probs,
        flat_range_thresh=flat_thresh,
        mono_frac=mono_frac,
        stagnation_ratio=stagnation_ratio,
    )

    rng = np.random.default_rng(seed)

    for cat_name in categories:
        if cat_name not in CATEGORIES:
            print(f"[skip] unknown category: {cat_name}")
            continue
        cat_idx = CATEGORIES.index(cat_name)
        matches = np.nonzero(cat == cat_idx)[0]
        emit(f"\n=== {cat_name}: {len(matches)} total samples ===")
        if len(matches) == 0:
            continue
        take = rng.choice(matches, size=min(n_per_cat, len(matches)), replace=False)

        # Group by parquet file so we only open each file once
        rows = []
        for i in take:
            wid = str(word_ids[i])
            did = _doc_id_from_word_id(wid)
            fname = doc_to_file.get(did)
            rows.append((int(i), wid, did, fname))

        by_file: dict[str, list] = {}
        for row in rows:
            by_file.setdefault(row[3], []).append(row)

        resolved: dict[int, dict] = {}
        for fname, group in by_file.items():
            if fname is None:
                for (i, wid, did, _) in group:
                    resolved[i] = {"word": "<unknown file>", "domain": "?", "context": ""}
                continue
            pq_path = os.path.join(input_feature_dir, fname)
            want_ids = {r[1] for r in group}
            want_docs = {r[2] for r in group}
            df = pd.read_parquet(pq_path, columns=["word_id", "doc_id", "domain", "word"])
            df_docs = df[df["doc_id"].astype(str).isin(want_docs)]
            for (i, wid, did, _) in group:
                target_row = df_docs[df_docs["word_id"] == wid]
                if len(target_row) == 0:
                    resolved[i] = {"word": "<missing>", "domain": "?", "context": ""}
                    continue
                target_row = target_row.iloc[0]
                doc_words_df = df_docs[df_docs["doc_id"].astype(str) == did].sort_values("word_id", key=lambda s: s.str.rsplit("_", n=1).str[-1].astype(int))
                words = doc_words_df["word"].astype(str).tolist()
                word_index_in_doc = int(wid.rsplit("_", 1)[1])
                try:
                    local_idx = next(j for j, w_id in enumerate(doc_words_df["word_id"].tolist()) if w_id == wid)
                except StopIteration:
                    local_idx = 0
                ctx = _context_window(words, local_idx, radius=context_radius)
                resolved[i] = {
                    "word": str(target_row["word"]),
                    "domain": str(target_row["domain"]),
                    "context": ctx,
                    "word_index_in_doc": word_index_in_doc,
                }

        header = "  ".join(f"{m.split('-')[-1]:>10}" for m in model_names)
        emit(f"  probs @ checkpoints: {header}")
        for (i, wid, did, _) in rows:
            info = resolved.get(i, {})
            p = probs[i]
            p_str = "  ".join(f"{v:>10.4f}" for v in p)
            emit(f"\n  [idx={i} word_id={wid} doc={did} domain={info.get('domain','?')}]")
            emit(f"    word : {_decode_bpe(info.get('word','?'))!r}  (pos in doc: {info.get('word_index_in_doc','?')})")
            emit(f"    ctx  : {info.get('context','')}")
            emit(f"    probs: {p_str}")

    if out_fp is not None:
        out_fp.close()
        print(f"\nWrote samples to {out_file}")


if __name__ == "__main__":
    main()
