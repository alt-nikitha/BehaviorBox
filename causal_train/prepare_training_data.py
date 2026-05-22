"""
Build training data for the pos-up / neg-up causal experiments.

Two-phase pipeline:

  1) Shared phase (run once per model): tokenize the source corpus and save
     token_ids / attention_mask / global_word_ids arrays under {output_dir}/shared/.
  2) Per-task phase: load {task}_bestc.npz + shared global_word_ids, compute
     pos_mask / neg_mask via smooth exponential weighting on max-|c| signed
     correlation, and save only the per-task masks.

Weighting (tau=threshold, alpha=slope):
    pos_mask[i] = exp(alpha * max(0, c - tau))       # only c >  tau boosted
    neg_mask[i] = exp(alpha * max(0, -tau - c))      # only c < -tau boosted
Non-word active tokens and |c| <= tau → weight 1.0; padding → 0.

Layout:
    {output_dir}/shared/
        token_ids.npy            int32   (n_docs, max_length)
        attention_mask.npy       uint8   (n_docs, max_length)
        global_word_ids.npy      int32   (n_docs, max_length)   -1 where no word
        meta.json
    {output_dir}/{task}/
        pos_mask.npy             float32 (n_docs, max_length)
        neg_mask.npy             float32 (n_docs, max_length)
        stats.json

If the shared files already exist they are reused; pass --force_retokenize to rebuild.

Usage:
    python prepare_training_data.py \
        --tasks arc_challenge,bbh,... \
        --task_mapped_dir /data/.../olmo/task_mapped_results \
        --word_ids_path /data/.../olmo/raw_activations/word_ids.pkl \
        --source_data /data/.../olmo_256000_unseen.jsonl \
        --tokenizer allenai/Olmo-3-1025-7B \
        --tau 0.6 --alpha 5.0 \
        --output_dir /data/.../olmo/expweight_tasks


        python prepare_training_data.py \
        --tasks arc_challenge,bbh,... \
        --task_mapped_dir /data/.../olmo/task_mapped_results \
        --word_ids_path /data/.../olmo/raw_activations/word_ids.pkl \
        --source_data /data/.../olmo_256000_unseen.jsonl \
        --tokenizer allenai/Olmo-3-1025-7B \
        --tau 0.75 --alpha 8.0 \
        --output_dir /data/.../olmo/expweight_tasks
"""

import argparse
import json
import os
import pickle
import re
import time
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True, help="Comma-separated task names.")
    p.add_argument("--task_mapped_dir", required=True)
    p.add_argument("--word_ids_path", required=True)
    p.add_argument("--source_data", required=True)
    p.add_argument("--tokenizer", default="LLM360/Amber")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--chunk_size", type=int, default=10000,
                   help="Docs processed per numpy chunk in the per-task phase.")
    p.add_argument("--force_retokenize", action="store_true")
    return p.parse_args()


def text_to_word_boundaries(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def build_local_word_ids(encoding, text):
    """Map each token position to the local word index (-1 if none)."""
    offsets = encoding["offset_mapping"]
    word_bounds = text_to_word_boundaries(text)
    out = np.full(len(offsets), -1, dtype=np.int32)
    if not word_bounds:
        return out
    wp = 0
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        while wp < len(word_bounds) - 1 and word_bounds[wp][1] <= s:
            wp += 1
        ws, we = word_bounds[wp]
        if s < we and e > ws:
            out[i] = wp
    return out


def build_doc_start(word_ids):
    """From flat word_ids list (['<doc>_<word>']) build {doc_id: first_global_idx}.

    Assumes entries for a given doc are contiguous and local word indices start
    at 0 and increment by 1 (which is how get_token_correlations' upstream
    pipeline emits them).
    """
    doc_start = {}
    prev_d = None
    for gi in tqdm(range(len(word_ids)), desc="Indexing word_ids"):
        s = word_ids[gi]
        sep = s.rfind("_")
        d = int(s[:sep])
        if d != prev_d:
            doc_start[d] = gi
            prev_d = d
    return doc_start


# ---------- shared tokenization phase ----------

def run_shared_phase(args):
    shared_dir = os.path.join(args.output_dir, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    meta_path = os.path.join(shared_dir, "meta.json")
    tok_path = os.path.join(shared_dir, "token_ids.npy")
    attn_path = os.path.join(shared_dir, "attention_mask.npy")
    gwid_path = os.path.join(shared_dir, "global_word_ids.npy")

    if (not args.force_retokenize and os.path.exists(tok_path)
            and os.path.exists(attn_path) and os.path.exists(gwid_path)
            and os.path.exists(meta_path)):
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta.get("tokenizer") == args.tokenizer
                and meta.get("max_length") == args.max_length
                and meta.get("source_data") == args.source_data):
            print(f"[shared] reusing existing tokenization at {shared_dir}")
            return shared_dir, meta["n_docs"]

    print(f"[shared] tokenizing {args.source_data} with {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.source_data) as f:
        source_docs = [json.loads(line) for line in f]
    n_docs = len(source_docs)
    L = args.max_length

    print(f"[shared] loading word_ids from {args.word_ids_path}")
    with open(args.word_ids_path, "rb") as f:
        word_ids = pickle.load(f)
    doc_start = build_doc_start(word_ids)
    n_words_total = len(word_ids)

    token_ids = np.zeros((n_docs, L), dtype=np.int32)
    attn = np.zeros((n_docs, L), dtype=np.uint8)
    gwid = np.full((n_docs, L), -1, dtype=np.int32)

    for row, doc in enumerate(tqdm(source_docs, desc="Tokenizing")):
        doc_id = int(doc["id"])
        text = doc["text"]
        enc = tokenizer(text, add_special_tokens=False, max_length=L,
                        truncation=True, padding="max_length",
                        return_offsets_mapping=True)
        token_ids[row] = np.asarray(enc["input_ids"], dtype=np.int32)
        attn[row] = np.asarray(enc["attention_mask"], dtype=np.uint8)
        local = build_local_word_ids(enc, text)
        start = doc_start.get(doc_id)
        if start is not None:
            valid = local >= 0
            gwid_row = gwid[row]
            gwid_row[valid] = local[valid] + start

    np.save(tok_path, token_ids)
    np.save(attn_path, attn)
    np.save(gwid_path, gwid)
    with open(meta_path, "w") as f:
        json.dump({
            "tokenizer": args.tokenizer,
            "max_length": args.max_length,
            "source_data": args.source_data,
            "word_ids_path": args.word_ids_path,
            "n_docs": n_docs,
            "n_words_total": n_words_total,
        }, f, indent=2)
    print(f"[shared] wrote shared arrays to {shared_dir}")
    return shared_dir, n_docs


# ---------- per-task phase ----------

def build_corr_flat(npz_path, n_words_total):
    data = np.load(npz_path)
    w_idx = data["w_idx"]
    best_c = data["best_c"].astype(np.float32, copy=False)
    corr = np.zeros(n_words_total, dtype=np.float32)
    corr[w_idx] = best_c
    has = np.zeros(n_words_total, dtype=bool)
    has[w_idx] = True
    return corr, has


def process_task(task, args, shared_dir, n_docs, n_words_total):
    task_out = os.path.join(args.output_dir, task)
    os.makedirs(task_out, exist_ok=True)
    npz_path = os.path.join(args.task_mapped_dir, f"{task}_bestc.npz")
    if not os.path.exists(npz_path):
        print(f"[{task}] SKIP: {npz_path} missing")
        return

    t0 = time.time()
    corr, has = build_corr_flat(npz_path, n_words_total)

    L = args.max_length
    gwid = np.load(os.path.join(shared_dir, "global_word_ids.npy"), mmap_mode="r")
    attn = np.load(os.path.join(shared_dir, "attention_mask.npy"), mmap_mode="r")

    pos_mask = np.zeros((n_docs, L), dtype=np.float32)
    neg_mask = np.zeros((n_docs, L), dtype=np.float32)

    stats = {
        "pos_boosted": 0, "neg_boosted": 0,
        "pos_sum": 0.0, "neg_sum": 0.0,
        "pos_sqsum": 0.0, "neg_sqsum": 0.0,
        "pos_max": 0.0, "neg_max": 0.0,
    }

    tau = args.tau
    alpha = args.alpha
    cw = args.chunk_size

    for s in range(0, n_docs, cw):
        e = min(s + cw, n_docs)
        gw = np.asarray(gwid[s:e])                # (cw, L) int32
        am = np.asarray(attn[s:e]).astype(np.float32)

        idx = np.maximum(gw, 0)
        c = corr[idx]
        c[gw < 0] = 0.0                            # non-word tokens

        pw = np.exp(alpha * np.maximum(0.0, c - tau), dtype=np.float32) * am
        nw = np.exp(alpha * np.maximum(0.0, -tau - c), dtype=np.float32) * am

        pos_mask[s:e] = pw
        neg_mask[s:e] = nw

        active = am > 0
        stats["pos_boosted"] += int(((pw > 1.0) & active).sum())
        stats["neg_boosted"] += int(((nw > 1.0) & active).sum())
        stats["pos_sum"] += float(pw.sum())
        stats["neg_sum"] += float(nw.sum())
        stats["pos_sqsum"] += float((pw * pw).sum())
        stats["neg_sqsum"] += float((nw * nw).sum())
        stats["pos_max"] = max(stats["pos_max"], float(pw.max()))
        stats["neg_max"] = max(stats["neg_max"], float(nw.max()))

    stats["pos_ess"] = (stats["pos_sum"] ** 2 / stats["pos_sqsum"]) if stats["pos_sqsum"] > 0 else 0.0
    stats["neg_ess"] = (stats["neg_sum"] ** 2 / stats["neg_sqsum"]) if stats["neg_sqsum"] > 0 else 0.0

    np.save(os.path.join(task_out, "pos_mask.npy"), pos_mask)
    np.save(os.path.join(task_out, "neg_mask.npy"), neg_mask)
    with open(os.path.join(task_out, "stats.json"), "w") as f:
        json.dump({
            "task": task,
            "tau": tau, "alpha": alpha,
            "n_correlated_words": int(has.sum()),
            "stats": stats,
        }, f, indent=2)
    print(f"[{task}] done in {time.time()-t0:.1f}s "
          f"(pos_boosted={stats['pos_boosted']}, neg_boosted={stats['neg_boosted']}, "
          f"ess pos/neg = {stats['pos_ess']:.0f}/{stats['neg_ess']:.0f})")


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    shared_dir, n_docs = run_shared_phase(args)
    with open(os.path.join(shared_dir, "meta.json")) as f:
        n_words_total = json.load(f)["n_words_total"]

    for task in tasks:
        process_task(task, args, shared_dir, n_docs, n_words_total)


if __name__ == "__main__":
    main()
