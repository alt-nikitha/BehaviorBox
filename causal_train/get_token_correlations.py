import argparse
import json
import numpy as np
import os
import pickle
import time
from tqdm import tqdm

# Constants derived from your environment
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
BASE_OUTPUT_DIR = "/data/user_data/nsrikant/bbox_data/causal_data"

'''
python get_token_correlations.py \
    --task all \
    --dataset Amber-300-16000-k200-0.7 \
    --activation_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/raw_activations


python get_token_correlations.py \
    --task all \
    --dataset Amber-300-12000-k25-0.8 \
    --activation_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/raw_activations


python get_token_correlations.py \
    --task all \
    --dataset OLMo3-7b-256k-12000-k25-0.8 \
    --activation_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/raw_activations
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Map binary activations to all task correlations")
    parser.add_argument("--task", default="all", help="Task name or 'all' to process all available")
    parser.add_argument("--dataset", required=True, help="e.g., Amber-300-16000-k200-0.5")
    parser.add_argument("--activation_dir", required=True, help="Path to binary activations folder")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()

def load_all_task_correlations(target_task, dataset):
    """
    Returns a dictionary of dictionaries:
    { "task_name": { feature_id: correlation_value, ... }, ... }
    """
    task_maps = {}
    
    # Identify task folders: they start with 'precomputed_data_'
    potential_folders = [d for d in os.listdir(ANALYSIS_DIR) if d.startswith("precomputed_data_")]
    
    for folder in potential_folders:
        task_name = folder.replace("precomputed_data_", "")
        
        # If user specified a single task, skip others
        if target_task != "all" and task_name != target_task:
            continue
            
        task_file = os.path.join(ANALYSIS_DIR, folder, f"{dataset}.json")
        if os.path.exists(task_file):
            with open(task_file, 'r') as f:
                data = json.load(f)
            
            task_maps[task_name] = {
                int(f["feature_id"]): f["partial_pearson_corr"] 
                for f in data["features"] if f["partial_pearson_corr"] is not None
            }
            print(f"Loaded {len(task_maps[task_name])} features for task: {task_name}")
            
    return task_maps

def main():
    args = parse_args()
    
    # 1. Load Task Data
    task_lookups = load_all_task_correlations(args.task, args.dataset)
    if not task_lookups:
        print(f"No task files found for dataset {args.dataset}")
        return

    # 2. Load Binary Metadata
    offsets = np.load(os.path.join(args.activation_dir, "offsets.npy"))
    with open(os.path.join(args.activation_dir, "word_ids.pkl"), "rb") as f:
        word_ids = pickle.load(f)
    
    # 3. Setup Output
    model_name = "amber" if "amber" in args.dataset.lower() else "olmo"
    out_dir = args.output_dir or os.path.join(BASE_OUTPUT_DIR, model_name, "task_mapped_results")
    os.makedirs(out_dir, exist_ok=True)
    
    # We will save one JSONL per task to keep analysis clean
    # Accumulate (w_idx, best_c) across chunks for a compact per-task .npz
    bestc_accum = {t_name: [] for t_name in task_lookups.keys()}

    # Build dense per-task correlation arrays for O(1) vectorized lookup
    max_fid = 0
    for tm in task_lookups.values():
        if tm:
            max_fid = max(max_fid, max(tm.keys()))
    n_feats = max_fid + 1
    task_arrays = {}
    for tname, tm in task_lookups.items():
        arr = np.full(n_feats, np.nan, dtype=np.float32)
        for fid, c in tm.items():
            arr[fid] = c
        task_arrays[tname] = arr

    # 4. Process Binary Chunks
    chunk_files = sorted([f for f in os.listdir(args.activation_dir) if f.startswith("indices_")])

    for chunk_name in tqdm(chunk_files, desc="Chunks", dynamic_ncols=True):
        chunk_idx = chunk_name.split("_")[1].split(".")[0]
        t0 = time.time()
        indices = np.load(os.path.join(args.activation_dir, f"indices_{chunk_idx}.npy"),
                          mmap_mode="r")
        # We only need column 0 (tok) and 1 (feat) once; copy into contiguous arrays
        tok_ids = np.ascontiguousarray(indices[:, 0])
        feat_ids = np.ascontiguousarray(indices[:, 1])
        del indices
        print(f"[chunk {chunk_idx}] loaded indices ({len(tok_ids):,} rows) "
              f"in {time.time()-t0:.1f}s")

        # Prefilter: any feat_id that's not referenced by any task contributes nothing
        t0 = time.time()
        any_task_valid = np.zeros(n_feats, dtype=bool)
        for arr in task_arrays.values():
            any_task_valid |= ~np.isnan(arr)
        in_range = feat_ids < n_feats
        safe_feat = np.where(in_range, feat_ids, 0)
        global_mask = in_range & any_task_valid[safe_feat]
        g_tok = tok_ids[global_mask]
        g_feat = feat_ids[global_mask]
        print(f"[chunk {chunk_idx}] prefilter kept {len(g_tok):,}/{len(tok_ids):,} "
              f"in {time.time()-t0:.1f}s")

        for tname, corr_arr in task_arrays.items():
            t0 = time.time()
            corrs = corr_arr[g_feat]
            mask = ~np.isnan(corrs)
            if not mask.any():
                continue
            sel_tok = g_tok[mask]
            sel_corr = corrs[mask].astype(np.float32, copy=False)
            np.round(sel_corr, 4, out=sel_corr)

            uniq, starts = np.unique(sel_tok, return_index=True)
            ends = np.append(starts[1:], len(sel_tok))

            abs_c = np.abs(sel_corr)
            max_abs = np.maximum.reduceat(abs_c, starts)
            group_id = np.repeat(np.arange(len(starts)), ends - starts)
            is_max = abs_c == max_abs[group_id]
            pos = np.nonzero(is_max)[0]
            pos_groups = group_id[pos]
            _, first_in_group = np.unique(pos_groups, return_index=True)
            best_c_group = sel_corr[pos[first_in_group]]
            bestc_accum[tname].append(
                (uniq.astype(np.int32, copy=False), best_c_group)
            )
            print(f"  [{tname}] {len(sel_tok):,} rows → {len(uniq):,} tokens "
                  f"in {time.time()-t0:.2f}s")

    # Write per-task .npz with (w_idx, best_c). If the same w_idx appears in
    # multiple chunks, keep the entry with the larger |c|.
    for tname, parts in bestc_accum.items():
        if not parts:
            continue
        w_idx = np.concatenate([p[0] for p in parts])
        best_c = np.concatenate([p[1] for p in parts])
        # Dedup by w_idx, keep max |c|
        order = np.lexsort((-np.abs(best_c), w_idx))
        w_idx = w_idx[order]
        best_c = best_c[order]
        keep = np.empty(len(w_idx), dtype=bool)
        keep[0] = True
        keep[1:] = w_idx[1:] != w_idx[:-1]
        w_idx = w_idx[keep]
        best_c = best_c[keep]
        np.savez(
            os.path.join(out_dir, f"{tname}_bestc.npz"),
            w_idx=w_idx, best_c=best_c,
        )
    print(f"\nProcessing complete. Results saved in: {out_dir}")

if __name__ == "__main__":
    main()