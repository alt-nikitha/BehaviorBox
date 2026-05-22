import argparse
import json
import numpy as np
import os
import pickle
import sys
import torch
import orjson
from tqdm import tqdm # New requirement
# Add sae/ to path so we can import sae_utils
SAE_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sae")
sys.path.insert(0, SAE_CODE_DIR)

from sae_utils import load_sae, DTYPES, DataLoader, BackgroundDataLoader

# Base configuration
BASE_OUTPUT_DIR = "/data/user_data/nsrikant/bbox_data/causal_data"
CACHE_DIR = "/data/user_data/nsrikant/.cache"


'''
python save_activations.py --sae_dir "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.8_N=12000_k=25_lp=None"     --dataset_name olmo
python save_activations.py --sae_dir "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.8_N=12000_k=25_lp=None"     --dataset_name amber

'''

def parse_args():
    parser = argparse.ArgumentParser(description="High-performance SAE activation collection for L40")
    parser.add_argument("--sae_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8192) 
    return parser.parse_args()

def resolve_cached_data(sae_dir, cache_dir):
    with open(os.path.join(sae_dir, "config.json")) as f:
        cfg = json.load(f)

    ofw = cfg["output_feature_weight"]
    model_string = cfg["name"].split("_seed=")[0]

    all_data = []
    all_word_ids = []

    for data_dir in cfg["train_data_dirs"]:
        data_name = os.path.basename(data_dir)
        cached_dir = os.path.join(cache_dir, model_string, data_name)
        
        with open(os.path.join(cached_dir, "word_ids.pkl"), "rb") as f:
            all_word_ids.append(pickle.load(f))

        preprocessed_dir = os.path.join(cached_dir, f"ofw={ofw}")
        with open(os.path.join(preprocessed_dir, "cached_data_info.json")) as f:
            info = json.load(f)

        data = np.memmap(os.path.join(preprocessed_dir, "preprocessed_data.dat"), 
                         dtype=DTYPES[info["dtype"]], mode="r", shape=tuple(info["shape"]))
        all_data.append(data)

    return all_data, np.concatenate(all_word_ids), cfg

def save_chunk(output_dir, count, indices, values):
    if not indices: return
    idx_path = os.path.join(output_dir, f"indices_{count}.npy")
    val_path = os.path.join(output_dir, f"values_{count}.npy")
    np.save(idx_path, np.concatenate(indices, axis=0))
    np.save(val_path, np.concatenate(values, axis=0))

@torch.no_grad()
def run_collection(all_data, word_ids, encoder, cfg, output_dir, batch_size):
    """
    Saves in a binary sparse format:
    1. word_ids.pkl (The strings)
    2. indices.npy (The N x 2 matrix of [token_index, feature_id])
    3. values.npy (The N values)
    4. offsets.npy (Where each token's data starts in the indices/values arrays)
    """
    total_tokens = len(word_ids)
    all_indices = []
    all_values = []
    all_offsets = [0]
    
    current_total_sparse = 0
    global_row_offset = 0

    # We will save in chunks of 5M tokens to prevent RAM overflow
    chunk_size = 5_000_000
    chunk_count = 0

    pbar = tqdm(total=total_tokens, desc="Binary Extraction", unit="tokens")

    for data in all_data:
        dataloader = DataLoader(data, batch_size, indices=np.arange(data.shape[0]))
        bg_loader = BackgroundDataLoader(dataloader)

        for i, batch in enumerate(bg_loader):
            batch = batch.to(cfg["device"], dtype=torch.bfloat16, non_blocking=True)
            _, batch_acts, _, _ = encoder(batch, return_acts=True)

            # Thresholding
            batch_acts[batch_acts < 1e-4] = 0

            # Get sparse data
            coords = torch.nonzero(batch_acts) # [N, 2] -> (row, col)
            vals = batch_acts[coords[:, 0], coords[:, 1]]

            # Count activations per token in batch to build offsets
            # This is much faster than Python grouping
            counts = torch.bincount(coords[:, 0], minlength=batch.shape[0]).cpu().numpy()

            coords_np = coords.cpu().numpy()
            coords_np[:, 0] += global_row_offset
            all_indices.append(coords_np)
            all_values.append(vals.float().cpu().numpy())
            all_offsets.extend(counts)
            global_row_offset += batch.shape[0]
            
            current_total_sparse += len(vals)
            pbar.update(batch.shape[0])

            # Periodic Save to Disk to keep RAM lean
            if pbar.n >= (chunk_count + 1) * chunk_size:
                save_chunk(output_dir, chunk_count, all_indices, all_values)
                all_indices, all_values = [], []
                chunk_count += 1

    # Save final bits
    save_chunk(output_dir, chunk_count, all_indices, all_values)
    
    # Save the metadata for the whole run
    np.save(os.path.join(output_dir, "offsets.npy"), np.cumsum(all_offsets).astype(np.uint64))
    with open(os.path.join(output_dir, "word_ids.pkl"), "wb") as f:
        pickle.dump(word_ids, f)




def main():
    args = parse_args()
    
    # Load SAE and force to BF16 for L40 efficiency
    encoder, sae_cfg = load_sae(args.sae_dir)
    encoder.eval()
    encoder.to(dtype=torch.bfloat16)

    all_data, word_ids, _ = resolve_cached_data(args.sae_dir, args.cache_dir)

    out_dir = args.output_dir or os.path.join(BASE_OUTPUT_DIR, args.dataset_name, "raw_activations")
    os.makedirs(out_dir, exist_ok=True)
    

    print(f"Starting optimized run on {sae_cfg['device']} (BF16)")
    print(f"Target: {out_dir}\n")
    
    run_collection(all_data, word_ids, encoder, sae_cfg, out_dir, args.batch_size)
    
    print(f"\nSuccessfully saved {len(word_ids)} tokens.")

if __name__ == "__main__":
    main()