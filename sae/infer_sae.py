"""
Ad-hoc SAE inference: takes raw documents, runs Longformer + vLLM + SAE
in-memory, and outputs which features activate per word — no intermediate
parquet files.

Handles the full lifecycle:
  1. Launches vLLM servers for all required model checkpoints (via SLURM)
  2. Waits for all servers to be ready
  3. Runs Longformer embeddings + vLLM logprobs + SAE inference
  4. Cleans up all vLLM servers on exit

Usage:
    python infer_sae.py \
        --data /path/to/documents.jsonl \
        --sae_dir /path/to/trained_sae \
        --output /path/to/output.parquet \
        --training_cache_dir /home/nsrikant/.cache/n_moreearly_amber/amber_300_unseen/ofw=0.7

    # If vLLM servers are already running, skip launch:
    python infer_sae.py \
        --data /path/to/documents.jsonl \
        --sae_dir /path/to/trained_sae \
        --model_addrs amber-ckpt_001=host:port,amber-ckpt_300=host:port,... \
        --output /path/to/output.parquet \
        --training_cache_dir ...
"""

import atexit
import click
import gc
import json
import math
import numpy as np
import os
import signal
import subprocess
import sys
import time
import torch
import pandas as pd

from transformers import LongformerModel, AutoTokenizer
from tqdm import tqdm

sae_root = os.path.abspath(os.path.dirname(__file__))
if sae_root not in sys.path:
    sys.path.insert(0, sae_root)

data_gen_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_generation'))
if data_gen_root not in sys.path:
    sys.path.insert(0, data_gen_root)

from sae_utils import load_sae, DTYPES
from utils.input_features import get_longformer_word_features
from utils.output_features import get_output_overlapping_strings, get_model_logprobs
from utils.text_samples import filter_and_sort_data, get_batch_data

BBOX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_DIR = os.path.join(BBOX_ROOT, "scripts/data_generation/tmp")


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def launch_vllm_servers(
    model_names: list[str],
    model_id: str,
    use_slurm: bool = True,
) -> dict[str, str]:
    """Launch vLLM servers for each model checkpoint.

    Returns:
        addr_map: dict mapping model_name -> "host:port"
    """
    vllm_job_ids = {}

    for name in model_names:
        # Extract revision from model name (e.g. "amber-ckpt_001" -> "ckpt_001")
        revision = name.split("-", 1)[1] if "-" in name else None

        # Clean up stale files
        model_tmp = os.path.join(TMP_DIR, name)
        os.makedirs(model_tmp, exist_ok=True)
        for f in ("host_port.txt", "ready.flag", "vllm_jobid.txt"):
            path = os.path.join(model_tmp, f)
            if os.path.exists(path):
                os.remove(path)

        # Build sbatch command
        vllm_script = os.path.join(BBOX_ROOT, "scripts/data_generation/vllm_host.sh")
        cmd = ["sbatch", "--parsable", vllm_script,
               f"--model_id={model_id}", f"--model_name={name}"]
        if revision:
            cmd.append(f"--revision={revision}")

        if use_slurm:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=BBOX_ROOT)
            job_id = result.stdout.strip()
            if not job_id:
                print(f"ERROR launching vLLM for {name}: {result.stderr}", flush=True)
                raise RuntimeError(f"Failed to launch vLLM for {name}")
            vllm_job_ids[name] = job_id
            # Save job ID for cleanup
            with open(os.path.join(model_tmp, "vllm_jobid.txt"), "w") as f:
                f.write(job_id)
            print(f"  [{name}] SLURM job {job_id}", flush=True)
        else:
            raise NotImplementedError("Direct launch not supported — use SLURM")

    # Register cleanup handler
    def cleanup_vllm():
        print("\nCleaning up vLLM servers...", flush=True)
        for name, job_id in vllm_job_ids.items():
            subprocess.run(["scancel", job_id], capture_output=True)
            model_tmp = os.path.join(TMP_DIR, name)
            for f in ("host_port.txt", "ready.flag", "vllm_jobid.txt"):
                path = os.path.join(model_tmp, f)
                if os.path.exists(path):
                    os.remove(path)
            print(f"  [{name}] cancelled job {job_id}", flush=True)

    atexit.register(cleanup_vllm)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Wait for all servers to be ready
    print(f"\nWaiting for {len(model_names)} vLLM servers to be ready...", flush=True)
    addr_map = {}
    timeout_secs = 1200  # 20 minutes
    start_time = time.time()

    while len(addr_map) < len(model_names):
        elapsed = time.time() - start_time
        if elapsed > timeout_secs:
            missing = [n for n in model_names if n not in addr_map]
            raise TimeoutError(
                f"vLLM servers not ready after {timeout_secs}s. Missing: {missing}"
            )

        for name in model_names:
            if name in addr_map:
                continue
            ready_flag = os.path.join(TMP_DIR, name, "ready.flag")
            host_port_file = os.path.join(TMP_DIR, name, "host_port.txt")
            if os.path.exists(ready_flag) and os.path.exists(host_port_file):
                with open(host_port_file) as f:
                    addr = f.read().strip()
                addr_map[name] = addr
                print(f"  [{name}] ready at {addr} ({int(elapsed)}s)", flush=True)

        if len(addr_map) < len(model_names):
            # Print progress every 30s
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                ready = len(addr_map)
                total = len(model_names)
                print(f"  {ready}/{total} servers ready ({int(elapsed)}s elapsed)...", flush=True)
            time.sleep(5)

    print(f"All {len(addr_map)} vLLM servers ready.\n", flush=True)
    return addr_map


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def compute_normalization_constants(training_cache_dir: str, input_dim: int = 768):
    """Compute normalization constants from the cached training memmap."""
    with open(os.path.join(training_cache_dir, "cached_data_info.json")) as f:
        info = json.load(f)
    data = np.memmap(
        os.path.join(training_cache_dir, "preprocessed_data.dat"),
        dtype=DTYPES[info["dtype"]],
        mode='r',
        shape=tuple(info["shape"]),
    )
    chunk_size = 50000
    n = data.shape[0]
    total_norms, emb_norms, prob_norms = [], [], []
    for start in range(0, n, chunk_size):
        chunk = data[start:start + chunk_size].astype(np.float64)
        total_norms.append(np.linalg.norm(chunk, axis=1))
        emb_norms.append(np.linalg.norm(chunk[:, :input_dim], axis=1))
        prob_norms.append(np.linalg.norm(chunk[:, input_dim:], axis=1))
    constants = {
        "mean_total_norm": float(np.mean(np.concatenate(total_norms))),
        "mean_embedding_norm": float(np.mean(np.concatenate(emb_norms))),
        "mean_prob_norm": float(np.mean(np.concatenate(prob_norms))),
    }
    del data
    return constants


def preprocess_batch(
    embeddings: np.ndarray,
    logprobs: np.ndarray,
    ofw: float,
    norm_constants: dict,
    input_dim: int = 768,
) -> np.ndarray:
    """Preprocess a batch of embeddings + logprobs into SAE input format."""
    probs = np.exp(logprobs.astype(np.float64)).astype(np.float16)
    data = np.concatenate([embeddings, probs], axis=1).astype(np.float64)
    if 0 < ofw < 1:
        mt = norm_constants["mean_total_norm"]
        me = norm_constants["mean_embedding_norm"]
        mp = norm_constants["mean_prob_norm"]
        data[:, :input_dim] *= (1 - ofw) * mt / me
        data[:, input_dim:] *= ofw * mt / mp
    return data.astype(np.float16)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_longformer_embeddings(texts, sample_ids, tokenizer, model, device):
    """Get Longformer word embeddings in-memory."""
    return get_longformer_word_features(
        device, tokenizer, model,
        input_text=texts,
        sample_ids=sample_ids,
        use_sliding_window=True,
    )


def get_vllm_logprobs_batch(
    texts, sample_ids, model_addrs, model_id,
    amber_tokenizer, longformer_tokenizer,
    async_limiter=100, stride=900,
):
    """Get logprobs from all vLLM models for a batch of texts."""
    all_model_logprobs = {}
    for model_name in model_addrs:
        overlapping_strings, window_start_idx, window_to_sample_mapping, window_word_ids = \
            get_output_overlapping_strings(
                text=texts,
                sample_ids=sample_ids,
                tokenizer=amber_tokenizer,
                embedding_model_tokenizer=longformer_tokenizer,
                max_length=2046,
                stride=stride,
            )
        word_logprobs = get_model_logprobs(
            model_addr=model_addrs[model_name],
            model_id=model_id,
            model_name=model_name,
            input_text=overlapping_strings,
            sample_ids=sample_ids,
            window_word_ids=window_word_ids,
            window_start_idx=window_start_idx,
            window_to_sample_mapping=window_to_sample_mapping,
            async_limiter=async_limiter,
        )
        all_model_logprobs[model_name] = word_logprobs
    return all_model_logprobs


@torch.no_grad()
def run_sae_inference(data: np.ndarray, encoder, device: str) -> np.ndarray:
    """Run SAE forward pass and return feature activations."""
    batch_size = 512
    all_acts = []
    for i in range(0, data.shape[0], batch_size):
        batch = torch.tensor(data[i:i + batch_size], dtype=torch.float32).to(device)
        _, acts, _, _ = encoder(batch, return_acts=True, return_l2_error_per_sample=True)
        all_acts.append(acts.cpu().numpy())
    return np.concatenate(all_acts, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--data", type=click.Path(exists=True), required=True,
              help="JSONL file with documents")
@click.option("--sae_dir", type=click.Path(exists=True), required=True,
              help="Directory containing trained SAE (sae.pt + config.json)")
@click.option("--model_addrs", type=str, default=None,
              help="Comma-separated model_name=host:port pairs (skip server launch)")
@click.option("--model_id", type=str, default="LLM360/Amber",
              help="HuggingFace model ID")
@click.option("--training_cache_dir", type=click.Path(exists=True), required=True,
              help="Path to cached training data dir (for normalization constants)")
@click.option("--output", type=click.Path(), required=True,
              help="Output parquet path for feature activations")
@click.option("--batch_size", type=int, default=10,
              help="Number of documents per batch")
@click.option("--async_limiter", type=int, default=100)
@click.option("--norm_cache", type=click.Path(), default=None,
              help="Path to cache normalization constants JSON")
def main(
    data, sae_dir, model_addrs, model_id, training_cache_dir,
    output, batch_size, async_limiter, norm_cache,
):
    # Load SAE config to find required model checkpoints
    print("Loading SAE...", flush=True)
    encoder, cfg = load_sae(sae_dir)
    encoder.eval()
    device = cfg["device"]
    ofw = cfg["output_feature_weight"]
    model_names = cfg["model_names"]
    print(f"SAE requires {len(model_names)} models: {model_names}")

    # Either parse provided addresses or launch servers
    if model_addrs:
        print("Using provided vLLM addresses.", flush=True)
        addr_map = {}
        for pair in model_addrs.split(","):
            name, addr = pair.split("=", 1)
            addr_map[name] = addr
    else:
        print("Launching vLLM servers via SLURM...", flush=True)
        addr_map = launch_vllm_servers(model_names, model_id, use_slurm=True)

    # Verify all required models have addresses
    for m in model_names:
        if m not in addr_map:
            raise ValueError(f"Missing vLLM address for model {m}. Got: {list(addr_map.keys())}")

    # Normalization constants
    if norm_cache and os.path.exists(norm_cache):
        print(f"Loading cached normalization constants from {norm_cache}", flush=True)
        with open(norm_cache) as f:
            norm_constants = json.load(f)
    else:
        print("Computing normalization constants from training data...", flush=True)
        norm_constants = compute_normalization_constants(training_cache_dir)
        if norm_cache:
            with open(norm_cache, "w") as f:
                json.dump(norm_constants, f, indent=2)
            print(f"Cached to {norm_cache}")
    print(f"Norms: {norm_constants}")

    # Load Longformer + tokenizers
    print("Loading Longformer...", flush=True)
    lf_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", torch_dtype=torch.float16)
    lf_model = LongformerModel.from_pretrained("allenai/longformer-base-4096", torch_dtype=torch.float16)
    lf_model.to(device).eval()

    amber_tokenizer = AutoTokenizer.from_pretrained(model_id)
    amber_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load data
    print("Loading documents...", flush=True)
    texts, sample_ids, domains = filter_and_sort_data(data, is_sorted=False)
    num_batches = math.ceil(len(texts) / batch_size)
    print(f"{len(texts)} documents, {num_batches} batches")

    all_results = []

    for batch_idx in tqdm(range(num_batches), desc="Processing"):
        batch_texts, batch_ids, _ = get_batch_data(texts, sample_ids, domains, batch_size, batch_idx)
        if not batch_texts:
            continue

        # 1. Longformer embeddings
        input_features, words, num_words = get_longformer_embeddings(
            batch_texts, batch_ids, lf_tokenizer, lf_model, device,
        )

        # 2. vLLM logprobs for all model checkpoints
        model_logprobs = get_vllm_logprobs_batch(
            batch_texts, batch_ids, addr_map, model_id,
            amber_tokenizer, lf_tokenizer, async_limiter,
        )

        # 3. Align and build per-word vectors
        for doc_idx, doc_id in enumerate(input_features.keys()):
            n_words = num_words[doc_idx]
            emb = np.stack(input_features[doc_id][:n_words], axis=0)

            lp_cols = []
            skip_doc = False
            for m in model_names:
                if doc_id not in model_logprobs[m]:
                    skip_doc = True
                    break
                lp_cols.append(model_logprobs[m][doc_id][:n_words])
            if skip_doc:
                continue
            logprobs_arr = np.stack(lp_cols, axis=1)

            # Clip FP16 underflow
            min_neg_fp16 = -6.10352e-05
            logprobs_arr = np.where(
                (logprobs_arr < 0) & (logprobs_arr > min_neg_fp16),
                min_neg_fp16, logprobs_arr,
            )

            # 4. Preprocess + 5. SAE inference
            sae_input = preprocess_batch(emb, logprobs_arr, ofw, norm_constants)
            activations = run_sae_inference(sae_input, encoder, device)

            # 6. Collect sparse results
            for word_idx in range(n_words):
                word_acts = activations[word_idx]
                active_mask = word_acts > 0
                if not active_mask.any():
                    continue
                active_features = np.where(active_mask)[0]
                active_values = word_acts[active_mask]
                all_results.append({
                    "word_id": f"{doc_id}_{word_idx}",
                    "doc_id": doc_id,
                    "word": words[doc_idx][word_idx] if word_idx < len(words[doc_idx]) else "",
                    "active_features": active_features.tolist(),
                    "activation_values": active_values.tolist(),
                })

        del input_features, model_logprobs
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    print(f"Saving {len(all_results)} activated words to {output}", flush=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_parquet(output, engine="pyarrow")
    print("Done.")


if __name__ == "__main__":
    main()
