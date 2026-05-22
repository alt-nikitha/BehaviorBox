"""
Script to get activation values from SAE for each word in the dataset.
This script will create files for:
    - top k activations for each feature in the SAE,
    - feature metrics for each feature in the SAE,
    - top k words in context for each feature in the SAE,
    - [optional] (a random sampling of) activation values
    - [optional] feature histograms and densities for each feature in the SAE 
"""

import click
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import torch

from create_cached_data import cache_data_per_dir
from dask import config as dask_cfg
from dask.distributed import Client
from npy_append_array import NpyAppendArray
from tqdm import tqdm

from data_utils import copy_temp_memmap, cleanup_temp_memmap
from sae_utils import (
    DTYPES,
    DataLoader,
    BackgroundDataLoader,
    get_sae_name,
    get_encoder,
    load_sae,
    get_config,
    calc_feature_metrics,
    calc_feature_hist_and_densities,
    get_topk_words_in_context,
)

# seed to select the same subset of data for evaluation
SEED = 42

@torch.no_grad()
def get_eval_metrics_and_topk_feature_acts(
    all_data: list[np.memmap],
    encoder,
    cfg,
    save_dir,
    topk: int = 50,
    save_activations: bool = False,
    random_acts_percent: float = 0.01,
) -> tuple[dict, dict]:
    """
    Calculates metrics for SAE performance (MSE, percentage of dead features)
        and the top k activations for each feature in the SAE, along with their data indices and associated error.
    If save_activations is True, saves all activations to a file (feature_activations.npy).
    
    Args:
        all_data: list of np.memmap, each containing the preprocessed data for a dataset subset
        encoder: SAE model
        cfg: dict, configuration for the SAE model
        topk: int, number of top activations to save per feature
        save_activations: bool, whether to save all activations
        random_acts_percent: float, percentage of random activations to save
    
    Returns:
        eval_metrics: dict, metrics for SAE performance
        topk_dict: dict, containing the top k activations for each feature in the SAE
    """
    batch_size = 512
    topk_acts = torch.full((topk, cfg["dict_size"]), -np.inf).to(cfg["device"])
    topk_indices = torch.full((topk, cfg["dict_size"]), -1, dtype=torch.long).to(cfg["device"])

    mse = 0
    num_samples = 0
    activated = torch.zeros(cfg["dict_size"]).to(cfg["device"])
    # Collect all per-sample errors in a flat list; look up by index at the end
    all_errors = []

    prev_data_size = 0
    num_random_indices = math.floor(512 * random_acts_percent)
    if save_activations:
        save_file = save_dir + f"/feature_activations.npy"
        print(save_file, flush=True)
        if os.path.exists(save_file):
            os.remove(save_file)
    for data in all_data:
        dataloader = DataLoader(data, batch_size, indices=np.arange(data.shape[0]))
        background_loader = BackgroundDataLoader(dataloader)
        if save_activations:
            with NpyAppendArray(save_file) as npaa:
                for i, batch in tqdm(enumerate(background_loader)):
                    batch = batch.to(cfg["device"]) # 512 x input_dim
                    random_indices = np.random.choice(batch.shape[0], num_random_indices, replace=False)
                    _, batch_acts, _, error_per_sample = encoder(batch, return_acts=True, return_l2_error_per_sample=True)
                    mse = mse * (num_samples / (num_samples + batch.shape[0])) + torch.sum(error_per_sample).item() / (num_samples + batch.shape[0])
                    num_samples += batch.shape[0]
                    activated += (batch_acts > 0).sum(0)
                    all_errors.append(error_per_sample.squeeze(-1).cpu())
                    npaa.append(batch_acts[random_indices].cpu().numpy())
                    acts = torch.cat([topk_acts, batch_acts], 0)   # (topk+batch) x dict_size
                    start_idx = (i * batch_size) + prev_data_size
                    end_idx = start_idx + batch.shape[0]
                    batch_indices = torch.arange(start_idx, end_idx, device=cfg["device"]).unsqueeze(1).expand(-1, cfg["dict_size"])
                    indices = torch.cat([topk_indices, batch_indices], 0)
                    topk_acts, sorted_indices = torch.topk(acts, topk, dim=0)
                    topk_indices = torch.gather(indices, 0, sorted_indices)
        else:
            for i, batch in tqdm(enumerate(background_loader)):
                batch = batch.to(cfg["device"])
                _, batch_acts, _, error_per_sample = encoder(batch, return_acts=True, return_l2_error_per_sample=True)
                mse = mse * (num_samples / (num_samples + batch.shape[0])) + torch.sum(error_per_sample).item() / (num_samples + batch.shape[0])
                num_samples += batch.shape[0]
                activated += (batch_acts > 0).sum(0)
                all_errors.append(error_per_sample.squeeze(-1).cpu())
                acts = torch.cat([topk_acts, batch_acts], 0)   # (topk+batch) x dict_size
                start_idx = (i * batch_size) + prev_data_size
                end_idx = start_idx + batch.shape[0]
                batch_indices = torch.arange(start_idx, end_idx, device=cfg["device"]).unsqueeze(1).expand(-1, cfg["dict_size"])
                indices = torch.cat([topk_indices, batch_indices], 0)
                topk_acts, sorted_indices = torch.topk(acts, topk, dim=0)
                topk_indices = torch.gather(indices, 0, sorted_indices)
        prev_data_size += data.shape[0]
    percent_dead = (activated == 0).sum().item() / cfg["dict_size"]

    # Look up errors for the topk indices
    all_errors = torch.cat(all_errors, dim=0)  # (total_samples,)
    topk_error = all_errors[topk_indices.cpu()]  # (topk, dict_size)

    metrics_dict = {
        "num_samples": num_samples,
        "mse": mse,
        "percent_dead": percent_dead,
    }
    topk_dict = {
        "topk_acts": topk_acts,
        "topk_indices": topk_indices,
        "topk_error": topk_error
    }
    return metrics_dict, topk_dict


@click.command()
@click.option(
    "--args",
    help="JSON file containing arguments",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--cache_dir",
    help="Directory to cache data to or load cached data from",
    type=click.Path(exists=True),
    default="/home/nsrikant/.cache",
)
@click.option(
    "--config_path",
    help="Custom config for model",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--eval_sample",
    help="Proportion of data to evaluate on, should be between 0 and 1",
    type=float,
    default=None,
)
@click.option(
    "--k",
    help="Number of top examples per feature to save",
    type=int,
    default=50,
)
@click.option(
    "--random_sae",
    help="Whether to use a randomly initialized SAE model",
    type=bool,
    default=False,
)
@click.option(
    "--base_sae_dir",
    help="Directory where base SAE should be loaded from",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--eval_sae_dir",
    help="Directory where eval SAE should be stored",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--save_activations",
    help="Whether to save all activations",
    type=bool,
    default=False,
)
# @click.option(
#     "--sae_name_prefix",
#     help="Prefix of the name of the SAE model",
#     type=str,
# )
@click.option(
    "--model_string",
    help="String to identify the SAE experiment",
    type=str,
)
@click.option(
    "--spill_dir",
    help="Directory to save spilled data",
    type=click.Path(exists=True),
)
@click.option(
    "--temp_dir",
    help="Directory to save temporary data",
    type=click.Path(exists=True),
    default = None,
)
@click.option(
    "--workers",
    type=int,
    default=16,
)
@click.option(
    "--seed",
    help="Random seed",
    type=int,
    default=42,
)
@click.option(
    "--ofw",
    help="Output Feature Weight",
    type=float,
    default=0.7,
)
@click.option(
    "--use_delta_prob",
    help="Use consecutive prob deltas (p_i - p_{i+1}) instead of raw probs as output features",
    type=bool,
    default=False,
)
@click.option(
    "--use_delta_logprob",
    help="Use consecutive logprob deltas (log p_i - log p_{i+1}) as output features (mutually exclusive with --use_delta_prob)",
    type=bool,
    default=False,
)
@click.option(
    "--decoder_ortho_loss_weight",
    help="If > 0, decoder orthogonality penalty weight (used to locate the trained SAE checkpoint)",
    type=float,
    default=0.0,
)
@click.option(
    "--normalize_per_part",
    help="Z-score embedding/prob blocks independently (must match training run for SAE folder lookup)",
    type=bool,
    default=False,
)
@click.option(
    "--output_dim_loss_weight",
    help="Per-dim loss weight on prob block used at training time (must match for SAE folder lookup): "
         "'auto', a float, or unset",
    type=str,
    default=None,
)
@click.option(
    "--variance_filter_top_pct",
    help="Variance filter fraction used at training time (must match for SAE folder lookup). "
         "Eval still runs over the full dataset.",
    type=float,
    default=None,
)
@click.option(
    "--variance_filter_mode",
    help="Variance filter mode used at training time (must match for SAE folder lookup).",
    type=click.Choice(["linear", "log"]),
    default="linear",
)
@click.option(
    "--cluster_strat_k",
    help="Cluster-stratified sampling K used at training time (must match for SAE folder lookup).",
    type=int,
    default=0,
)
@click.option(
    "--cluster_strat_per_cluster",
    help="Per-cluster count used at training time (must match for SAE folder lookup).",
    type=int,
    default=2000,
)
@click.option(
    "--cluster_strat_mode",
    help="Cluster-stratified mode used at training time (must match for SAE folder lookup).",
    type=click.Choice(["linear", "log"]),
    default="linear",
)
@click.option(
    "--superimpose_loss_weight",
    help="Superimpose-loss weight used at training time (must match for SAE folder lookup).",
    type=float,
    default=0.0,
)
@click.option(
    "--self_eval_mode",
    help="Self Eval Mode",
    type=bool,
    default=True,
)
@click.option(
    "--train_data_dirs",
    help="train directories for probs",
    type=list,
    default=[],
)
@click.option(
    "--eval_data_dirs",
    help="data directories for probs",
    type=list,
    default=[],
)
def main(
    args: str,
    cache_dir: str,
    config_path: str,
    eval_sample: float,
    k: int,
    random_sae: bool,
    base_sae_dir: str,
    eval_sae_dir: str,
    model_string: str,
    save_activations: bool,
    spill_dir: str,
    temp_dir: str,
    workers: int = 16,
    seed: int = 42,
    ofw: float = 0.7,
    use_delta_prob: bool = False,
    use_delta_logprob: bool = False,
    decoder_ortho_loss_weight: float = 0.0,
    normalize_per_part: bool = False,
    output_dim_loss_weight: str = None,
    variance_filter_top_pct: float = None,
    variance_filter_mode: str = "linear",
    cluster_strat_k: int = 0,
    cluster_strat_per_cluster: int = 2000,
    cluster_strat_mode: str = "linear",
    superimpose_loss_weight: float = 0.0,
    self_eval_mode: bool = True,
    train_data_dirs: list = [],
    eval_data_dirs: list = []
):
    np.random.seed(seed)
    if args is not None:
        with open(args, "r") as f:
            args_dict = json.load(f)
        cache_dir = args_dict.get("cache_dir", cache_dir)
        ofw = args_dict.get("output_feature_weight", None)
        seed = args_dict.get("seed", seed)
        use_delta_prob = args_dict.get("use_delta_prob", use_delta_prob)
        use_delta_logprob = args_dict.get("use_delta_logprob", use_delta_logprob)
        decoder_ortho_loss_weight = args_dict.get("decoder_ortho_loss_weight", decoder_ortho_loss_weight)
        normalize_per_part = args_dict.get("normalize_per_part", normalize_per_part)
        output_dim_loss_weight = args_dict.get("output_dim_loss_weight", output_dim_loss_weight)
        variance_filter_top_pct = args_dict.get("variance_filter_top_pct", variance_filter_top_pct)
        variance_filter_mode = args_dict.get("variance_filter_mode", variance_filter_mode)
        cluster_strat_k = args_dict.get("cluster_strat_k", cluster_strat_k)
        cluster_strat_per_cluster = args_dict.get("cluster_strat_per_cluster", cluster_strat_per_cluster)
        cluster_strat_mode = args_dict.get("cluster_strat_mode", cluster_strat_mode)
        superimpose_loss_weight = args_dict.get("superimpose_loss_weight", superimpose_loss_weight)
        if use_delta_prob and use_delta_logprob:
            raise ValueError("use_delta_prob and use_delta_logprob are mutually exclusive")
        if config_path:
            orig_cfg = get_config(config_path)
            if base_sae_dir is None:
                sae_name_prefix = f"{model_string}_seed={seed}_ofw={ofw}"
                if use_delta_prob:
                    sae_name_prefix = f"{sae_name_prefix}_delta"
                elif use_delta_logprob:
                    sae_name_prefix = f"{sae_name_prefix}_logdelta"
                if decoder_ortho_loss_weight > 0:
                    sae_name_prefix = f"{sae_name_prefix}_ortho={decoder_ortho_loss_weight}"
                if variance_filter_top_pct is not None and variance_filter_top_pct < 1.0:
                    mode_tag = "" if variance_filter_mode == "linear" else f"-{variance_filter_mode}"
                    sae_name_prefix = f"{sae_name_prefix}_varfilt={variance_filter_top_pct}{mode_tag}"
                if cluster_strat_k and cluster_strat_k > 0:
                    cmode_tag = "" if cluster_strat_mode == "linear" else f"-{cluster_strat_mode}"
                    sae_name_prefix = (
                        f"{sae_name_prefix}_cstrat=k{cluster_strat_k}x{cluster_strat_per_cluster}{cmode_tag}"
                    )
                if superimpose_loss_weight and float(superimpose_loss_weight) > 0:
                    sae_name_prefix = f"{sae_name_prefix}_sup={superimpose_loss_weight}"
                # Match the suffixes get_sae_name appends for these flags
                orig_cfg["normalize_per_part"] = normalize_per_part
                if output_dim_loss_weight is None or (isinstance(output_dim_loss_weight, str) and output_dim_loss_weight.lower() in ("none", "")):
                    orig_cfg["output_dim_loss_weight"] = None
                elif isinstance(output_dim_loss_weight, str) and output_dim_loss_weight.lower() == "auto":
                    orig_cfg["output_dim_loss_weight"] = "auto"
                else:
                    orig_cfg["output_dim_loss_weight"] = float(output_dim_loss_weight)
                sae_model_name = get_sae_name(orig_cfg, sae_name_prefix)
                base_sae_dir = f"{args_dict['train_save_dir']}/{sae_model_name}"
        if not self_eval_mode:
            if eval_sae_dir is None:
                eval_sae_dir = f"{args_dict['eval_save_dir']}/{sae_model_name}"
            

        else:
            eval_sae_dir = base_sae_dir


        
    if not self_eval_mode:
        data_dirs = args_dict.get("eval_data_dirs", eval_data_dirs)
        
    else:
        data_dirs = args_dict.get("train_data_dirs", train_data_dirs)

    
    print(f"Loading SAE from {base_sae_dir}...", flush=True)
    encoder, base_cfg = load_sae(base_sae_dir)
    if not os.path.exists(os.path.join(eval_sae_dir, "config.json")):
        config = {
            "model_names": base_cfg["model_names"]
        }
        os.makedirs(eval_sae_dir, exist_ok=True)
        with open (os.path.join(eval_sae_dir, "config.json"), "w") as fp:
            json.dump(config, fp)


    encoder.eval()
    pprint.pprint(base_cfg)
    use_delta_prob = base_cfg.get("use_delta_prob", use_delta_prob)
    use_delta_logprob = base_cfg.get("use_delta_logprob", use_delta_logprob)
    normalize_per_part = base_cfg.get("normalize_per_part", normalize_per_part)
    cache_subdir = f"ofw={base_cfg['output_feature_weight']}"
    if use_delta_prob:
        cache_subdir = f"{cache_subdir}_delta"
    elif use_delta_logprob:
        cache_subdir = f"{cache_subdir}_logdelta"
    if normalize_per_part:
        cache_subdir = f"{cache_subdir}_znorm"
    print("Loading data...", flush=True)
    data_dir_names = [os.path.basename(d) for d in data_dirs]
    cached_dataset_dirs = [f"{cache_dir}/{model_string}/{os.path.basename(d)}" for d in data_dir_names]
    data_dirs_to_cache = []
    for i, dir in enumerate(cached_dataset_dirs):
        if not os.path.exists(os.path.join(dir, cache_subdir)):
            data_dirs_to_cache.append(data_dirs[i])
    if len(data_dirs_to_cache) > 0:
        dask_cfg.set({'distributed.scheduler.worker-ttl': None})
        client = Client(
            n_workers=workers, memory_limit='12GB', processes=True, timeout='30s', local_directory=spill_dir
        )
        
        for data_dir in data_dirs_to_cache:
            print(data_dir)
            _ = cache_data_per_dir(
                model_string,
                client,
                cache_dir,
                data_dir,
                base_cfg["model_names"],
                base_cfg["output_feature_weight"],
                use_delta_prob=use_delta_prob,
                use_delta_logprob=use_delta_logprob,
                normalize_per_part=normalize_per_part,
            )
    all_data = []
    all_word_ids = []
    all_logprobs = {}
    all_zscores = {}
    all_tempfiles = []
    for data_dir in cached_dataset_dirs:
        preprocessed_data_dir = os.path.join(data_dir, cache_subdir)
        cached_data_filepath = os.path.join(preprocessed_data_dir, "preprocessed_data.dat")
        # try:
            
        # except FileNotFoundError:
        with open(f"{preprocessed_data_dir}/cached_data_info.json", "r") as f:
            cached_data_info = json.load(f)
            
        cached_data_dir_name = os.path.join(os.path.basename(data_dir), cache_subdir)
        tempfile_path = f"{temp_dir}/{cached_data_dir_name}/preprocessed_data.dat"
        if not os.path.exists(tempfile_path):
            os.makedirs(f"{temp_dir}/{cached_data_dir_name}", exist_ok=True)
            print(f"copying data to {tempfile_path}...", flush=True)
            copy_temp_memmap(cached_data_filepath, tempfile_path)
        all_tempfiles.append(tempfile_path)
        dtype = DTYPES[cached_data_info["dtype"]]
        shape = tuple(cached_data_info["shape"])
        data = np.array(np.memmap(tempfile_path, dtype=dtype, mode='r', shape=shape))
        print(f"getting word ids, logprobs, and zscores...", flush=True)
        with open(f"{data_dir}/word_ids.pkl", "rb") as f:
            word_ids = pickle.load(f)
        logprobs = {}
        zscores = {}
        for model in base_cfg["model_names"]:
            with open(f"{data_dir}/{model}/logprobs.pkl", "rb") as f:
                logprobs[model] = pickle.load(f)
            with open(f"{data_dir}/{model}/zscores.pkl", "rb") as f:
                zscores[model] = pickle.load(f)
    
        if eval_sample is not None:
            assert 0 < eval_sample < 1, "eval_sample must be between 0 and 1"
            num_samples = int(data.shape[0] * eval_sample)
            indices = np.random.choice(data.shape[0], num_samples, replace=False)
            data = data[indices]
            word_ids = word_ids[indices]
            for model in base_cfg["model_names"]:
                logprobs[model] = logprobs[model][indices]
                zscores[model] = zscores[model][indices]

        all_data.append(data)
        all_word_ids += word_ids.tolist()
        for model in base_cfg["model_names"]:
            if model not in all_logprobs:
                all_logprobs[model] = [logprobs[model]]
                all_zscores[model] = [zscores[model]]
            else:
                all_logprobs[model].append(logprobs[model])
                all_zscores[model].append(zscores[model])
    
    for model in base_cfg["model_names"]:
        all_logprobs[model] = np.concatenate(all_logprobs[model], axis=0)
        all_zscores[model] = np.concatenate(all_zscores[model], axis=0)

    print(f"Getting eval metrics and top {k} activations...", flush=True)
    eval_metrics, topk_dict = get_eval_metrics_and_topk_feature_acts(
        all_data, encoder, base_cfg, eval_sae_dir, topk=k, save_activations=save_activations,
    )
    print(f"Saving eval metrics to {eval_sae_dir}/eval_metrics.json", flush=True)
    with open(f"{eval_sae_dir}/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    topk_acts = topk_dict["topk_acts"].cpu().numpy()
    topk_indices = topk_dict["topk_indices"].cpu().numpy()
    topk_error = topk_dict["topk_error"].cpu().numpy()

    # Build topk CSV using vectorized operations
    n_topk, n_features = topk_acts.shape
    # Feature column: [0,0,...,0, 1,1,...,1, ..., N-1,...]  (column-major flatten)
    feature_col = np.repeat(np.arange(n_features), n_topk)
    # Flatten in column-major order to match: for each feature, all topk rows
    act_col = topk_acts.T.ravel()
    error_col = topk_error.T.ravel()
    idx_col = topk_indices.T.ravel()
    all_word_ids_arr = np.array(all_word_ids)
    word_id_col = all_word_ids_arr[idx_col]

    df = pd.DataFrame({
        "feature": feature_col,
        "act_value": act_col,
        "sample_error": error_col,
        "word_id": word_id_col,
    })
    for model in base_cfg["model_names"]:
        df[model] = all_logprobs[model][idx_col]
        df[model + "_zscore"] = all_zscores[model][idx_col]
    # calculate variance of z-scores per word across models
    zscore_cols = [model + "_zscore" for model in base_cfg["model_names"]]
    df["var"] = df[zscore_cols].var(axis=1)
    # drop activations equal to 0
    df = df[df["act_value"] > 0]
    print(f"Saving top {k} activations to {eval_sae_dir}/top-{k}_activations.csv", flush=True)
    df.to_csv(f"{eval_sae_dir}/top-{k}_activations.csv", index=False)
    
    print("Getting top k words in context", flush=True)
    get_topk_words_in_context(eval_sae_dir, k, data_dirs)
    
    print("Calculating feature metrics", flush=True)
    calc_feature_metrics(eval_sae_dir, data_dirs)
    
    if save_activations:
        print(f"Calculating feature histograms and densities", flush=True)
        calc_feature_hist_and_densities(eval_sae_dir)
    
    print("eval complete, cleanup...", flush=True)
    for tempfile_path in all_tempfiles:
        print(f"removing {tempfile_path}...", flush=True)
        cleanup_temp_memmap(data, tempfile_path)


        
        



 
    
    




if __name__ == "__main__":
    main()