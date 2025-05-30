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
SEED = 0

@torch.no_grad()
def get_eval_metrics_and_topk_feature_acts(
    all_data: list[np.memmap],
    encoder,
    cfg,
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
    topk_indices = torch.full((topk, cfg["dict_size"]), -1).to(cfg["device"])
    topk_error = torch.full((topk, cfg["dict_size"]), np.inf).to(cfg["device"])
    
    mse = 0
    num_samples = 0
    activated = torch.zeros(cfg["dict_size"]).to(cfg["device"])
    
    prev_data_size = 0
    num_random_indices = math.floor(512 * random_acts_percent)
    if save_activations:
        save_file = cfg["save_dir"] + f"/feature_activations.npy"
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
                    feature_samples_error = torch.tile(error_per_sample, (1, cfg["dict_size"]))
                    npaa.append(batch_acts[random_indices].cpu().numpy())
                    acts = torch.cat([topk_acts, batch_acts], 0)   # 562 x dict_size
                    error = torch.cat([topk_error, feature_samples_error], 0) # 562
                    start_idx = (i * batch_size) + prev_data_size
                    end_idx = start_idx + batch.shape[0]
                    batch_indices = torch.tile(
                        torch.arange(start_idx, end_idx).reshape(-1, 1), (1, cfg["dict_size"])
                    ).to(cfg["device"])
                    indices = torch.cat([topk_indices, batch_indices], 0) 
                    topk_acts, sorted_indices = torch.topk(acts, topk, dim=0)
                    topk_error = torch.gather(error, 0, sorted_indices)
                    topk_indices = torch.gather(indices, 0, sorted_indices)
        else:
            for i, batch in tqdm(enumerate(background_loader)):
                batch = batch.to(cfg["device"])
                _, batch_acts, _, error_per_sample = encoder(batch, return_acts=True, return_l2_error_per_sample=True)
                mse = mse * (num_samples / (num_samples + batch.shape[0])) + torch.sum(error_per_sample).item() / (num_samples + batch.shape[0])
                num_samples += batch.shape[0]
                activated += (batch_acts > 0).sum(0)
                feature_samples_error = torch.tile(error_per_sample, (1, cfg["dict_size"]))
                acts = torch.cat([topk_acts, batch_acts], 0)   # 562 x dict_size
                error = torch.cat([topk_error, feature_samples_error], 0)
                start_idx = (i * batch_size) + prev_data_size
                end_idx = start_idx + batch.shape[0]
                batch_indices = torch.tile(
                    torch.arange(start_idx, end_idx).reshape(-1, 1), (1, cfg["dict_size"])
                ).to(cfg["device"])
                indices = torch.cat([topk_indices, batch_indices], 0) 
                topk_acts, sorted_indices = torch.topk(acts, topk, dim=0)
                topk_error = torch.gather(error, 0, sorted_indices)
                topk_indices = torch.gather(indices, 0, sorted_indices)
        prev_data_size += data.shape[0]
    percent_dead = (activated == 0).sum().item() / cfg["dict_size"]

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
    default="/data/tir/projects/tir1/users/lindiat/bbox/cache",
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
    "--sae_dir",
    help="Directory where SAE should be loaded from",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--save_activations",
    help="Whether to save all activations",
    type=bool,
    default=False,
)
@click.option(
    "--sae_name_prefix",
    help="Prefix of the name of the SAE model",
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
def main(
    args: str,
    cache_dir: str,
    config_path: str,
    eval_sample: float,
    k: int,
    random_sae: bool,
    sae_dir: str,
    sae_name_prefix: str,
    save_activations: bool,
    spill_dir: str,
    temp_dir: str,
    workers: int = 16,
):    
    np.random.seed(SEED)
    
    if args is not None:
        with open(args, "r") as f:
            args_dict = json.load(f)
        cache_dir = args_dict.get("cache_dir", cache_dir)
    
    if config_path:
        orig_cfg = get_config(config_path)
        if sae_dir is None:
            assert sae_name_prefix is not None, "sae_name_prefix must be provided if sae_dir is not"
            sae_model_name = get_sae_name(orig_cfg, sae_name_prefix)
            sae_dir = f"{args_dict['save_dir']}/{sae_model_name}"
    else:
        assert sae_dir is not None, "sae_dir must be provided if config_path is not"

    if random_sae:
        print("Loading random SAE...", flush=True)
        cfg = json.load(open(f"{sae_dir}/config.json", "r"))
        orig_sae_dir = cfg["save_dir"]
        sae_dir = f"{os.path.dirname(orig_sae_dir)}/random_{os.path.basename(orig_sae_dir)}"
        cfg["save_dir"] = sae_dir
        os.makedirs(sae_dir, exist_ok=True)
        with open(f"{sae_dir}/config.json", "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"Saving random SAE results to {sae_dir}", flush=True)
        encoder = get_encoder(cfg)
        torch.save(encoder.state_dict(), f"{sae_dir}/sae.pt")
    else:
        print(f"Loading SAE from {sae_dir}...", flush=True)
        encoder, cfg = load_sae(sae_dir)
    encoder.eval()

    pprint.pprint(cfg)
    print("Loading data...", flush=True)

    # process the cached data, one dataset subset at a time
    model_string = "_".join(cfg["model_names"])
    data_dir_names = [os.path.basename(d) for d in cfg["data_dirs"]]
    # should already be sorted in natural order
    cached_dataset_dirs = [f"{cache_dir}/{model_string}/{os.path.basename(d)}" for d in data_dir_names]
    data_dirs_to_cache = []
    for i, dir in enumerate(cached_dataset_dirs):
        if not os.path.exists(os.path.join(dir, f"ofw={cfg['output_feature_weight']}")):
            data_dirs_to_cache.append(cfg["data_dirs"][i]) 

    # if the data hasn't been preprocessed and cached, do that first
    if len(data_dirs_to_cache) > 0:
        dask_cfg.set({'distributed.scheduler.worker-ttl': None})
        client = Client(
            n_workers=workers, memory_limit='12GB', processes=True, timeout='30s', local_directory=spill_dir
        )
        print(client)
        for data_dir in data_dirs_to_cache:
            _ = cache_data_per_dir(
                client,
                cache_dir,
                data_dir,
                cfg["model_names"],
                cfg["output_feature_weight"],
            )

    all_data = []
    all_word_ids = []
    all_logprobs = {}
    all_zscores = {}
    all_tempfiles = []
    for data_dir in cached_dataset_dirs:
        preprocessed_data_dir = os.path.join(data_dir, f"ofw={cfg['output_feature_weight']}")
        cached_data_filepath = os.path.join(preprocessed_data_dir, "preprocessed_data.dat")
        with open(f"{preprocessed_data_dir}/cached_data_info.json", "r") as f:
            cached_data_info = json.load(f)
        cached_data_dir_name = os.path.join(os.path.basename(data_dir), f"ofw={cfg['output_feature_weight']}")
        tempfile_path = f"{temp_dir}/{cached_data_dir_name}/preprocessed_data.dat"
        if not os.path.exists(tempfile_path):
            os.makedirs(f"{temp_dir}/{cached_data_dir_name}", exist_ok=True)
            print(f"copying data to {tempfile_path}...", flush=True)
            copy_temp_memmap(cached_data_filepath, tempfile_path)
        all_tempfiles.append(tempfile_path)
        dtype = DTYPES[cached_data_info["dtype"]]
        shape = tuple(cached_data_info["shape"])
        data = np.memmap(tempfile_path, dtype=dtype, mode='r', shape=shape)
        print(f"getting word ids, logprobs, and zscores...", flush=True)
        with open(f"{data_dir}/word_ids.pkl", "rb") as f:
            word_ids = pickle.load(f)
        logprobs = {}
        zscores = {}
        for model in cfg["model_names"]:
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
            for model in cfg["model_names"]:
                logprobs[model] = logprobs[model][indices]
                zscores[model] = zscores[model][indices]

        all_data.append(data)
        all_word_ids += word_ids.tolist()
        for model in cfg["model_names"]:
            if model not in all_logprobs:
                all_logprobs[model] = [logprobs[model]]
                all_zscores[model] = [zscores[model]]
            else:
                all_logprobs[model].append(logprobs[model])
                all_zscores[model].append(zscores[model])
    
    for model in cfg["model_names"]:
        all_logprobs[model] = np.concatenate(all_logprobs[model], axis=0)
        all_zscores[model] = np.concatenate(all_zscores[model], axis=0)

    print(f"Getting eval metrics and top {k} activations...", flush=True)
    eval_metrics, topk_dict = get_eval_metrics_and_topk_feature_acts(
        all_data, encoder, cfg, topk=k, save_activations=save_activations,
    )
    print(f"Saving eval metrics to {sae_dir}/eval_metrics.json", flush=True)
    with open(f"{sae_dir}/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    topk_acts = topk_dict["topk_acts"].cpu().numpy()
    topk_indices = topk_dict["topk_indices"].cpu().numpy()
    topk_error = topk_dict["topk_error"].cpu().numpy()

    # iterate through all sae features, save as csv
    feature = []
    act_value = []
    sample_error = []
    word_id = []
    word_logprobs = {}
    word_zscores = {}
    for model in cfg["model_names"]:
        word_logprobs[model] = []
        word_zscores[model + "_zscore"] = []
    for i in tqdm(range(topk_acts.shape[1])):
        feature += [i] * topk_acts.shape[0]
        act_value += topk_acts[:, i].tolist()
        sample_error += topk_error[:, i].tolist()
        feature_indices = topk_indices[:, i].tolist()
        feature_word_ids = [all_word_ids[j] for j in feature_indices]
        word_id += feature_word_ids
        for model in logprobs.keys():
            word_logprobs[model] += all_logprobs[model][feature_indices].tolist()
            word_zscores[model + "_zscore"] += all_zscores[model][feature_indices].tolist()
    df = pd.DataFrame(
        {
            "feature": feature,
            "act_value": act_value,
            "sample_error": sample_error,
            "word_id": word_id,
        }
    )
    model_logprobs_df = pd.DataFrame.from_dict(word_logprobs)
    model_zscores_df = pd.DataFrame.from_dict(word_zscores)
    # calculate variance of z-scores per word across models
    model_zscores_df["var"] = model_zscores_df.var(axis=1)
    df = pd.concat([df, model_logprobs_df, model_zscores_df], axis=1)
    # drop activations equal to 0
    df = df[df["act_value"] > 0]
    print(f"Saving top {k} activations to {sae_dir}/top-{k}_activations.csv", flush=True)
    df.to_csv(f"{sae_dir}/top-{k}_activations.csv", index=False)
    
    print("Getting top k words in context", flush=True)
    get_topk_words_in_context(sae_dir, k)
    
    print("Calculating feature metrics", flush=True)
    calc_feature_metrics(sae_dir)
    
    if save_activations:
        print(f"Calculating feature histograms and densities", flush=True)
        calc_feature_hist_and_densities(sae_dir)
    
    print("eval complete, cleanup...", flush=True)
    for tempfile_path in all_tempfiles:
        print(f"removing {tempfile_path}...", flush=True)
        cleanup_temp_memmap(data, tempfile_path)

if __name__ == "__main__":
    main()