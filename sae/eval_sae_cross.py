import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
from create_cached_data import cache_data_per_dir
from dask import config as dask_cfg
from dask.distributed import Client
from npy_append_array import NpyAppendArray
from tqdm import tqdm
import json
import numpy as np
import os
import pandas as pd
import pickle
from data_utils import copy_temp_memmap, cleanup_temp_memmap
from sae_utils import (
    DTYPES,
    DataLoader,
    BackgroundDataLoader,
    get_sae_name,
    get_encoder,
    load_sae,
    get_config,
    calc_feature_hist_and_densities,
    get_words_in_context
)
import math

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
    topk_indices = torch.full((topk, cfg["dict_size"]), -1).to(cfg["device"])
    topk_error = torch.full((topk, cfg["dict_size"]), np.inf).to(cfg["device"])
    
    mse = 0
    num_samples = 0
    activated = torch.zeros(cfg["dict_size"]).to(cfg["device"])
    
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


def get_topk_words_in_context(
    sae_dir: str,
    k_activations: str,
    data_dir: str
):
    topk_filename = f"top-{k_activations}_activations.csv"
    topk_file = os.path.join(sae_dir, topk_filename)
    topk_df = pd.read_csv(topk_file)
    # drop activations that are 0
    topk_df = topk_df[topk_df["act_value"] != 0]
    word_ids = topk_df["word_id"].unique()
    
    
    input_feature_dirs = [os.path.join(data_dir, "input_features")]
    words_in_context = {}
    for input_feature_dir in input_feature_dirs:
        words_in_context.update(get_words_in_context(input_feature_dir, word_ids))
    topk = topk_filename.split("_")[0]
    output_file = os.path.join(sae_dir, f"{topk}_words_in_context.json")
    with open(output_file, "w") as f:
        wic = json.dumps(words_in_context, indent=4)
        f.write(wic)
    return



def calc_feature_metrics(sae_dir: str, data_dir:str, k: int = 50, model_string: str = None, ):
    def get_embeddings_from_word_ids(
        input_feature_dir: str,
        word_ids: list[str],
    ) -> dict[str, np.ndarray]:
        file_df = pd.read_csv(f"{input_feature_dir}/file_to_doc.csv")
        file_df.drop(columns=["num_words"], inplace=True)
        all_doc_word_ids = {}
        for word_id in word_ids:
            doc_id = "_".join(word_id.split("_")[:-1])
            if doc_id in all_doc_word_ids:
                all_doc_word_ids[doc_id].append(word_id)
            else:
                all_doc_word_ids[doc_id] = [word_id]
        doc_ids = list(all_doc_word_ids.keys())
        file_df = file_df[file_df["doc_id"].isin(doc_ids)]
        orig_embeddings = {}
        for file in tqdm(file_df["file"].to_list()):
            docs = file_df[file_df["file"] == file]["doc_id"].to_list()
            file = os.path.join(input_feature_dir, file)
            file_word_ids = []
            for doc in docs:
                file_word_ids += all_doc_word_ids[doc]
            embedding_cols = [f"embedding_{i}" for i in range(768)]
            cols = ["word_id"] + embedding_cols
            words_df = pd.read_parquet(file, columns=cols, filters=[("word_id", 'in', file_word_ids)])
            ordered_word_ids = words_df["word_id"].tolist()
            embeddings_df = words_df[embedding_cols]
            ordered_embeddings = embeddings_df.to_numpy()
            for i, word_id in enumerate(ordered_word_ids):
                orig_embeddings[word_id] = ordered_embeddings[i]
        return orig_embeddings
    
    
    
    input_feature_dirs = [os.path.join(data_dir, "input_features")]
    
    topk_filename = f"top-{k}_activations.csv"
    topk_file = os.path.join(sae_dir, topk_filename)
    topk_df = pd.read_csv(topk_file)
    # filter out dead features
    topk_df = topk_df[topk_df["act_value"] != 0]
    word_ids = topk_df["word_id"].unique()

    
    #TODO: add support for > 2 models
    # assert len(model_names) == 2, "Currently only supports comparison between 2 models"
    # model_names_label = "_".join(model_names)
    # model_names_label = "n_moreearly_models"
    model_names_label = model_string

    if not os.path.exists(os.path.join(sae_dir, "topk_feature_word_embeddings.pkl")):
        word_id_embeddings = {}
        for input_feature_dir in input_feature_dirs:
            word_id_embeddings.update(get_embeddings_from_word_ids(input_feature_dir, word_ids))
        with open(os.path.join(sae_dir, "topk_feature_word_embeddings.pkl"), "wb") as f:
            pickle.dump(word_id_embeddings, f)
    else:
        with open(os.path.join(sae_dir, "topk_feature_word_embeddings.pkl"), "rb") as f:
            word_id_embeddings = pickle.load(f)

    embedding_avg_dist = []
    embedding_avg_cos_sim = []
    prob_avg_dist = []
    
    prob_means = []
    prob_medians = []
    prob_variances = []
    logprob_means = []
    logprob_medians = []
    logprob_variances = []

    num_samples = []
    model_prob_variances = {model_name: [] for model_name in model_names}
    
    prob_avg_ranks_list = []
    logprob_avg_ranks_list = []

    prob_median_ranks_list = []
    logprob_median_ranks_list = []
    
    feature_indices = []
    sample_centroid_embedding_dist = []
    sample_centroid_cos_sim = []
    
    for feature in tqdm(topk_df["feature"].unique()):
        feature_df = topk_df[topk_df["feature"] == feature]
        acts = feature_df["act_value"].values
        feature_embeddings = np.array([word_id_embeddings[word_id] for word_id in feature_df["word_id"].values])    # 50 x 768
        # max act value should be at top of vector since topk sorts
        max_act = acts[0]
        # keep activation and associated sample if
        # activation value is in the top 3 quartiles or >= 0.25 * max_act
        bottom_quartile = np.percentile(acts, 25, method="nearest")
        max_act_threshold = 0.25 * max_act
        sample_indices = np.nonzero(((acts > bottom_quartile) | (acts > max_act_threshold)))
        feature_embeddings = feature_embeddings[sample_indices] # num_samples x 768
        feature_logprobs = []
        for model_name in model_names:
            feature_logprobs.append(feature_df[model_name].values)
        feature_logprobs = np.array(feature_logprobs).T # 50 x n
        feature_logprobs_mean = np.mean(feature_logprobs, axis=0)   # n
        feature_logprobs_median = np.median(feature_logprobs, axis=0)   # n
        feature_logprobs_variance = np.var(feature_logprobs, axis=0)   # n
        feature_probs = np.exp(feature_logprobs)
        feature_probs = feature_probs[sample_indices] # num_samples x n
        feature_embeddings_mean = np.mean(feature_embeddings, axis=0)   # 768
        feature_probs_mean = np.mean(feature_probs, axis=0)   # n
        feature_probs_median = np.median(feature_probs, axis=0)   # n
        feature_probs_variance = np.var(feature_probs, axis=0)   # n
        embedding_dist = np.linalg.norm(feature_embeddings - feature_embeddings_mean, axis=1)
        embedding_cos_sim = np.dot(feature_embeddings, feature_embeddings_mean) / (np.linalg.norm(feature_embeddings, axis=1) * np.linalg.norm(feature_embeddings_mean))
        prob_dist = np.linalg.norm(feature_probs - feature_probs_mean, axis=1)

        

        prob_ranks_all = np.argsort(np.argsort(-feature_probs, axis=1), axis=1) + 1 # shape: (n_samples, n_models)
        logprob_ranks_all = np.argsort(np.argsort(-feature_logprobs, axis=1), axis=1)  + 1# shape: (n_samples, n_models)
        

        feature_prob_ranks_avg = np.mean(prob_ranks_all, axis=0).tolist()  # shape: (n_models,)
        feature_logprob_ranks_avg = np.mean(logprob_ranks_all, axis=0).tolist()  # shape: (n_models,)

        feature_prob_ranks_median = np.median(prob_ranks_all, axis=0).tolist()  # shape: (n_models,)
        feature_logprob_ranks_median = np.median(logprob_ranks_all, axis=0).tolist()  # shape: (n_models,)
        

        prob_avg_ranks_list.append(feature_prob_ranks_avg)
        logprob_avg_ranks_list.append(feature_logprob_ranks_avg)

        prob_median_ranks_list.append(feature_prob_ranks_median)
        logprob_median_ranks_list.append(feature_logprob_ranks_median)
        
        feature_indices.append([feature] * feature_probs.shape[0])
        sample_centroid_embedding_dist.append(embedding_dist)
        sample_centroid_cos_sim.append(embedding_cos_sim)

        embedding_avg_dist.append(np.mean(embedding_dist))
        embedding_avg_cos_sim.append(np.mean(embedding_cos_sim))
        prob_avg_dist.append(np.mean(prob_dist))

        prob_means.append(feature_probs_mean)
        prob_medians.append(feature_probs_median)
        prob_variances.append(feature_probs_variance)

        logprob_means.append(feature_logprobs_mean)
        logprob_medians.append(feature_logprobs_median)
        logprob_variances.append(feature_logprobs_variance)
        
        for i, model_name in enumerate(model_names):
            model_prob_variances[model_name].append(np.var(feature_probs[:, i]))
        num_samples.append(sample_indices[0].shape[0])
    distance_df = pd.DataFrame({
        "feature": topk_df["feature"].unique(),
        "num_samples_considered": num_samples,
        "embedding_avg_dist": embedding_avg_dist,
        "embedding_avg_cos_sim": embedding_avg_cos_sim,
        "prob_avg_dist": prob_avg_dist,
        "prob_avg_ranks": prob_avg_ranks_list,
        "logprob_avg_ranks": logprob_avg_ranks_list,
        "prob_median_ranks": prob_median_ranks_list,
        "logprob_median_ranks": logprob_median_ranks_list,
        "prob_means": prob_means,
        "prob_medians": prob_medians,
        "prob_variances": prob_variances,
        "logprob_means": logprob_means,
        "logprob_medians": logprob_medians,
        "logprob_variances": logprob_variances
    })
    for model_name in model_names:
        distance_df[f"{model_name}_prob_variance"] = model_prob_variances[model_name]
    distance_df.to_csv(os.path.join(sae_dir, f"feature_metrics-{model_names_label}.csv"), index=False)
    print(f"Saved feature metrics to {os.path.join(sae_dir, f'feature_metrics-{model_names_label}.csv')}")
    
    feature_indices = np.concatenate(feature_indices)
    sample_centroid_embedding_dist = np.concatenate(sample_centroid_embedding_dist)
    sample_centroid_cos_sim = np.concatenate(sample_centroid_cos_sim)
    print(len(feature_indices), len(sample_centroid_embedding_dist), len(sample_centroid_cos_sim))
    feature_sample_centroid_df = pd.DataFrame({
        "feature": feature_indices,
        "sample_centroid_embedding_dist": sample_centroid_embedding_dist,
        "sample_centroid_cos_sim": sample_centroid_cos_sim
    })
    feature_sample_centroid_df.to_csv(os.path.join(sae_dir, f"feature_sample_centroid-metrics.csv"), index=False)
    

# Example usage
if __name__ == "__main__":
    # Configuration
    k = 50
    base_sae_dir = "/home/nsrikant/bbox_outputs/sae_outputs_larger_pile/n_moreearly_larger_pile_seed=42_ofw=0.7_N=3000_k=50_lp=None"
    eval_data_dir = "/home/nsrikant/bbox_outputs/output/blimp_full"
    temp_dir="/scratch/nsrikant/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    model_string = "blimp_trained_on_pile"
    output_dir = f"/home/nsrikant/bbox_outputs/sae_outputs_{model_string}"
    
    os.makedirs(output_dir, exist_ok=True)
    # data_dir = "/home/nsrikant/bbox_outputs/output/blimp_full"
    cached_data_dir = ["/home/nsrikant/.cache/n_moreearly_blimp_full/blimp_full"]
    
    

    encoder, base_config = load_sae(base_sae_dir)
    model_names = base_config["model_names"]
    
    all_data = []
    all_word_ids = []
    all_logprobs = {}
    all_zscores = {}
    all_tempfiles = []
    for data_dir in cached_data_dir:
        preprocessed_data_dir = os.path.join(data_dir, f"ofw={base_config['output_feature_weight']}")
        cached_data_filepath = os.path.join(preprocessed_data_dir, "preprocessed_data.dat")
        with open(f"{preprocessed_data_dir}/cached_data_info.json", "r") as f:
            cached_data_info = json.load(f)
        cached_data_dir_name = os.path.join(os.path.basename(data_dir), f"ofw={base_config['output_feature_weight']}")
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
        for model in model_names:
            with open(f"{data_dir}/{model}/logprobs.pkl", "rb") as f:
                logprobs[model] = pickle.load(f)
            with open(f"{data_dir}/{model}/zscores.pkl", "rb") as f:
                zscores[model] = pickle.load(f)
    
        
        all_data.append(data)
        all_word_ids += word_ids.tolist()
        for model in model_names:
            if model not in all_logprobs:
                all_logprobs[model] = [logprobs[model]]
                all_zscores[model] = [zscores[model]]
            else:
                all_logprobs[model].append(logprobs[model])
                all_zscores[model].append(zscores[model])
    for model in model_names:
        all_logprobs[model] = np.concatenate(all_logprobs[model], axis=0)
        all_zscores[model] = np.concatenate(all_zscores[model], axis=0)

    eval_metrics, topk_dict = get_eval_metrics_and_topk_feature_acts(
        all_data, encoder, base_config, output_dir, topk=50, save_activations=True,
    )

    print(f"Saving eval metrics to {output_dir}/eval_metrics.json", flush=True)
    with open(f"{output_dir}/eval_metrics.json", "w") as f:
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
    for model in model_names:
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
    
    print(f"Saving top {k} activations to {output_dir}/top-{k}_activations.csv", flush=True)
    df.to_csv(f"{output_dir}/top-{k}_activations.csv", index=False)
    print("Getting top k words in context", flush=True)
    get_topk_words_in_context(output_dir, k, eval_data_dir)
    
    print("Calculating feature metrics", flush=True)
    calc_feature_metrics(output_dir, eval_data_dir, model_names, model_string=model_string)
    
    
    print(f"Calculating feature histograms and densities", flush=True)
    calc_feature_hist_and_densities(output_dir)
    
    print("eval complete, cleanup...", flush=True)
    for tempfile_path in all_tempfiles:
        print(f"removing {tempfile_path}...", flush=True)
        cleanup_temp_memmap(data, tempfile_path)


    

    




    
    

    