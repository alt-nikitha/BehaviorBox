# adapted from https://github.com/neelnanda-io/1L-Sparse-Autoencoder
import dask.array as da
import dask.dataframe as dd
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import queue
import seaborn as sns
import sys
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

sae_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if sae_root not in sys.path:
    sys.path.insert(0, sae_root)
    
from data_utils import get_words_in_context

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "np16": np.float16,
    "bf16": torch.bfloat16
}

logger = logging.getLogger(__name__)


def get_sae_name(cfg: dict, sae_name_prefix: str):
    if "dec_penalty_coeff" not in cfg:
        cfg["dec_penalty_coeff"] = None
    if cfg["topk"] is None:
        if sae_name_prefix:
            return f"{sae_name_prefix}_N={cfg['dict_size']}_l1={cfg['l1_coeff']}_lp={cfg['dec_penalty_coeff']}"
        else:
            return f"N={cfg['dict_size']}_l1={cfg['l1_coeff']}_lp={cfg['dec_penalty_coeff']}"
    else:
        if sae_name_prefix:
            return f"{sae_name_prefix}_N={cfg['dict_size']}_k={cfg['topk']}_lp={cfg['dec_penalty_coeff']}"
        else:
            return f"N={cfg['dict_size']}_k={cfg['topk']}_lp={cfg['dec_penalty_coeff']}"


def get_encoder(cfg, model=None):
    # initializing the appropriate encoder based on config
    if cfg["type"] == "AutoEncoder":
        encoder = AutoEncoder(cfg).to(cfg["device"])
    elif cfg["type"] == "TopKAutoEncoder":
        encoder = TopKAutoEncoder(cfg).to(cfg["device"])
    elif cfg["type"] == "BatchTopKAutoEncoder":
        encoder = BatchTopKAutoEncoder(cfg).to(cfg["device"])
    if model is not None:
        encoder.load_state_dict(model['model_state_dict'])
    return encoder


def load_checkpoint(cfg: dict, checkpoint_path: str):
    # loading a checkpoint to continue training
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_model = torch.load(checkpoint_path)
    encoder = get_encoder(cfg, checkpoint_model)
    encoder_optim = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"])
    )
    encoder_optim.load_state_dict(checkpoint_model['optimizer_state_dict'])
    checkpoint = {
        "step": checkpoint_model["step"],
        "global_step": checkpoint_model["global_step"],
        "encoder": encoder,
        "encoder_optim": encoder_optim,
    }
    return checkpoint


def load_sae(sae_dir: str):
    # loading a saved model (training completed, not a checkpoint)
    cfg = json.load(open(f"{sae_dir}/config.json", "r"))
    if cfg["type"] == "AutoEncoder":
        encoder = AutoEncoder.load(cfg, sae_dir)
    elif cfg["type"] == "TopKAutoEncoder":
        encoder = TopKAutoEncoder.load(cfg, sae_dir)
    elif cfg["type"] == "BatchTopKAutoEncoder":
        encoder = BatchTopKAutoEncoder.load(cfg, sae_dir)
    encoder = encoder.to(get_device())
    return encoder, cfg


class DataLoader:
    def __init__(
        self,
        data: np.memmap,
        batch_size: int,
        indices: list[int] = None,
    ):
        self.indices = indices
        self.length = len(indices)
        self.data = data
        self.batch_indices = [
            self.indices[x : x + batch_size] for x in range(0, self.length, batch_size)
        ]
        self.num_batches = len(self.batch_indices)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.num_batches:
            raise StopIteration
        batch_indices = self.batch_indices[self.index]
        samples = self.data[batch_indices]
        batch = torch.from_numpy(samples)
        self.index += 1
        return batch


class BackgroundDataLoader:
    def __init__(
        self,
        dataloader,
    ):
        self.dataloader = dataloader
        self.queue = queue.Queue(maxsize=2)  # Preload two batches
        self.stop_signal = False
        self.loader_thread = threading.Thread(target=self._load_batches)
        self.loader_thread.start()

    def _load_batches(self):
        while not self.stop_signal:
            try:
                batch = next(self.dataloader)
                self.queue.put(batch)
            except StopIteration:
                self.queue.put(None)
                self.stop_signal = True

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            self.stop()
            raise StopIteration
        return batch

    def stop(self):
        self.stop_signal = True
        self.loader_thread.join()


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using GPU")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def get_config(
    path: str = None,
    output_feature_weight: float = 0.0,
    seed: int = 0,
) -> dict:
    if path is not None:
        with open(path, "r") as f:
            cfg = json.load(f)
        cfg["device"] = get_device()
    else:
        cfg = {
            "batch_size": 128,
            "lr": 1e-4,
            "l1_coeff": 3e-4,
            "beta1": 0.9,
            "beta2": 0.99,
            "dict_size": 3000,
            "input_dim": 768,  # Longformer hidden size
            "enc_dtype": "fp32",
            "device": get_device(),
            "topk": None,
            "type": "AutoEncoder",
            "dec_penalty_coeff": 0.0,
        }
    cfg["output_feature_weight"] = output_feature_weight
    sae_type = cfg["type"]
    if sae_type == "TopKAutoEncoder" or sae_type == "BatchTopKAutoEncoder":
        cfg["l1_coeff"] = None
    cfg["seed"] = seed
    return cfg


def l2_loss_per_sample(x, x_reconstruct):
    return (x_reconstruct - x.float()).pow(2).sum(-1).reshape(-1, 1)


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"] if "l1_coeff" in cfg else None
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])

        self.enc = nn.Linear(
            in_features=cfg["input_dim"], out_features=d_hidden, dtype=dtype
        )
        self.enc.weight = nn.Parameter(torch.nn.init.kaiming_uniform_(self.enc.weight))
        self.enc.bias = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.dec = nn.Linear(
            in_features=d_hidden, out_features=cfg["input_dim"], dtype=dtype
        )
        self.dec.weight = nn.Parameter(torch.nn.init.kaiming_uniform_(self.dec.weight))
        self.dec.bias = nn.Parameter(torch.zeros(cfg["input_dim"], dtype=dtype))

        self.dec.weight.data = self.dec.weight / self.dec.weight.norm(
            dim=-1, keepdim=True
        )

        # add a penalty on the decoder output feature dims
        self.dec_penalty_coeff = cfg["dec_penalty_coeff"] if "dec_penalty_coeff" in cfg else None
        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.save_dir = cfg["save_dir"]

    def forward(self, x, return_acts=False, return_l2_error_per_sample=False):
        x_cent = x - self.dec.bias
        acts = F.relu(self.enc(x_cent))
        x_reconstruct = self.dec(acts)
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct)
        l2_loss = l2_error_per_sample.mean(0)
        l1_loss = self.l1_coeff * (acts.abs().sum().mean())
        loss = l2_loss + l1_loss
        if self.dec_penalty_coeff is not None:
            penalty = self.laplace_dec_penalty()
            loss += penalty
        if not return_acts:
            acts = None
        if not return_l2_error_per_sample:
            l2_error_per_sample = None
        return loss, acts, penalty, l2_error_per_sample

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.dec.weight / self.dec.weight.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.dec.weight.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.dec.weight.grad -= W_dec_grad_proj
        self.dec.weight.data = W_dec_normed

    def make_histogram(self, act_freqs: list[float], step: list[int]):
        hist_df = pd.DataFrame.from_dict({"act_freqs": act_freqs, "step": step})
        hist_df["log_act_freqs"] = hist_df["act_freqs"].apply(
            lambda x: min(max(1e-8, x), np.log10(x + 1e-7))
        )
        hist_df.to_csv(f"{self.save_dir}/act_freqs.csv", index=False)
        hist = sns.histplot(data=hist_df, x="log_act_freqs", stat="percent")
        hist.axvline(np.log10(1e-6), color='r', linestyle='dashed', linewidth=2)
        step_int = step[0]
        hist.set_title(f"Activation Frequencies at Step {step_int}")
        hist_fig = hist.get_figure()
        return hist_fig

    @classmethod
    def load(cls, cfg, save_dir: str):
        # loads a saved model (training completed, not a checkpoint)
        cfg["device"] = get_device()
        self = cls(cfg=cfg)
        model_weights = torch.load(f"{save_dir}/sae.pt", map_location=torch.device(get_device()))
        self.load_state_dict(model_weights)
        return self


# TopK AutoEncoder, as described in Gao et al. 2024
class TopKAutoEncoder(AutoEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.k = cfg["topk"]

    def forward(self, x, return_acts=False, return_l2_error_per_sample=False):
        x_cent = x - self.dec.bias
        post_enc = self.enc(x_cent)
        topk = torch.topk(post_enc, k=self.k, dim=-1)
        # use relu to ensure non-negative activations, but may not be necessary
        values = F.relu(topk.values)
        acts = torch.zeros_like(post_enc).scatter_(-1, topk.indices, values)
        x_reconstruct = self.dec(acts)
        penalty = torch.tensor(0.0)
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct)
        loss = l2_error_per_sample.mean(0)
        if self.dec_penalty_coeff is not None:
            penalty = self.laplace_dec_penalty()
            loss += penalty
        if not return_acts:
            acts = None
        if not return_l2_error_per_sample:
            l2_error_per_sample = None
        return loss, acts, penalty, l2_error_per_sample


# As described in Bussmann et al. 2024
# https://arxiv.org/abs/2412.06410
# also add a penalty on the decoder
# to ensure probs are being used to construct the higher dim representation
class BatchTopKAutoEncoder(TopKAutoEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x, return_acts=False, return_l2_error_per_sample=False):
        batch_k = self.k * x.shape[0]
        x_cent = x - self.dec.bias
        post_enc = self.enc(x_cent)
        # flatten the batch, apply batch-wise topk, then reshape back to batch
        post_enc_flat = post_enc.view(1, -1)
        batch_topk = torch.topk(post_enc_flat, k=batch_k, dim=-1)
        # use relu to ensure non-negative activations, but may not be necessary
        values = F.relu(batch_topk.values)
        acts = torch.zeros_like(post_enc_flat).scatter_(-1, batch_topk.indices, values)
        acts = acts.view(post_enc.size())
        x_reconstruct = self.dec(acts)
        penalty = torch.tensor(0.0)
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct)
        loss = l2_error_per_sample.mean(0)
        if self.dec_penalty_coeff is not None:
            penalty = self.laplace_dec_penalty()
            loss += penalty
        if not return_acts:
            acts = None
        if not return_l2_error_per_sample:
            l2_error_per_sample = None
        return loss, acts, penalty, l2_error_per_sample


@torch.no_grad()
def get_freqs_and_l0_norm(dataloader: DataLoader, encoder, cfg) -> torch.Tensor:
    act_freq_scores = torch.zeros(encoder.d_hidden).to(
        cfg["device"]
    )
    total = 0
    avg_l0_per_batch = []
    for batch in dataloader:
        features = batch.to(cfg["device"])
        hidden = encoder(features, return_acts=True)[1]
        avg_l0_per_batch.append((hidden > 0).sum(1).float().mean().item())
        act_freq_scores += (hidden > 0).sum(0)
        total += hidden.shape[0]
    act_freq_scores /= total
    avg_l0 = sum(avg_l0_per_batch) / len(avg_l0_per_batch)
    return act_freq_scores, avg_l0


@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.enc.weight))
    new_W_dec = torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.dec.weight))
    new_b_enc = torch.zeros_like(encoder.enc.bias)
    encoder.enc.weight.data[indices, :] = new_W_enc[indices,]
    encoder.dec.weight.data[:, indices] = new_W_dec[:, indices]
    encoder.enc.bias.data[indices] = new_b_enc[indices]


####################
# Eval functions

def calc_feature_hist_and_densities(sae_dir: str):
    acts = np.load(os.path.join(sae_dir, "feature_activations.npy"), mmap_mode="r")
    all_hist = []
    all_bin_edges = []
    all_densities = []
    for i in tqdm(range(acts.shape[1])):
        feature_acts = acts[:, i]
        hist, bin_edges = np.histogram(feature_acts, bins="auto")
        all_hist.append(hist)
        all_bin_edges.append(bin_edges)
        density = feature_acts[feature_acts > 0].shape[0] / feature_acts.shape[0]
        all_densities.append(density)
    np.savez(os.path.join(sae_dir, "feature_histograms.npz"), *all_hist)
    np.savez(os.path.join(sae_dir, "feature_bin_edges.npz"), *all_bin_edges)
    np.save(os.path.join(sae_dir, "feature_densities.npy"), all_densities)


def calc_feature_metrics(sae_dir: str, k: int = 50,):
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
    
    with open(f"{sae_dir}/config.json", 'r') as f:
        cfg = json.load(f)
    input_feature_dirs = cfg["data_dirs"]
    input_feature_dirs = [os.path.join(data_dir, "input_features") for data_dir in input_feature_dirs]
    
    topk_filename = f"top-{k}_activations.csv"
    topk_file = os.path.join(sae_dir, topk_filename)
    topk_df = pd.read_csv(topk_file)
    # filter out dead features
    topk_df = topk_df[topk_df["act_value"] != 0]
    word_ids = topk_df["word_id"].unique()

    model_names = cfg["model_names"]
    #TODO: add support for > 2 models
    assert len(model_names) == 2, "Currently only supports comparison between 2 models"
    model_names_label = "_".join(model_names)

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
    prob_avg_diff = []
    logprob_avg_diff = []
    logprob_median_diff = []
    prob_median_diff = []
    prob_diff_var = []
    prob_diff_kurtosis = []
    diff_consistency = []
    num_samples = []
    model_prob_variances = {model_name: [] for model_name in model_names}
    
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
        feature_logprobs = np.array(feature_logprobs).T # 50 x 2
        feature_probs = np.exp(feature_logprobs)
        feature_probs = feature_probs[sample_indices] # num_samples x 2
        feature_embeddings_mean = np.mean(feature_embeddings, axis=0)   # 768
        feature_probs_mean = np.mean(feature_probs, axis=0)   # 2
        embedding_dist = np.linalg.norm(feature_embeddings - feature_embeddings_mean, axis=1)
        embedding_cos_sim = np.dot(feature_embeddings, feature_embeddings_mean) / (np.linalg.norm(feature_embeddings, axis=1) * np.linalg.norm(feature_embeddings_mean))
        prob_dist = np.linalg.norm(feature_probs - feature_probs_mean, axis=1)
        prob_diff = feature_probs[:, 0] - feature_probs[:, 1]
        logprob_diff = feature_logprobs[:, 0] - feature_logprobs[:, 1]

        feature_indices.append([feature] * feature_probs.shape[0])
        sample_centroid_embedding_dist.append(embedding_dist)
        sample_centroid_cos_sim.append(embedding_cos_sim)

        embedding_avg_dist.append(np.mean(embedding_dist))
        embedding_avg_cos_sim.append(np.mean(embedding_cos_sim))
        prob_avg_dist.append(np.mean(prob_dist))
        prob_avg_diff.append(np.mean(prob_diff))
        prob_median_diff.append(np.median(prob_diff))
        logprob_avg_diff.append(np.mean(logprob_diff))
        logprob_median_diff.append(np.median(logprob_diff))
        diff_consistency.append(max(np.sum(prob_diff > 0), np.sum(prob_diff < 0)) / prob_diff.shape[0])
        prob_diff_var.append(np.var(prob_diff))
        prob_diff_kurtosis.append(pd.Series(prob_diff).kurtosis())
        
        for i, model_name in enumerate(model_names):
            model_prob_variances[model_name].append(np.var(feature_probs[:, i]))
        num_samples.append(sample_indices[0].shape[0])
    distance_df = pd.DataFrame({
        "feature": topk_df["feature"].unique(),
        "num_samples_considered": num_samples,
        "embedding_avg_dist": embedding_avg_dist,
        "embedding_avg_cos_sim": embedding_avg_cos_sim,
        "prob_avg_dist": prob_avg_dist,
        "prob_avg_diff": prob_avg_diff,
        "prob_median_diff": prob_median_diff,
        "logprob_avg_diff": logprob_avg_diff,
        "logprob_median_diff": logprob_median_diff,
        "prob_diff_consistency": diff_consistency,
        "prob_diff_var": prob_diff_var,
        "prob_diff_kurtosis": prob_diff_kurtosis,
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
    
def get_topk_words_in_context(
    sae_dir: str,
    k_activations: str
):
    topk_filename = f"top-{k_activations}_activations.csv"
    topk_file = os.path.join(sae_dir, topk_filename)
    topk_df = pd.read_csv(topk_file)
    # drop activations that are 0
    topk_df = topk_df[topk_df["act_value"] != 0]
    word_ids = topk_df["word_id"].unique()
    sae_cfg = json.load(open(os.path.join(sae_dir, "config.json")))
    input_feature_dirs = sae_cfg["data_dirs"]
    input_feature_dirs = [os.path.join(data_dir, "input_features") for data_dir in input_feature_dirs]
    words_in_context = {}
    for input_feature_dir in input_feature_dirs:
        words_in_context.update(get_words_in_context(input_feature_dir, word_ids))
    topk = topk_filename.split("_")[0]
    output_file = os.path.join(sae_dir, f"{topk}_words_in_context.json")
    with open(output_file, "w") as f:
        wic = json.dumps(words_in_context, indent=4)
        f.write(wic)
    return