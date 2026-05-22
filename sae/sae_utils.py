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
            base = f"{sae_name_prefix}_N={cfg['dict_size']}_l1={cfg['l1_coeff']}_lp={cfg['dec_penalty_coeff']}"
        else:
            base = f"N={cfg['dict_size']}_l1={cfg['l1_coeff']}_lp={cfg['dec_penalty_coeff']}"
    else:
        if sae_name_prefix:
            base = f"{sae_name_prefix}_N={cfg['dict_size']}_k={cfg['topk']}_lp={cfg['dec_penalty_coeff']}"
        else:
            base = f"N={cfg['dict_size']}_k={cfg['topk']}_lp={cfg['dec_penalty_coeff']}"
    if cfg.get("normalize_per_part", False):
        base = f"{base}_znorm"
    odlw = cfg.get("output_dim_loss_weight", None)
    if odlw is not None:
        base = f"{base}_odlw={odlw}"
    return base


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
    cfg.setdefault("normalize_per_part", False)
    cfg.setdefault("output_dim_loss_weight", None)
    cfg.setdefault("embedding_dim", 768)
    cfg.setdefault("output_feature_dim", 0)
    cfg.setdefault("superimpose_loss_weight", 0.0)
    return cfg


def l2_loss_per_sample(x, x_reconstruct, loss_weights=None):
    sq_err = (x_reconstruct - x.float()).pow(2)
    if loss_weights is not None:
        sq_err = sq_err * loss_weights
    return sq_err.sum(-1).reshape(-1, 1)


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
        self.save_dir = cfg["train_save_dir"]

        # Within-feature curve consistency loss (superimpose-style). Penalizes the mean
        # squared distance between each active sample's min-max-normalized prob curve and
        # the feature's centroid. self._last_superimpose_loss is updated each forward()
        # so callers (train_sae.py) can log it without re-computing.
        self.superimpose_loss_weight = float(cfg.get("superimpose_loss_weight", 0.0))
        self._emb_dim = cfg.get("embedding_dim", 768)
        self._out_dim = cfg.get("output_feature_dim", 0)
        self._last_superimpose_loss = 0.0

        # Per-dim loss weights: 1.0 on embedding dims, output_dim_loss_weight on prob dims.
        # If output_dim_loss_weight == "auto", the per-dim prob weight is set so that the
        # prob block's total MSE contribution matches the ratio implied by
        # output_feature_weight (ofw):
        #     tail_w = (ofw / (1-ofw)) * (embedding_dim / output_feature_dim)
        # so total_prob_loss / total_emb_loss == ofw / (1-ofw).
        # Falls back to embedding_dim / output_feature_dim (i.e. 50:50) when ofw is unset
        # or not in (0, 1).
        odlw = cfg.get("output_dim_loss_weight", None)
        emb_dim = cfg.get("embedding_dim", 768)
        out_dim = cfg.get("output_feature_dim", 0)
        if odlw is not None and out_dim > 0:
            if isinstance(odlw, str) and odlw == "auto":
                ofw = cfg.get("output_feature_weight", None)
                if ofw is not None and 0.0 < float(ofw) < 1.0:
                    ratio = float(ofw) / (1.0 - float(ofw))
                else:
                    ratio = 1.0
                tail_w = ratio * float(emb_dim) / float(out_dim)
            else:
                tail_w = float(odlw)
            weights = torch.ones(cfg["input_dim"], dtype=dtype)
            weights[emb_dim:emb_dim + out_dim] = tail_w
            self.register_buffer("loss_weights", weights)
        else:
            self.loss_weights = None

    def forward(self, x, return_acts=False, return_l2_error_per_sample=False):
        x_cent = x - self.dec.bias
        acts = F.relu(self.enc(x_cent))
        x_reconstruct = self.dec(acts)
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct, self.loss_weights)
        l2_loss = l2_error_per_sample.mean(0)
        l1_loss = self.l1_coeff * (acts.abs().sum().mean())
        loss = l2_loss + l1_loss
        if self.dec_penalty_coeff is not None:
            penalty = self.laplace_dec_penalty()
            loss += penalty
        if self.superimpose_loss_weight > 0.0:
            sup_loss = self.superimpose_consistency_loss(x, acts)
            self._last_superimpose_loss = float(sup_loss.detach().item())
            loss = loss + self.superimpose_loss_weight * sup_loss
        else:
            self._last_superimpose_loss = 0.0
        if not return_acts:
            acts = None
        if not return_l2_error_per_sample:
            l2_error_per_sample = None
        return loss, acts, penalty, l2_error_per_sample

    def _compute_acts(self, x):
        """Return (acts, x_cent) for this encoder's activation pattern. Subclasses
        override to apply topk / batch-topk gating. The base class uses ReLU."""
        x_cent = x - self.dec.bias
        acts = F.relu(self.enc(x_cent))
        return acts, x_cent

    def forward_with_components(self, x):
        """Forward pass returning loss components separately as a dict, so callers
        can take per-loss gradients for GradNorm-style automatic weight balancing.

        Returns dict:
          recon_loss: scalar — weighted reconstruction MSE (the main objective)
          sup_loss:   scalar — raw superimpose loss (before its weight); 0 if disabled
          acts:       (B, F) sparse activations (or dense for base AutoEncoder)
          x_reconstruct: (B, D) reconstruction
          l2_error_per_sample: (B, 1) per-sample weighted MSE (used by freq-weighting)

        Note: l1 penalty and laplace_dec_penalty are NOT included here — they aren't
        active in the GradNorm-targeted training paths. Reconstruction is the only
        primary loss; sup_loss and (externally-computed) ortho_loss are the aux losses
        whose weights GradNorm tunes.
        """
        acts, _ = self._compute_acts(x)
        x_reconstruct = self.dec(acts)
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct, self.loss_weights)
        recon_loss = l2_error_per_sample.mean(0).squeeze()
        sup_loss = self.superimpose_consistency_loss(x, acts)
        self._last_superimpose_loss = float(sup_loss.detach().item())
        return {
            "recon_loss": recon_loss,
            "sup_loss": sup_loss,
            "acts": acts,
            "x_reconstruct": x_reconstruct,
            "l2_error_per_sample": l2_error_per_sample,
        }

    def superimpose_consistency_loss(self, x, acts):
        """Mean squared distance between each active sample's min-max-normalized prob
        curve and its feature's centroid curve, uniformly averaged over active (sample,
        feature) pairs. Gradient flows through `acts` via the centroid (a weighted mean
        of active samples' curves), not via a per-pair activation weight — that
        previously let the encoder dodge the loss by attenuating activations on
        dissimilar samples. Only features with ≥2 active samples contribute. Returns 0
        if output_feature_dim ≤ 1 (structural — only meaningful with ≥2 timesteps).
        Always computed when called from forward_with_components so GradNorm can take
        per-loss grads regardless of the static superimpose_loss_weight."""
        if self._out_dim <= 1 or acts is None:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        emb_dim, out_dim = self._emb_dim, self._out_dim
        prob = x[:, emb_dim:emb_dim + out_dim].float()  # (B, T) — no grad through x
        mins = prob.min(dim=-1, keepdim=True).values
        maxs = prob.max(dim=-1, keepdim=True).values
        rng = (maxs - mins).clamp(min=1e-6)
        norm_prob = (prob - mins) / rng  # (B, T) each row in [0,1]

        # Use activation magnitudes (not a hard mask) as soft weights so gradient flows
        # back through `acts` → encoder weights.
        w = acts.float()  # (B, F) — sparse from topk, zeros are exactly zero
        active_count = (w > 0).float().sum(dim=0)  # (F,)
        active_feat = active_count >= 2  # only features with ≥2 active samples
        if not active_feat.any():
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        w_sum = w.sum(dim=0).clamp(min=1e-6)  # (F,)
        # Weighted centroid prob curve per feature
        w_detached = w.detach()
        centroids = (w_detached.T @ norm_prob) / w_sum.unsqueeze(-1)  # (F, T)

        # Active (sample, feature) pairs
        sample_idx, feat_idx = (w > 0).nonzero(as_tuple=True)
        keep = active_feat[feat_idx]
        sample_idx = sample_idx[keep]
        feat_idx = feat_idx[keep]
        if sample_idx.numel() == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        diffs = norm_prob[sample_idx] - centroids[feat_idx]  # (n_active, T)
        # Mean squared diff along the curve. Smoother gradient than L∞ (which
        # only penalizes the single worst timestep) and uniform weighting across
        # active pairs prevents the encoder from dodging the loss by attenuating
        # activations on curve-dissimilar samples — gradient still flows through
        # `acts` via the centroid term `(w.T @ norm_prob) / w_sum`.
        l2_per_pair = diffs.pow(2).mean(dim=-1)  # (n_active,)
        return l2_per_pair.mean()

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

    def _compute_acts(self, x):
        x_cent = x - self.dec.bias
        post_enc = self.enc(x_cent)
        topk = torch.topk(post_enc, k=self.k, dim=-1)
        values = F.relu(topk.values)
        acts = torch.zeros_like(post_enc).scatter_(-1, topk.indices, values)
        return acts, x_cent

    def forward(self, x, return_acts=False, return_l2_error_per_sample=False):
        x_cent = x - self.dec.bias
        post_enc = self.enc(x_cent)
        topk = torch.topk(post_enc, k=self.k, dim=-1)
        # use relu to ensure non-negative activations, but may not be necessary
        values = F.relu(topk.values)
        acts = torch.zeros_like(post_enc).scatter_(-1, topk.indices, values)
        x_reconstruct = self.dec(acts)
        penalty = torch.tensor(0.0)
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct, self.loss_weights)
        loss = l2_error_per_sample.mean(0)
        if self.dec_penalty_coeff is not None:
            penalty = self.laplace_dec_penalty()
            loss += penalty
        if self.superimpose_loss_weight > 0.0:
            sup_loss = self.superimpose_consistency_loss(x, acts)
            self._last_superimpose_loss = float(sup_loss.detach().item())
            loss = loss + self.superimpose_loss_weight * sup_loss
        else:
            self._last_superimpose_loss = 0.0
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

    def _compute_acts(self, x):
        batch_k = self.k * x.shape[0]
        x_cent = x - self.dec.bias
        post_enc = self.enc(x_cent)
        post_enc_flat = post_enc.view(1, -1)
        batch_topk = torch.topk(post_enc_flat, k=batch_k, dim=-1)
        values = F.relu(batch_topk.values)
        acts = torch.zeros_like(post_enc_flat).scatter_(-1, batch_topk.indices, values)
        acts = acts.view(post_enc.size())
        return acts, x_cent

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
        l2_error_per_sample = l2_loss_per_sample(x, x_reconstruct, self.loss_weights)
        loss = l2_error_per_sample.mean(0)
        if self.dec_penalty_coeff is not None:
            penalty = self.laplace_dec_penalty()
            loss += penalty
        if self.superimpose_loss_weight > 0.0:
            sup_loss = self.superimpose_consistency_loss(x, acts)
            self._last_superimpose_loss = float(sup_loss.detach().item())
            loss = loss + self.superimpose_loss_weight * sup_loss
        else:
            self._last_superimpose_loss = 0.0
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
    # Load fully into RAM to avoid repeated column-slice reads on row-major memmap
    acts = np.load(os.path.join(sae_dir, "feature_activations.npy"))
    all_hist = []
    all_bin_edges = []
    # Vectorize density: fraction of non-zero activations per feature
    all_densities = ((acts > 0).sum(axis=0) / acts.shape[0]).tolist()
    for i in tqdm(range(acts.shape[1])):
        feature_acts = acts[:, i]
        hist, bin_edges = np.histogram(feature_acts, bins="auto")
        all_hist.append(hist)
        all_bin_edges.append(bin_edges)
    np.savez(os.path.join(sae_dir, "feature_histograms.npz"), *all_hist)
    np.savez(os.path.join(sae_dir, "feature_bin_edges.npz"), *all_bin_edges)
    np.save(os.path.join(sae_dir, "feature_densities.npy"), all_densities)


def calc_feature_metrics(sae_dir: str, data_dirs:list, k: int = 50):
    def get_embeddings_from_word_ids(
        input_feature_dir: str,
        word_ids: list[str],
    ) -> dict[str, np.ndarray]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        file_df = pd.read_csv(f"{input_feature_dir}/file_to_doc.csv")
        file_df.drop(columns=["num_words"], inplace=True)
        file_df["doc_id"] = file_df["doc_id"].astype(str)
        all_doc_word_ids = {}
        for word_id in word_ids:
            doc_id = "_".join(word_id.split("_")[:-1])
            if doc_id in all_doc_word_ids:
                all_doc_word_ids[doc_id].append(word_id)
            else:
                all_doc_word_ids[doc_id] = [word_id]
        doc_ids = list(all_doc_word_ids.keys())
        file_df = file_df[file_df["doc_id"].isin(doc_ids)]
        # Group doc_ids by file upfront to avoid O(n) scan per file
        file_to_docs = file_df.groupby("file")["doc_id"].apply(list).to_dict()

        embedding_cols = [f"embedding_{i}" for i in range(768)]
        cols = ["word_id"] + embedding_cols

        def _process_file(file, docs):
            filepath = os.path.join(input_feature_dir, file)
            file_word_ids = []
            for doc in docs:
                file_word_ids += all_doc_word_ids[doc]
            words_df = pd.read_parquet(filepath, columns=cols, filters=[("word_id", 'in', file_word_ids)])
            embeddings = words_df[embedding_cols].to_numpy()
            return dict(zip(words_df["word_id"].tolist(), embeddings))

        orig_embeddings = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_process_file, file, docs): file
                for file, docs in file_to_docs.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                orig_embeddings.update(future.result())
        return orig_embeddings
    
    with open(f"{sae_dir}/config.json", 'r') as f:
        cfg = json.load(f)
    input_feature_dirs = data_dirs
    input_feature_dirs = [os.path.join(data_dir, "input_features") for data_dir in input_feature_dirs]
    
    topk_filename = f"top-{k}_activations.csv"
    topk_file = os.path.join(sae_dir, topk_filename)
    topk_df = pd.read_csv(topk_file)
    # filter out dead features
    topk_df = topk_df[topk_df["act_value"] != 0]
    word_ids = topk_df["word_id"].unique()

    model_names = cfg["model_names"]
    #TODO: add support for > 2 models
    # assert len(model_names) == 2, "Currently only supports comparison between 2 models"
    # model_names_label = "_".join(model_names)
    # model_names_label = "n_moreearly_models"
    

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
    
    grouped = topk_df.groupby("feature")
    for feature, feature_df in tqdm(grouped):
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
    distance_df.to_csv(os.path.join(sae_dir, f"feature_metrics.csv"), index=False)
    print(f"Saved feature metrics to {os.path.join(sae_dir, f'feature_metrics.csv')}")
    
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
    k_activations: str,
    data_dirs: list
):
    topk_filename = f"top-{k_activations}_activations.csv"
    topk_file = os.path.join(sae_dir, topk_filename)
    topk_df = pd.read_csv(topk_file)
    # drop activations that are 0
    topk_df = topk_df[topk_df["act_value"] != 0]
    word_ids = topk_df["word_id"].unique()
    sae_cfg = json.load(open(os.path.join(sae_dir, "config.json")))
    input_feature_dirs = data_dirs
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