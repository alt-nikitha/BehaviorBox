import click
import dask.dataframe as dd
import json
import logging
import numpy as np
import os
import pandas as pd
import pprint
import time
import torch
import wandb

from collections import Counter

from dask import config as dask_cfg
from dask.distributed import Client
from natsort import natsorted
from tqdm import tqdm

from create_cached_data import cache_data_per_dir
from data_utils import copy_temp_memmap, cleanup_temp_memmap
from sae_utils import (
    DTYPES,
    DataLoader,
    BackgroundDataLoader,
    get_encoder,
    load_checkpoint,
    get_sae_name,
    get_config,
    get_freqs_and_l0_norm,
    re_init,
)

# number of batches
# may need to be adjusted depending on
# 1) size of your data and 
# 2) batch size
log_eval_loss_every = 50
# log_eval_act_freqs_every = 30000
log_eval_act_freqs_every = 2000
checkpoint_every = 50000

reset_dead_threshold = 0.15


def load_frequency_weights(
    cache_dir: str,
    unigram_freq_path: str,
    data_dir: str,
    smoothing: float = 1.0,
    power: float = 1.0,
) -> np.ndarray:
    """Load unigram frequencies and compute inverse frequency weights for each sample.
    
    Args:
        cache_dir: Directory containing word_ids.pkl
        unigram_freq_path: Path to unigram_freqs.csv file
        data_dir: Original data directory to get word_id mapping
        smoothing: Smoothing factor for inverse frequency (higher = less weight variation)
        power: Power to raise inverse frequency to (higher = more extreme weighting)
    
    Returns:
        Array of weights, one per sample
    """
    import dask.dataframe as dd
    
    # Load unigram frequencies
    freq_df = pd.read_csv(unigram_freq_path)
    word_to_count = dict(zip(freq_df['word'], freq_df['count']))
    max_count = max(word_to_count.values())
    
    # Load word_id -> word mapping from original data
    input_feat_dir = os.path.join(data_dir, "input_features")
    df = dd.read_parquet(input_feat_dir, columns=["word_id", "word"]).compute()
    df["word_id"] = df["word_id"].astype(str)
    word_id_to_word = dict(zip(df["word_id"], df["word"]))
    
    # Load word_ids for this cached data
    word_ids_path = os.path.join(cache_dir, "word_ids.pkl")
    
    word_ids = pd.read_pickle(word_ids_path)
    
    # Compute weight for each sample
    weights = []
    for wid in tqdm(word_ids, desc="Computing frequency weights"):
        word = word_id_to_word.get(str(wid), '')
        count = word_to_count.get(word, 1)  # default to 1 if not found
        # Inverse frequency weight: rarer words get higher weights
        weight = ((max_count + smoothing) / (count + smoothing)) ** power
        weights.append(weight)
    
    weights = np.array(weights, dtype=np.float32)
    # Normalize so mean weight is 1.0 (preserves overall loss scale)
    weights = weights / weights.mean()
    return weights


def compute_unigram_freqs(cache_root: str, data_dir: str, out_path: str) -> None:
    """Compute unigram frequencies from cached word_ids and write to CSV."""
    input_feat_dir = os.path.join(data_dir, "input_features")
    df = dd.read_parquet(input_feat_dir, columns=["word_id", "word"]).compute()
    df["word_id"] = df["word_id"].astype(str)
    word_id_to_word = dict(zip(df["word_id"], df["word"]))

    ctr = Counter()
    for root, _, files in os.walk(cache_root):
        if "word_ids.pkl" in files:
            ids = pd.read_pickle(os.path.join(root, "word_ids.pkl"))
            
            words = [word_id_to_word.get(str(wid), "") for wid in ids]
            ctr.update([w for w in words if w])

    rows = [{"word": word, "count": cnt} for word, cnt in ctr.most_common()]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def get_logger():
    logging_dir = "../logs/sae"
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        filename=f"{logging_dir}/{timestr}.log",
        format='%(asctime)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    return logger


def init_wandb(cfg):
    model_type = cfg["type"]
    if model_type == "TopKAutoEncoder":
        wandb.init(
            project="bbox-sae",
                    config=cfg,
                    name=cfg["name"],
                    tags=["topk_sae", f"{cfg['dict_size']=}", f"{cfg['topk']=}"],
        )
    elif model_type == "BatchTopKAutoEncoder":
        wandb.init(
            project="bbox-sae",
                    config=cfg,
                    name=cfg["name"],
                    tags=["batch_topk_sae", f"{cfg['dict_size']=}", f"{cfg['topk']=}"],
        )
    else:
        wandb.init(
            project="bbox-sae",
                    config=cfg,
                    name=cfg["name"],
                    tags=[f"{cfg['dict_size']=}", f"{cfg['l1_coeff']=}"],
                )


def check_data_training_complete(checkpoint_dir):
    if os.path.exists(f"{checkpoint_dir}/training_complete"):
        return True
    return False


def load_data(
    workers: int,
    data_dirs: list[str],
    model_names: list[str],
    output_feature_weight: float,
    cache_dir: str,
    spill_dir: str,
    model_string: str,
    use_delta_prob: bool = False,
    use_delta_logprob: bool = False,
    normalize_per_part: bool = False,
):
    dask_cfg.set({'distributed.scheduler.worker-ttl': None})
    dask_cfg.set({
        "distributed.worker.memory.spill": 0.85,
        "distributed.worker.memory.target": 0.75,
        "distributed.worker.memory.terminate": 0.98,
    })
    client = Client(
        n_workers=workers, memory_limit='30GB', processes=True, timeout='30s', local_directory=spill_dir
    )
    print(client, flush=True)
    cache_data_filepaths = []
    for data_dir in data_dirs:
        cache_data_dir = cache_data_per_dir(
            model_string=model_string,
            client=client,
            cache_dir=cache_dir,
            data_dir=data_dir,
            model_names=model_names,
            output_feature_weight=output_feature_weight,
            use_delta_prob=use_delta_prob,
            use_delta_logprob=use_delta_logprob,
            normalize_per_part=normalize_per_part,
        )
        cache_data_filepath = os.path.join(cache_data_dir, "preprocessed_data.dat")
        cache_data_filepaths.append(cache_data_filepath)
    return cache_data_filepaths


def get_train_eval_indices(
    data,
    train_eval_ratio: float = 0.9,
    data_shuffling_seed: int = 0,
    split_seed: int = 0,
) -> tuple[list[int], list[int]]:
    np.random.seed(split_seed)
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.seed(data_shuffling_seed)
    print(f"Shuffling data with seed={data_shuffling_seed}...", flush=True)
    np.random.shuffle(indices)
    # split the data into 90% train and 10% eval
    train_indices = indices[:int(train_eval_ratio*num_samples)]
    eval_indices = indices[int(train_eval_ratio*num_samples):]
    return train_indices, eval_indices


def filter_train_indices_by_variance(
    data: np.ndarray,
    train_indices: np.ndarray,
    embedding_dim: int,
    top_pct: float,
    logger,
    mode: str = "linear",
) -> np.ndarray:
    """Restrict train_indices to the top `top_pct` fraction by per-sample variance
    across the prob/output block (columns embedding_dim:).

    `mode` controls the metric:
      - "linear": variance of raw values in the prob block (existing behavior).
      - "log":    variance of log(clip(value, eps, inf)) -- rebalances toward
                  relative changes, so rare-token emergence (1e-5 -> 1e-2) is
                  weighed comparably to easy-token swings (0.1 -> 0.7).

    Variance is computed over the full dataset and the top-X% set is then intersected
    with the existing train split, so the eval split is unchanged (still drawn from the
    full data distribution).
    """
    if top_pct is None or top_pct >= 1.0:
        return train_indices
    if not (0.0 < top_pct < 1.0):
        raise ValueError(f"variance_filter_top_pct must be in (0, 1], got {top_pct}")
    if mode not in ("linear", "log"):
        raise ValueError(f"variance_filter_mode must be 'linear' or 'log', got {mode}")
    print(
        f"Computing per-sample prob-block variance for variance filtering "
        f"(top {top_pct*100:.1f}%, mode={mode})...",
        flush=True,
    )
    prob_block = np.asarray(data[:, embedding_dim:], dtype=np.float32)
    if mode == "log":
        # Caveat: assumes prob_block is in approximately linear-prob space.
        # If you've enabled normalize_per_part or use_delta_logprob, prob_block
        # is not raw probability -- log-mode will still run but the metric is
        # `var(log|clip(x, eps, inf)|)`, which may not match the intuition.
        eps = 1e-8
        prob_block = np.log(np.maximum(np.abs(prob_block), eps))
    variances = prob_block.var(axis=1)
    n_keep_total = max(1, int(len(variances) * top_pct))
    # argpartition gives a (potentially unordered) set of the top-k by variance
    top_global = np.argpartition(-variances, n_keep_total - 1)[:n_keep_total]
    keep_mask = np.zeros(len(variances), dtype=bool)
    keep_mask[top_global] = True
    filtered = train_indices[keep_mask[train_indices]]
    msg = (
        f"Variance filter ({mode}): kept {len(filtered)}/{len(train_indices)} train "
        f"samples (global threshold = top {top_pct*100:.1f}% of {len(variances)} "
        f"samples by var)"
    )
    logger.info(msg)
    print(msg, flush=True)
    return filtered


def filter_train_indices_by_word_ids(
    train_indices: np.ndarray,
    cache_data_dir: str,
    subset_path: str,
    logger,
) -> np.ndarray:
    """Restrict train_indices to memmap rows whose word_id is in `subset_path`.

    `subset_path` is a JSONL with one object per line carrying a "word_id" field
    (e.g. the jump-interval balanced sample). Alignment is by `word_ids.pkl`,
    which is row-aligned to the cached memmap. The eval split is untouched, so
    evaluation still reflects the full data distribution.
    """
    keep = set()
    with open(subset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                keep.add(str(json.loads(line)["word_id"]))

    wid_path = os.path.join(cache_data_dir, "word_ids.pkl")
    if not os.path.exists(wid_path):
        # word_ids.pkl is written one level up from the ofw=*/ memmap subdir
        wid_path = os.path.join(os.path.dirname(cache_data_dir), "word_ids.pkl")
    word_ids = pd.read_pickle(wid_path)
    wid_arr = np.asarray([str(w) for w in word_ids])

    mask = np.isin(wid_arr, list(keep))
    filtered = train_indices[mask[train_indices]]
    msg = (
        f"word_id subset filter: kept {len(filtered)}/{len(train_indices)} train "
        f"rows (subset has {len(keep)} ids; {int(mask.sum())} matched in memmap) "
        f"from {subset_path}"
    )
    logger.info(msg)
    print(msg, flush=True)
    if len(filtered) == 0:
        raise RuntimeError(
            "word_id subset filter kept 0 train rows -- check that subset "
            "word_ids match the cache's word_ids.pkl format."
        )
    return filtered


def cluster_stratify_train_indices(
    data: np.ndarray,
    train_indices: np.ndarray,
    embedding_dim: int,
    n_clusters: int,
    per_cluster: int,
    logger,
    mode: str = "linear",
    seed: int = 0,
    fit_subsample: int = 10_000_000,
    predict_chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Cluster training samples by their prob-block trajectory shape, then take an
    equal-sized sample from each cluster.

    This forces the training set to cover diverse trajectory shapes (flat-high,
    flat-low, early-rise, late-rise, jump, ...) instead of being dominated by the
    most common shape.

    `mode`:
      - "linear": cluster on raw prob-block values.
      - "log":    cluster on log(clip(|prob|, eps, inf)) -- same rationale as the
                  log-mode variance filter.

    Memory handling:
      - If `len(train_indices) > fit_subsample`, K-means is fit on a random subset
        of that size, then cluster labels are predicted for the full train set in
        chunks of `predict_chunk_size`. Default fit_subsample=10M handles datasets
        up to that size in-memory; subsamples beyond.
      - Pass `fit_subsample=0` to disable subsampling and force fit on all rows.

    Returns: subset of `train_indices`, size ~= n_clusters * per_cluster
    (less if some clusters are smaller than `per_cluster`).
    """
    if n_clusters is None or n_clusters <= 0 or per_cluster is None or per_cluster <= 0:
        return train_indices
    if mode not in ("linear", "log"):
        raise ValueError(f"cluster_strat_mode must be 'linear' or 'log', got {mode}")
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    n_train = len(train_indices)
    eps = 1e-8

    def _slice(rows: np.ndarray) -> np.ndarray:
        """Materialize the prob block for the given absolute rows, applying mode."""
        arr = np.asarray(data[rows, embedding_dim:], dtype=np.float32)
        if mode == "log":
            arr = np.log(np.maximum(np.abs(arr), eps))
        return arr

    use_subsample = fit_subsample and 0 < fit_subsample < n_train
    if use_subsample:
        fit_local = rng.choice(n_train, size=fit_subsample, replace=False)
        fit_rows = train_indices[fit_local]
        print(
            f"Cluster-stratified sampling: fitting K-means on {fit_subsample}-sample "
            f"subset of {n_train} train rows, k={n_clusters}, mode={mode}...",
            flush=True,
        )
        fit_traj = _slice(fit_rows)
    else:
        print(
            f"Cluster-stratified sampling: fitting K-means on all {n_train} train rows, "
            f"k={n_clusters}, mode={mode}...",
            flush=True,
        )
        fit_traj = _slice(train_indices)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=3,
        batch_size=min(10000, max(1024, len(fit_traj) // 100)),
        max_iter=100,
    )
    km.fit(fit_traj)
    del fit_traj  # free memory before predict pass

    # Predict labels for every train index, chunked to bound memory.
    labels = np.empty(n_train, dtype=np.int32)
    chunk = max(1, predict_chunk_size)
    for start in range(0, n_train, chunk):
        end = min(n_train, start + chunk)
        chunk_rows = train_indices[start:end]
        labels[start:end] = km.predict(_slice(chunk_rows))

    chosen_local: list[int] = []
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        take = min(len(idx), per_cluster)
        chosen_local.extend(rng.choice(idx, size=take, replace=False).tolist())
    chosen_local_arr = np.array(chosen_local, dtype=np.int64)
    filtered = train_indices[chosen_local_arr]
    msg = (
        f"Cluster-stratified sampling ({mode}): kept "
        f"{len(filtered)}/{n_train} train samples "
        f"(n_clusters={n_clusters}, per_cluster<= {per_cluster}; "
        f"fit_subsample={'all' if not use_subsample else fit_subsample}; "
        f"cluster sizes min/median/max = "
        f"{cluster_sizes.min()}/{int(np.median(cluster_sizes))}/{cluster_sizes.max()})"
    )
    logger.info(msg)
    print(msg, flush=True)
    return filtered


def train(
    data,
    logger,
    save_dir,
    model_checkpoint_dir: str = None,
    num_epochs: int = 1,
    continue_from_checkpoint: bool = False,
    checkpoint_path: str = None,
    cfg: dict = None,
    sample_weights: np.ndarray = None,
):      
    def _save_and_log_hist(freqs: list[float], cur_step: int):
        nonlocal encoder
        nonlocal save_dir
        nonlocal global_start_step
        cur_global_step = [cur_step + global_start_step] * len(freqs)
        hist_fig = encoder.make_histogram(freqs, cur_global_step)
        hist_fig.savefig(f"{save_dir}/act_freqs.png")
        wandb.log({"act_freqs": wandb.Image(f"{save_dir}/act_freqs.png")})

    if continue_from_checkpoint:
        logger.info("Resuming from checkpoint...")
        print("Resuming from checkpoint...", flush=True)
        # get the latest checkpoint if checkpoint path is not provided
        if checkpoint_path is None:
            checkpoints = os.listdir(model_checkpoint_dir)
            checkpoints = [checkpoint for checkpoint in checkpoints if ".pt" in checkpoint]
            steps = [int(checkpoint.split("_")[-1].split(".")[0]) for checkpoint in checkpoints]
            latest_step = max(steps)
            checkpoint_path = f"{model_checkpoint_dir}/checkpoint_{latest_step}.pt"
            logger.info(f"Loading {checkpoint_path}")
            checkpoint = load_checkpoint(cfg, checkpoint_path)
            start_batch = checkpoint['step']
            epoch = checkpoint['epoch']
            # get the global step, which accounts for previous data files
            global_start_step = checkpoint['global_step']
        # the checkpoint we may want to load may be from a prev data file,
        # so we need to load the checkpoint from a different directory
        else:
            logger.info(f"Loading {checkpoint_path}")
            checkpoint = load_checkpoint(cfg, checkpoint_path)
            start_batch = 0
            global_start_step = checkpoint['global_step']
        encoder = checkpoint['encoder']
        encoder_optim = checkpoint['encoder_optim']
        # load activation frequencies
        act_freqs_df = pd.read_csv(f"{save_dir}/act_freqs.csv")
        act_freqs = act_freqs_df["act_freqs"].to_list()
        step = act_freqs_df["step"].to_list()
    else:
        encoder = get_encoder(cfg)
        encoder_optim = torch.optim.AdamW(
            encoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"])
        )
        start_batch = 0
        global_start_step = 0
        act_freqs = []
        step = []
        
    logger.info({
        "start_batch": start_batch,
        "global_start_step": global_start_step
    })

    batch_size=cfg["batch_size"]
    train_indices, eval_indices = get_train_eval_indices(data, data_shuffling_seed=cfg["data_shuffling_seed"])

    train_word_id_subset = cfg.get("train_word_id_subset", None)
    if train_word_id_subset:
        train_indices = filter_train_indices_by_word_ids(
            train_indices=train_indices,
            cache_data_dir=cfg.get("_word_ids_cache_dir", ""),
            subset_path=train_word_id_subset,
            logger=logger,
        )

    variance_filter_top_pct = cfg.get("variance_filter_top_pct", None)
    variance_filter_mode = cfg.get("variance_filter_mode", "linear")
    if variance_filter_top_pct is not None and variance_filter_top_pct < 1.0:
        train_indices = filter_train_indices_by_variance(
            data=data,
            train_indices=train_indices,
            embedding_dim=cfg.get("embedding_dim", 768),
            top_pct=variance_filter_top_pct,
            logger=logger,
            mode=variance_filter_mode,
        )

    cluster_strat_k = cfg.get("cluster_strat_k", 0) or 0
    cluster_strat_per_cluster = cfg.get("cluster_strat_per_cluster", 0) or 0
    cluster_strat_mode = cfg.get("cluster_strat_mode", "linear")
    if cluster_strat_k > 0 and cluster_strat_per_cluster > 0:
        train_indices = cluster_stratify_train_indices(
            data=data,
            train_indices=train_indices,
            embedding_dim=cfg.get("embedding_dim", 768),
            n_clusters=cluster_strat_k,
            per_cluster=cluster_strat_per_cluster,
            logger=logger,
            mode=cluster_strat_mode,
            seed=cfg.get("seed", 42),
        )

    use_freq_weighting = sample_weights is not None
    decoder_ortho_loss_weight = cfg.get("decoder_ortho_loss_weight", 0.0)
    superimpose_loss_weight_static = float(cfg.get("superimpose_loss_weight", 0.0) or 0.0)

    # GradNorm: when target_ratio is set (>0), the aux loss's weight is auto-tuned so
    # its gradient norm on shared params equals target_ratio × reconstruction's grad
    # norm. Updated every `gradnorm_every` steps with EMA smoothing. Static weights
    # above are used as the starting value and as a fallback when GradNorm is off.
    gradnorm_target_sup = float(cfg.get("gradnorm_target_sup", 0.0) or 0.0)
    gradnorm_target_ortho = float(cfg.get("gradnorm_target_ortho", 0.0) or 0.0)
    gradnorm_every = int(cfg.get("gradnorm_every", 50) or 50)
    gradnorm_ema = float(cfg.get("gradnorm_ema", 0.9) or 0.9)
    gradnorm_min_weight = float(cfg.get("gradnorm_min_weight", 1e-3) or 1e-3)
    gradnorm_max_weight = float(cfg.get("gradnorm_max_weight", 1e6) or 1e6)
    use_gradnorm_sup = gradnorm_target_sup > 0.0
    use_gradnorm_ortho = gradnorm_target_ortho > 0.0
    sup_weight_effective = superimpose_loss_weight_static if superimpose_loss_weight_static > 0 else 1.0
    ortho_weight_effective = decoder_ortho_loss_weight if decoder_ortho_loss_weight > 0 else 1.0

    logger.info(
        f"Beginning training... (frequency weighting: {use_freq_weighting}, "
        f"decoder_ortho_loss_weight: {decoder_ortho_loss_weight}, "
        f"superimpose_loss_weight: {superimpose_loss_weight_static}, "
        f"gradnorm: sup_target={gradnorm_target_sup} ortho_target={gradnorm_target_ortho} "
        f"every={gradnorm_every})"
    )
    encoder.train()
    # Data-side rescale (znorm path only): divide inputs by sqrt(sum of per-dim loss
    # weights) so total weighted variance ~= 1 and recon_loss reads ~O(1) instead of
    # ~hundreds. Uniform scalar => preserves the odlw block balance and cancels in EV.
    # AdamW makes the resulting global gradient rescale ~invariant, so lr is unchanged.
    if cfg.get("normalize_per_part", False):
        _S = encoder.loss_weights.sum().item() if encoder.loss_weights is not None else float(cfg["input_dim"])
        data_input_scale = 1.0 / np.sqrt(_S)
    else:
        data_input_scale = 1.0
    for epoch in range(num_epochs):
        train_dataloader = DataLoader(data, batch_size, train_indices)
        background_loader = BackgroundDataLoader(train_dataloader)
        # Evaluate on the exact subset the SAE was trained on (train_indices, after all
        # variance/word_id/cluster filtering) rather than the held-out eval split.
        eval_dataloader = DataLoader(data, batch_size, train_indices)
        total_batches = train_dataloader.num_batches * num_epochs
        k_start = 100
        k_end = cfg["topk"] # Your target k (e.g., 25)
        # anneal_until_batch = int(0.3 * total_batches) # Anneal over first 30% of training

        anneal_start_batch = int(0.1 * total_batches)
        anneal_end_batch = int(0.5 * total_batches)   # Finish at 50% mark
        # -----------------------------
        for i, batch in enumerate(background_loader):
            i += epoch * train_dataloader.num_batches
            # if i < anneal_until_batch:
            #     current_k = k_start - (k_start - k_end) * (i / anneal_until_batch)
            # else:
            #     current_k = k_end
            # encoder.k = int(current_k)

            if i < anneal_start_batch:
                current_k = k_start
            elif i < anneal_end_batch:
                # Linear decay over the middle 60% of training
                progress = (i - anneal_start_batch) / (anneal_end_batch - anneal_start_batch)
                current_k = k_start - (k_start - k_end) * progress
            else:
                current_k = k_end
            
            encoder.k = int(current_k)
            if i <= start_batch:
                continue
            batch = batch.to(cfg["device"]) * data_input_scale

            # Forward — returns separate loss components so we can take per-loss
            # gradients for GradNorm. Old `forward()` is preserved for callers that
            # still expect the (loss, acts, penalty, l2_err) tuple.
            components = encoder.forward_with_components(batch)
            recon_loss_raw = components["recon_loss"]
            sup_loss_raw = components["sup_loss"]
            l2_error_per_sample = components["l2_error_per_sample"]

            # Frequency weighting recomputes recon as weighted mean over per-sample
            # errors. (l1 from base AutoEncoder isn't routed through this path; only
            # the topk variants are GradNorm-targeted and l1_coeff is None for them.)
            if use_freq_weighting:
                batch_indices = train_dataloader.batch_indices[i % train_dataloader.num_batches]
                weights = torch.from_numpy(sample_weights[batch_indices]).to(cfg["device"])
                recon_loss = (l2_error_per_sample.squeeze() * weights).mean()
            else:
                recon_loss = recon_loss_raw

            with torch.no_grad():
                mse = l2_error_per_sample.mean()
                x_var_sq = (batch - batch.mean(0, keepdim=True)).pow(2)
                if encoder.loss_weights is not None:
                    x_var_sq = x_var_sq * encoder.loss_weights
                var_x = x_var_sq.sum(-1).mean()
                ev = (1.0 - mse / (var_x + 1e-8)).item()

            # Ortho loss: computed when either the static weight is nonzero OR GradNorm
            # is targeting it (so we have a tensor for autograd.grad).
            ortho_loss = None
            ortho_loss_val = None
            if decoder_ortho_loss_weight > 0 or use_gradnorm_ortho:
                W = encoder.dec.weight
                W_col_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
                sims = W_col_norm.T @ W_col_norm
                K = sims.shape[0]
                off_diag_sq_sum = (sims ** 2).sum() - K
                ortho_loss = off_diag_sq_sum / (K * (K - 1))
                ortho_loss_val = ortho_loss.item()

            # GradNorm update: every `gradnorm_every` batches, measure per-loss gradient
            # norms on shared (enc+dec) weights and EMA-update effective weights so each
            # aux loss's gradient is target_ratio × reconstruction's gradient.
            # torch.autograd.grad does NOT touch .grad — safe to interleave with the
            # subsequent combined backward() (which writes .grad as usual).
            # allow_unused=True: superimpose touches only enc.weight, ortho only dec.weight,
            # so each will report None for the unused tensor — treated as zero gradient.
            def _grad_norm(loss_tensor, params):
                grads = torch.autograd.grad(loss_tensor, params, retain_graph=True, allow_unused=True)
                sq = sum(g.pow(2).sum() for g in grads if g is not None)
                if isinstance(sq, int):  # all None — shouldn't happen, but be safe
                    return 0.0
                return torch.sqrt(sq).item()

            if (use_gradnorm_sup or use_gradnorm_ortho) and i > 0 and i % gradnorm_every == 0:
                shared = [encoder.enc.weight, encoder.dec.weight]
                n_recon = _grad_norm(recon_loss, shared)

                if use_gradnorm_sup and sup_loss_raw.requires_grad:
                    n_sup = _grad_norm(sup_loss_raw, shared)
                    if n_sup > 1e-12:
                        target = gradnorm_target_sup * n_recon / n_sup
                        sup_weight_effective = max(
                            gradnorm_min_weight,
                            min(gradnorm_max_weight,
                                gradnorm_ema * sup_weight_effective + (1 - gradnorm_ema) * target),
                        )

                if use_gradnorm_ortho and ortho_loss is not None and ortho_loss.requires_grad:
                    n_ortho = _grad_norm(ortho_loss, shared)
                    if n_ortho > 1e-12:
                        target = gradnorm_target_ortho * n_recon / n_ortho
                        ortho_weight_effective = max(
                            gradnorm_min_weight,
                            min(gradnorm_max_weight,
                                gradnorm_ema * ortho_weight_effective + (1 - gradnorm_ema) * target),
                        )

            # Resolve effective weights for this step
            sup_w = sup_weight_effective if use_gradnorm_sup else superimpose_loss_weight_static
            ortho_w = ortho_weight_effective if use_gradnorm_ortho else decoder_ortho_loss_weight

            # Combine losses
            loss = recon_loss
            if sup_w > 0 and sup_loss_raw.requires_grad:
                loss = loss + sup_w * sup_loss_raw
            if ortho_w > 0 and ortho_loss is not None:
                loss = loss + ortho_w * ortho_loss

            loss_dict = {
                "epoch": epoch,
                "batch": f"{i} / {total_batches}",
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "explained_variance": ev,
            }
            if ortho_loss_val is not None:
                loss_dict["decoder_ortho_loss"] = ortho_loss_val
                loss_dict["decoder_ortho_weight"] = ortho_w
            sup_loss_val = float(getattr(encoder, "_last_superimpose_loss", 0.0))
            if sup_loss_val > 0 or sup_w > 0:
                loss_dict["superimpose_loss"] = sup_loss_val
                loss_dict["superimpose_weight"] = sup_w
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            encoder_optim.step()
            encoder_optim.zero_grad()
            if i % 100 == 0:
                wandb.log(loss_dict)
                logger.info(loss_dict)
            if (i+1) % log_eval_loss_every == 0:
                num_eval_batches = eval_dataloader.num_batches
                if cfg["topk"]:
                    total_eval_loss_dict = {
                        "total_eval_loss": 0,
                        # "total_penalty": 0
                    }
                else:
                    total_eval_loss_dict = {
                        "total_eval_loss": 0,
                        "total_eval_l2_loss": 0,
                        "total_eval_l1_loss": 0
                    }
                total_resid_ss = 0.0
                total_x_var_ss = 0.0
                total_eval_samples = 0
                # record loss on eval data
                with torch.no_grad():
                    for eval_idx, eval_batch in enumerate(eval_dataloader):
                        eval_batch = eval_batch.to(cfg["device"]) * data_input_scale
                        if cfg["topk"]:
                            # loss, _, penalty, _ = encoder(eval_batch)
                            loss, _, _, l2_error_per_sample = encoder(eval_batch, return_l2_error_per_sample=True)
                            if use_freq_weighting:
                                batch_indices = eval_dataloader.batch_indices[eval_idx]
                                weights = torch.from_numpy(sample_weights[batch_indices]).to(cfg["device"])
                                loss = (l2_error_per_sample.squeeze() * weights).mean()
                            total_eval_loss_dict["total_eval_loss"] += loss.item()
                            # total_eval_loss_dict["total_penalty"] += penalty.item()
                        else:
                            loss, _, l2_loss, l1_loss = encoder(eval_batch)
                            _, _, _, l2_error_per_sample = encoder(eval_batch, return_l2_error_per_sample=True)
                            if use_freq_weighting:
                                batch_indices = eval_dataloader.batch_indices[eval_idx]
                                weights = torch.from_numpy(sample_weights[batch_indices]).to(cfg["device"])
                                weighted_l2_loss = (l2_error_per_sample.squeeze() * weights).mean()
                                acts = encoder.enc(eval_batch - encoder.dec.bias)
                                acts = torch.nn.functional.relu(acts)
                                l1_loss = cfg["l1_coeff"] * (acts.abs().sum(-1).mean())
                                loss = weighted_l2_loss + l1_loss
                                l2_loss = weighted_l2_loss
                            total_eval_loss_dict["total_eval_loss"] += loss.item()
                            total_eval_loss_dict["total_eval_l2_loss"] += l2_loss.item()
                            total_eval_loss_dict["total_eval_l1_loss"] += l1_loss.item()

                        bsz = eval_batch.shape[0]
                        total_resid_ss += l2_error_per_sample.sum().item()
                        # Apply the same per-dim loss_weights to the variance term so
                        # ev = 1 - resid/var is apples-to-apples; otherwise resid is
                        # weighted (~70x on prob block) but var isn't, breaking the metric.
                        x_var_sq = (eval_batch - eval_batch.mean(0, keepdim=True)).pow(2)
                        if encoder.loss_weights is not None:
                            x_var_sq = x_var_sq * encoder.loss_weights
                        total_x_var_ss += x_var_sq.sum().item()
                        total_eval_samples += bsz
                eval_ev = 1.0 - (total_resid_ss / (total_x_var_ss + 1e-8))
                if cfg["topk"]:
                    avg_eval_loss_dict = {
                        "eval_loss": total_eval_loss_dict["total_eval_loss"] / num_eval_batches,
                        "eval_explained_variance": eval_ev,
                        # "penalty": total_eval_loss_dict["total_penalty"] / num_eval_batches
                    }
                    wandb.log(avg_eval_loss_dict)
                    logger.info(avg_eval_loss_dict)
                else:
                    avg_eval_loss_dict = {
                            "eval_loss": total_eval_loss_dict["total_eval_loss"] / num_eval_batches,
                            "eval_l2_loss": total_eval_loss_dict["total_eval_l2_loss"] / num_eval_batches,
                            "eval_l1_loss": total_eval_loss_dict["total_eval_l1_loss"] / num_eval_batches,
                            "eval_explained_variance": eval_ev,
                        }
                    wandb.log(avg_eval_loss_dict)
                    logger.info(avg_eval_loss_dict)
            if (i+1) % log_eval_act_freqs_every == 0:
                logger.info("Getting activation frequencies on eval data...")
                freqs, l0 = get_freqs_and_l0_norm(eval_dataloader, encoder, cfg)
                freq_dict = {
                    "dead": (freqs == 0).float().mean().item(),
                    "below_1e-6": (freqs < 1e-6).float().mean().item(),
                    "below_1e-5": (freqs < 1e-5).float().mean().item(),
                    "l0_norm": l0
                }
                wandb.log(freq_dict)
                logger.info(freq_dict)
                
                act_freqs += freqs.tolist()
                step += [i] * len(freqs)
                act_freqs_df = pd.DataFrame.from_dict({
                    "act_freqs": act_freqs, 
                    "step": step
                })
                act_freqs_df.to_csv(f"{model_checkpoint_dir}/act_freqs.csv")
                _save_and_log_hist(freqs.tolist(), cur_step=i)
                
                if freq_dict["dead"] > reset_dead_threshold and \
                    (total_batches - i) > (log_eval_act_freqs_every // 4):
                    logger.info("Resetting neurons...")
                    # if dict size is large, reset more of the dead features
                    to_be_reset = (freqs == 0)
                    # if cfg["dict_size"]/cfg["input_dim"] > 4:    
                    #     to_be_reset = (freqs < 1e-6)
                    # else:
                    #     to_be_reset = (freqs == 0)
                    re_init(to_be_reset, encoder)
            if (i+1) % checkpoint_every == 0:
                # checkpoint model
                logger.info("Checkpointing model...")
                torch.save({
                    "model_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": encoder_optim.state_dict(),
                    "epoch": epoch,
                    "step": i,
                    "global_step": global_start_step    # only update global step if we are done training on this data file
                }, f"{model_checkpoint_dir}/checkpoint_{i}.pt")
                with open(f"{save_dir}/config.json", "w") as f:
                    json.dump(cfg, f, indent=4)
                logger.info(f"Model saved to {model_checkpoint_dir}/checkpoint_{i}.pt")
            cur_step = i
    # get final activation frequencies
    logger.info("Getting final activation frequencies...")
    freqs, l0 = get_freqs_and_l0_norm(eval_dataloader, encoder, cfg)
    wandb.log({
        "dead": (freqs == 0).float().mean().item(),
        "below_1e-6": (freqs < 1e-6).float().mean().item(),
        "below_1e-5": (freqs < 1e-5).float().mean().item(),
        "l0_norm": l0
    })
    step += [cur_step + global_start_step] * len(freqs)
    act_freqs += freqs.tolist()
    act_freqs_df = pd.DataFrame.from_dict({
        "act_freqs": act_freqs, 
        "step": step
    })
    act_freqs_df.to_csv(f"{save_dir}/act_freqs.csv")
    _save_and_log_hist(freqs.tolist(), cur_step=i)
    
    # checkpoint final model
    logger.info("Checkpointing final model...")
    final_checkpoint_path = f"{model_checkpoint_dir}/checkpoint_{i}.pt"
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "optimizer_state_dict": encoder_optim.state_dict(),
        "epoch": epoch,
        "step": i,
        "global_step": i + global_start_step    # only update global step if we are done training on this data file
    }, final_checkpoint_path)
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=4)
    logger.info(f"Model saved to {final_checkpoint_path}")
    return final_checkpoint_path, encoder
    

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
    type=click.Path(),
    default="cache/",
)
@click.option(
    "--checkpoint_dir",
    help="Directory to save checkpointed models",
    default=None,
)
@click.option(
    "--config_path",
    help="Custom config for model",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--data_dirs",
    help="Directories containing input and output features",
    type=click.Path(exists=True),
    multiple=True,
)
@click.option(
    "--data_shuffling_seed",
    help="Seed for shuffling data",
    type=int,
    default=0,
)
@click.option(
    "--model_names",
    help="Names of language models to evaluate, should match output feature directory names",
    type=str,
    multiple=True,
)
@click.option(
    "--num_epochs",
    help="Number of epochs to train",
    type=int,
    default=1,
)
@click.option(
    "--output_feature_weight",
    help="Relative weight of model probabilities",
    default=None,
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
    "--save_dir",
    help="Directory to save model directory in",
)
@click.option(
    "--seed",
    help="Random seed",
    type=int,
    default=42,
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
    default=None,
)
@click.option(
    "--workers",
    type=int,
    default=16,
)
@click.option(
    "--only_probs",
    type=bool,
    default=False,
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
    help="If > 0, add a loss penalizing squared off-diagonal cosine similarity between decoder dictionary directions",
    type=float,
    default=0.0,
)
@click.option(
    "--use_freq_weighting",
    help="Use inverse frequency weighting for rare tokens",
    type=bool,
    default=False,
)
@click.option(
    "--freq_weight_smoothing",
    help="Smoothing factor for inverse frequency weights (higher = less variation)",
    type=float,
    default=1.0,
)
@click.option(
    "--freq_weight_power",
    help="Power to raise inverse frequency to (higher = more extreme weighting)",
    type=float,
    default=1.0,
)
@click.option(
    "--normalize_per_part",
    help="Z-score normalize embedding and prob blocks independently (per-column). "
         "Cached separately and reflected in the SAE folder name.",
    type=bool,
    default=False,
)
@click.option(
    "--output_dim_loss_weight",
    help="Per-dim loss weight applied to the prob/output block. Pass 'auto' to use "
         "(ofw/(1-ofw)) * embedding_dim/output_feature_dim, or a float (e.g. 110). "
         "Default None = uniform.",
    type=str,
    default=None,
)
@click.option(
    "--checkpoint_weight_scheme",
    help="Distribute the per-dim prob loss weight across checkpoints. 'uniform' (default) "
         "weights every checkpoint equally. 'log_step' weights checkpoint j by log(step_j) "
         "(normalized to mean 1), tilting toward later checkpoints while preserving the "
         "block-level prob:emb loss ratio. Requires model_names of the form '*stepNNNN*'.",
    type=click.Choice(["uniform", "log_step"]),
    default="uniform",
)
@click.option(
    "--variance_filter_top_pct",
    help="If set (e.g. 0.2), restrict TRAINING to the top-X fraction of samples by "
         "per-sample variance across the prob/output block. Eval split is unchanged "
         "so in-training eval still reflects the full data distribution.",
    type=float,
    default=None,
)
@click.option(
    "--variance_filter_mode",
    help="How to compute per-sample variance for the filter: 'linear' uses raw "
         "prob-block values (default, existing behavior); 'log' uses log(|x|+eps), "
         "which rebalances toward relative changes so rare-token emergence is "
         "weighed comparably to easy-token swings. Assumes prob-block is in "
         "approximately linear-prob space; see function docstring for caveats with "
         "normalize_per_part or use_delta_logprob.",
    type=click.Choice(["linear", "log"]),
    default="linear",
)
@click.option(
    "--cluster_strat_k",
    help="If > 0, cluster training samples by their prob-trajectory shape into K "
         "clusters (MiniBatchKMeans) and take an equal-sized sample from each. "
         "Forces diversity across trajectory shapes (flat, early-rise, late-jump, ...). "
         "Applied AFTER variance_filter if both are set.",
    type=int,
    default=0,
)
@click.option(
    "--cluster_strat_per_cluster",
    help="When --cluster_strat_k > 0, the (max) number of samples to take from each "
         "cluster. Total kept <= cluster_strat_k * cluster_strat_per_cluster.",
    type=int,
    default=2000,
)
@click.option(
    "--cluster_strat_mode",
    help="Feature space used for clustering trajectories. Same options/meaning as "
         "--variance_filter_mode.",
    type=click.Choice(["linear", "log"]),
    default="linear",
)
@click.option(
    "--superimpose_loss_weight",
    help="If > 0, add a within-feature curve consistency loss that pulls samples activating "
         "the same feature toward similar min-max-normalized prob curves (mean L∞ distance to "
         "the feature centroid). Matches the superimpose_sim metric used in analysis. "
         "When --gradnorm_target_sup > 0, this is only the INITIAL value; the effective "
         "weight is auto-tuned each step.",
    type=float,
    default=0.0,
)
@click.option(
    "--gradnorm_target_sup",
    help="If > 0, GradNorm-style auto-balance the superimpose loss weight so its gradient "
         "norm on shared (enc+dec) weights equals this fraction of reconstruction's gradient "
         "norm. Recommended: 0.05–0.2. Overrides --superimpose_loss_weight beyond the initial.",
    type=float,
    default=0.0,
)
@click.option(
    "--gradnorm_target_ortho",
    help="If > 0, GradNorm-style auto-balance the decoder ortho loss weight (same semantics "
         "as --gradnorm_target_sup). Recommended: 0.05–0.2. Overrides --decoder_ortho_loss_weight.",
    type=float,
    default=0.0,
)
@click.option(
    "--gradnorm_every",
    help="Update GradNorm effective weights every N batches (cheaper than every step). "
         "Default 50 → ~6% overhead.",
    type=int,
    default=50,
)
@click.option(
    "--gradnorm_ema",
    help="EMA decay for smoothing effective weight updates: w_new = ema*w_old + (1-ema)*target. "
         "Default 0.9 (slow, stable).",
    type=float,
    default=0.9,
)
@click.option(
    "--gradnorm_min_weight",
    help="Lower clamp on GradNorm-tuned effective weights (default 1e-3).",
    type=float,
    default=1e-3,
)
@click.option(
    "--gradnorm_max_weight",
    help="Upper clamp on GradNorm-tuned effective weights (default 1e6). Prevents blowups "
         "when an aux loss bottoms out (e.g., ortho near its geometric floor).",
    type=float,
    default=1e6,
)
def main(
    args: str,
    cache_dir: str,
    checkpoint_dir: str,
    config_path: str,
    data_dirs: list[str],
    data_shuffling_seed: int,
    model_names: list[str],
    num_epochs: int,
    output_feature_weight: float,
    model_string: str,
    save_dir: str,
    seed: int,
    spill_dir: str,
    temp_dir: str,
    workers: int,
    only_probs: bool,
    use_delta_prob: bool,
    use_delta_logprob: bool,
    decoder_ortho_loss_weight: float,
    use_freq_weighting: bool,
    freq_weight_smoothing: float,
    freq_weight_power: float,
    normalize_per_part: bool,
    output_dim_loss_weight: str,
    checkpoint_weight_scheme: str,
    variance_filter_top_pct: float,
    variance_filter_mode: str,
    cluster_strat_k: int,
    cluster_strat_per_cluster: int,
    cluster_strat_mode: str,
    superimpose_loss_weight: float,
    gradnorm_target_sup: float,
    gradnorm_target_ortho: float,
    gradnorm_every: int,
    gradnorm_ema: float,
    gradnorm_min_weight: float,
    gradnorm_max_weight: float,
):
    logger = get_logger()
    continue_from_checkpoint = False

    # if a path to an args file is provided, load the args
    train_word_id_subset = None
    if args is not None:
        with open(args, "r") as f:
            args_dict = json.load(f)
        train_word_id_subset = args_dict.get("train_word_id_subset", train_word_id_subset)
        cache_dir = args_dict.get("cache_dir", cache_dir)
        checkpoint_dir = args_dict.get("checkpoint_dir", checkpoint_dir)
        data_dirs = args_dict.get("train_data_dirs", data_dirs)
        model_names = args_dict.get("model_names", model_names)
        num_epochs = args_dict.get("num_epochs", num_epochs)
        output_feature_weight = args_dict.get("output_feature_weight", output_feature_weight)
        save_dir = args_dict.get("train_save_dir", save_dir)
        temp_dir = args_dict.get("temp_dir", temp_dir)
        workers = args_dict.get("workers", workers)
        only_probs = args_dict.get("only_probs", only_probs)
        use_delta_prob = args_dict.get("use_delta_prob", use_delta_prob)
        use_delta_logprob = args_dict.get("use_delta_logprob", use_delta_logprob)
        decoder_ortho_loss_weight = args_dict.get("decoder_ortho_loss_weight", decoder_ortho_loss_weight)
        use_freq_weighting = args_dict.get("use_freq_weighting", use_freq_weighting)
        freq_weight_smoothing = args_dict.get("freq_weight_smoothing", freq_weight_smoothing)
        freq_weight_power = args_dict.get("freq_weight_power", freq_weight_power)
        normalize_per_part = args_dict.get("normalize_per_part", normalize_per_part)
        output_dim_loss_weight = args_dict.get("output_dim_loss_weight", output_dim_loss_weight)
        checkpoint_weight_scheme = args_dict.get("checkpoint_weight_scheme", checkpoint_weight_scheme)
        variance_filter_top_pct = args_dict.get("variance_filter_top_pct", variance_filter_top_pct)
        variance_filter_mode = args_dict.get("variance_filter_mode", variance_filter_mode)
        cluster_strat_k = args_dict.get("cluster_strat_k", cluster_strat_k)
        cluster_strat_per_cluster = args_dict.get("cluster_strat_per_cluster", cluster_strat_per_cluster)
        cluster_strat_mode = args_dict.get("cluster_strat_mode", cluster_strat_mode)
        superimpose_loss_weight = args_dict.get("superimpose_loss_weight", superimpose_loss_weight)
        gradnorm_target_sup = args_dict.get("gradnorm_target_sup", gradnorm_target_sup)
        gradnorm_target_ortho = args_dict.get("gradnorm_target_ortho", gradnorm_target_ortho)
        gradnorm_every = args_dict.get("gradnorm_every", gradnorm_every)
        gradnorm_ema = args_dict.get("gradnorm_ema", gradnorm_ema)
        gradnorm_min_weight = args_dict.get("gradnorm_min_weight", gradnorm_min_weight)
        gradnorm_max_weight = args_dict.get("gradnorm_max_weight", gradnorm_max_weight)

    # if data_dirs are provided, sort them in natural order
    if len(data_dirs) > 1:    
        data_dirs = natsorted(data_dirs)
    
    cfg = get_config(config_path, output_feature_weight, seed)
    if use_delta_prob and use_delta_logprob:
        raise ValueError("use_delta_prob and use_delta_logprob are mutually exclusive")
    sae_name_prefix = f"{model_string}_seed={seed}_ofw={output_feature_weight}"
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
    if train_word_id_subset:
        subset_tag = os.path.splitext(os.path.basename(train_word_id_subset))[0]
        sae_name_prefix = f"{sae_name_prefix}_subset={subset_tag}"
    if superimpose_loss_weight and float(superimpose_loss_weight) > 0:
        sae_name_prefix = f"{sae_name_prefix}_sup={superimpose_loss_weight}"
    if gradnorm_target_sup and float(gradnorm_target_sup) > 0:
        sae_name_prefix = f"{sae_name_prefix}_gnsup={gradnorm_target_sup}"
    if gradnorm_target_ortho and float(gradnorm_target_ortho) > 0:
        sae_name_prefix = f"{sae_name_prefix}_gnortho={gradnorm_target_ortho}"
    cfg["use_delta_prob"] = use_delta_prob
    cfg["use_delta_logprob"] = use_delta_logprob
    cfg["decoder_ortho_loss_weight"] = decoder_ortho_loss_weight
    cfg["normalize_per_part"] = normalize_per_part
    cfg["variance_filter_top_pct"] = variance_filter_top_pct
    cfg["variance_filter_mode"] = variance_filter_mode
    cfg["cluster_strat_k"] = cluster_strat_k
    cfg["cluster_strat_per_cluster"] = cluster_strat_per_cluster
    cfg["cluster_strat_mode"] = cluster_strat_mode
    cfg["train_word_id_subset"] = train_word_id_subset
    cfg["superimpose_loss_weight"] = float(superimpose_loss_weight) if superimpose_loss_weight else 0.0
    cfg["gradnorm_target_sup"] = float(gradnorm_target_sup) if gradnorm_target_sup else 0.0
    cfg["gradnorm_target_ortho"] = float(gradnorm_target_ortho) if gradnorm_target_ortho else 0.0
    cfg["gradnorm_every"] = int(gradnorm_every) if gradnorm_every else 50
    cfg["gradnorm_ema"] = float(gradnorm_ema) if gradnorm_ema is not None else 0.9
    cfg["gradnorm_min_weight"] = float(gradnorm_min_weight) if gradnorm_min_weight is not None else 1e-3
    cfg["gradnorm_max_weight"] = float(gradnorm_max_weight) if gradnorm_max_weight is not None else 1e6
    # Normalize the loss-weight arg: None / "auto" / float
    if output_dim_loss_weight is None or (isinstance(output_dim_loss_weight, str) and output_dim_loss_weight.lower() in ("none", "")):
        parsed_odlw = None
    elif isinstance(output_dim_loss_weight, str) and output_dim_loss_weight.lower() == "auto":
        parsed_odlw = "auto"
    else:
        parsed_odlw = float(output_dim_loss_weight)
    cfg["output_dim_loss_weight"] = parsed_odlw
    cfg["checkpoint_weight_scheme"] = checkpoint_weight_scheme or "uniform"
    # Record embedding/output split so the encoder can build per-dim loss weights
    embedding_dim = cfg.get("embedding_dim", 768)
    is_delta = use_delta_prob or use_delta_logprob
    n_output_features = len(model_names) - 1 if is_delta else len(model_names)
    cfg["embedding_dim"] = embedding_dim
    cfg["output_feature_dim"] = n_output_features
    cfg["model_names"] = model_names
    sae_model_name = get_sae_name(cfg, sae_name_prefix)
    cfg["name"] = sae_model_name
    save_dir = f"{save_dir}/{sae_model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg["train_save_dir"] = save_dir
    cfg["data_shuffling_seed"] = data_shuffling_seed
    
    pprint.pprint(cfg)
    logger.info(cfg)
    
    if use_freq_weighting:
        logger.info(f"Frequency weighting ENABLED: smoothing={freq_weight_smoothing}, power={freq_weight_power}")
        print(f"Frequency weighting ENABLED: smoothing={freq_weight_smoothing}, power={freq_weight_power}", flush=True)
    else:
        logger.info("Frequency weighting DISABLED")
    
    # first check if model config exists in the save directory
    # if it does, then we can load the model and continue training
    if os.path.exists(f"{save_dir}/config.json"):
        with open(f"{save_dir}/config.json", "r") as f:
            cfg = json.load(f)
        continue_from_checkpoint = True
        if checkpoint_dir is not None:
            model_checkpoint_dir = f"{checkpoint_dir}/{cfg['name']}"
        else:
            model_checkpoint_dir = f"{save_dir}/checkpoints"
        data_dirs = cfg["train_data_dirs"]
        checkpoint_data_dirs = [f"{model_checkpoint_dir}/{os.path.basename(data_dir)}" for data_dir in data_dirs]
        checkpoint_data_complete = [check_data_training_complete(chk_dir) for chk_dir in checkpoint_data_dirs]
        # if all data directories have been trained on, then we can skip training and exit
        if all(checkpoint_data_complete):
            logger.info("Training already complete, skipping...")
            print("Training already complete, skipping...", flush=True)
            return
        # otherwise we need to check which data directories were used to train the model and continue from there
        # all cached data directories are trained on in order
        else:
            for i, data_dir in enumerate(data_dirs):
                if not checkpoint_data_complete[i]:
                    cached_data_idx = i
                    break
                else:
                    logger.info(f"Data directory {data_dir} already trained on, skipping...")
    else:
        # create directories
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if checkpoint_dir is not None:
            model_checkpoint_dir = f"{checkpoint_dir}/{cfg['name']}"
        else:
            model_checkpoint_dir = f"{save_dir}/checkpoints"
        if not os.path.exists(model_checkpoint_dir):
            os.makedirs(model_checkpoint_dir)
        cfg["model_checkpoint_dir"] = model_checkpoint_dir

        # update the input dimension of the SAE
        is_delta = use_delta_prob or use_delta_logprob
        n_output_features = len(model_names) - 1 if is_delta else len(model_names)
        cfg["input_dim"] = cfg["input_dim"] + n_output_features

        # start from the first data directory
        cached_data_idx = 0

    # check if the data is already cached
    cached_data_dirs = []
    cached_data_filepaths = []
    # model_string = "_".join(model_names)
    cache_models_dir = os.path.join(cache_dir, model_string)
    if os.path.exists(cache_models_dir):
        data_dir_names = os.listdir(cache_models_dir)
        # sort the data directories in natural order
        data_dir_names = natsorted(data_dir_names)
        for data_name in data_dir_names:
            cache_subdir = f"ofw={output_feature_weight}"
            if use_delta_prob:
                cache_subdir = f"{cache_subdir}_delta"
            elif use_delta_logprob:
                cache_subdir = f"{cache_subdir}_logdelta"
            if normalize_per_part:
                cache_subdir = f"{cache_subdir}_znorm"
            cache_data_dir = os.path.join(cache_models_dir, f"{data_name}/{cache_subdir}")
            data_path = os.path.join(cache_data_dir, "preprocessed_data.dat")
            data_info_path = os.path.join(cache_data_dir, "cached_data_info.json")
            if os.path.exists(data_path) and os.path.exists(data_info_path):
                cached_data_filepaths.append(data_path)
                cached_data_json = json.load(open(data_info_path, "r"))
                cached_data_dirs.append(cached_data_json["data_dir"])

    # if no data is cached, cache the data to the cache_dir
    if len(cached_data_filepaths) == 0:
        assert len(data_dirs) > 0, "Must provide data directories to train on"
        cfg["train_data_dirs"] = data_dirs
        logging.info("Loading data...")
        cached_data_filepaths = load_data(
            workers=workers,
            data_dirs=data_dirs,
            model_names=model_names,
            output_feature_weight=output_feature_weight,
            cache_dir=cache_dir,
            # cfg=cfg,
            spill_dir=spill_dir,
            model_string=model_string,
            use_delta_prob=use_delta_prob,
            use_delta_logprob=use_delta_logprob,
            normalize_per_part=normalize_per_part,
        )
    else:
        cfg["train_data_dirs"] = cached_data_dirs

    all_cached_data_filepaths = cached_data_filepaths
    cached_data_filepaths = cached_data_filepaths[cached_data_idx:]
    checkpoint_path = None
    
    init_wandb(cfg)
    for cached_data_filepath in tqdm(cached_data_filepaths):
        print(cached_data_filepath, flush=True)
        assert os.path.exists(cached_data_filepath), \
            f"Cached data at {cached_data_filepath} must exist"
        cache_dir = os.path.dirname(cached_data_filepath)
        assert os.path.exists(f"{cache_dir}/cached_data_info.json"), \
            f"Cached data info at {cache_dir}/cached_data_info.json must exist"

        if checkpoint_path is None:
            # checkpoints saved by data directory
            model_data_checkpoint_dir = \
                f"{model_checkpoint_dir}/{os.path.basename(os.path.dirname(cache_dir))}"
            if not os.path.exists(model_data_checkpoint_dir):
                os.makedirs(model_data_checkpoint_dir)
            if continue_from_checkpoint:
                checkpoints = os.listdir(model_data_checkpoint_dir)
                if len(checkpoints) == 0:
                    prev_cache_data_filepath = all_cached_data_filepaths[i-1]
                    prev_cache_dir = os.path.dirname(prev_cache_data_filepath)
                    prev_model_data_checkpoint_dir = \
                        f"{model_checkpoint_dir}/{os.path.basename(os.path.dirname(prev_cache_dir))}"
                    print(f"Loading checkpoint from {prev_model_data_checkpoint_dir}...", flush=True)
                    assert os.path.exists(prev_model_data_checkpoint_dir)
                    # get the latest checkpoint from the previous data directory
                    checkpoints = os.listdir(prev_model_data_checkpoint_dir)
                    checkpoints = [checkpoint for checkpoint in checkpoints if ".pt" in checkpoint]
                    steps = [int(checkpoint.split("_")[-1].split(".")[0]) for checkpoint in checkpoints]
                    latest_step = max(steps)
                    checkpoint_path = f"{prev_model_data_checkpoint_dir}/checkpoint_{latest_step}.pt"
                else:
                    checkpoints = [checkpoint for checkpoint in checkpoints if ".pt" in checkpoint]
                    steps = [int(checkpoint.split("_")[-1].split(".")[0]) for checkpoint in checkpoints]
                    latest_step = max(steps)
                    checkpoint_path = f"{model_data_checkpoint_dir}/checkpoint_{latest_step}.pt"
                    continue_from_checkpoint = True
    
        print(f"Loading cached data from {cached_data_filepath}...", flush=True)
        with open(f"{cache_dir}/cached_data_info.json", "r") as f:
            cached_data_info = json.load(f)
        logger.info(f"Using cached file: {cached_data_filepath}")
        dtype = DTYPES[cached_data_info["dtype"]]
        shape = tuple(cached_data_info["shape"])
        # copy the data to a temp file
        tempfile_name = os.path.basename(cached_data_filepath)
        tempfile_path = os.path.join(temp_dir, tempfile_name)
        copy_temp_memmap(cached_data_filepath, tempfile_path)
        logger.info(f"Copied data to temporary path {tempfile_path}")
        # Load into RAM for fast random-access during training
        data = np.array(np.memmap(tempfile_path, dtype=dtype, mode='r', shape=shape))
        
        # Load frequency weights if enabled
        sample_weights = None
        if use_freq_weighting:
            # Compute unigram freq path from cache structure
            # cache_dir is like: /path/to/cache/model_string/data_name/ofw=X
            # unigram_freqs.csv is at: /path/to/cache/model_string/data_name/unigram_freqs.csv
            data_cache_dir = os.path.dirname(cache_dir)  # Go up from ofw=X to data_name
            unigram_freq_path = os.path.join(data_cache_dir, "unigram_freqs.csv")
            if not os.path.exists(unigram_freq_path):
                logger.info(f"unigram_freqs.csv not found, computing at {unigram_freq_path}")
                compute_unigram_freqs(
                    cache_root=data_cache_dir,
                    data_dir=cached_data_info["data_dir"],
                    out_path=unigram_freq_path,
                )

            if os.path.exists(unigram_freq_path):
                logger.info(f"Loading frequency weights from {unigram_freq_path} "
                           f"(smoothing={freq_weight_smoothing}, power={freq_weight_power})")
                sample_weights = load_frequency_weights(
                    cache_dir=data_cache_dir,
                    unigram_freq_path=unigram_freq_path,
                    data_dir=cached_data_info["data_dir"],
                    smoothing=freq_weight_smoothing,
                    power=freq_weight_power,
                )
                logger.info(f"Loaded {len(sample_weights)} sample weights "
                           f"(mean={sample_weights.mean():.3f}, std={sample_weights.std():.3f}, "
                           f"min={sample_weights.min():.3f}, max={sample_weights.max():.3f})")
            else:
                logger.warning(f"Frequency weighting enabled but unigram_freqs.csv not found at {unigram_freq_path}. "
                              f"Run compute_unigram_freqs.py first. Training without frequency weighting for this dataset.")

        # Parent of the ofw=*/ memmap dir, where word_ids.pkl lives. Used by the
        # optional train_word_id_subset filter to align rows to word_ids.
        cfg["_word_ids_cache_dir"] = os.path.dirname(cache_dir)

        final_checkpoint_path, encoder = train(
            data=data,
            num_epochs=num_epochs,
            logger=logger,
            save_dir=save_dir,
            model_checkpoint_dir=model_data_checkpoint_dir,
            continue_from_checkpoint=continue_from_checkpoint,
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            sample_weights=sample_weights,
        )

        # load from checkpoint if we are training on multiple data directories
        continue_from_checkpoint = True
        checkpoint_path = final_checkpoint_path
        print(f"Training on {cached_data_filepath} complete, cleanup temporary path {tempfile_path}...", flush=True)
        cleanup_temp_memmap(data, tempfile_path)
        # create a flag file to show that training has completed on this data directory
        with open(f"{model_data_checkpoint_dir}/training_complete", "w") as f:
            f.write("training complete")

    logger.info("Saving final model...")
    torch.save(encoder.state_dict(), f"{save_dir}/sae.pt")
    logger.info(f"Model saved to {save_dir}/sae.pt")
    print(f"Training complete, model saved to {save_dir}/sae.pt", flush=True)
    return
    

if __name__ == "__main__":
    main()
