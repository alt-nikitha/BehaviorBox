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
log_eval_loss_every = 20000
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
    
    use_freq_weighting = sample_weights is not None
    decoder_ortho_loss_weight = cfg.get("decoder_ortho_loss_weight", 0.0)
    logger.info(f"Beginning training... (frequency weighting: {use_freq_weighting}, decoder_ortho_loss_weight: {decoder_ortho_loss_weight})")
    encoder.train()
    for epoch in range(num_epochs):
        train_dataloader = DataLoader(data, batch_size, train_indices)
        background_loader = BackgroundDataLoader(train_dataloader)
        eval_dataloader = DataLoader(data, batch_size, eval_indices)
        total_batches = train_dataloader.num_batches * num_epochs
        k_start = 100
        k_end = cfg["topk"] # Your target k (e.g., 25)
        # anneal_until_batch = int(0.3 * total_batches) # Anneal over first 30% of training

        anneal_start_batch = int(0.1 * total_batches) 
        anneal_end_batch = int(0.7 * total_batches)   # Finish at 70% mark
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
            batch = batch.to(cfg["device"])
            # loss, _, penalty, _ = encoder(batch)
            loss, _, _, l2_error_per_sample = encoder(batch, return_l2_error_per_sample=True)
            
            # Apply frequency weighting if available
            if use_freq_weighting:
                batch_indices = train_dataloader.batch_indices[i % train_dataloader.num_batches]
                weights = torch.from_numpy(sample_weights[batch_indices]).to(cfg["device"])
                weighted_l2_loss = (l2_error_per_sample.squeeze() * weights).mean()
                if not cfg["topk"]:
                    acts = encoder.enc(batch - encoder.dec.bias)
                    acts = torch.nn.functional.relu(acts)
                    l1_loss = cfg["l1_coeff"] * (acts.abs().sum(-1).mean())
                    loss = weighted_l2_loss + l1_loss
                else:
                    loss = weighted_l2_loss

            with torch.no_grad():
                mse = l2_error_per_sample.mean()
                var_x = (batch - batch.mean(0, keepdim=True)).pow(2).sum(-1).mean()
                ev = (1.0 - mse / (var_x + 1e-8)).item()

            ortho_loss_val = None
            if decoder_ortho_loss_weight > 0:
                W = encoder.dec.weight
                W_col_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
                sims = W_col_norm.T @ W_col_norm
                K = sims.shape[0]
                off_diag_sq_sum = (sims ** 2).sum() - K
                ortho_loss = off_diag_sq_sum / (K * (K - 1))
                loss = loss + decoder_ortho_loss_weight * ortho_loss
                ortho_loss_val = ortho_loss.item()

            loss_dict = {
                "epoch": epoch,
                "batch": f"{i} / {total_batches}",
                "loss": loss.item(),
                "explained_variance": ev,
                # "penalty": penalty.item()
            }
            if ortho_loss_val is not None:
                loss_dict["decoder_ortho_loss"] = ortho_loss_val
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
                        eval_batch = eval_batch.to(cfg["device"])
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
                        total_x_var_ss += (eval_batch - eval_batch.mean(0, keepdim=True)).pow(2).sum().item()
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
):
    logger = get_logger()
    continue_from_checkpoint = False

    # if a path to an args file is provided, load the args
    if args is not None:
        with open(args, "r") as f:
            args_dict = json.load(f)
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
    cfg["use_delta_prob"] = use_delta_prob
    cfg["use_delta_logprob"] = use_delta_logprob
    cfg["decoder_ortho_loss_weight"] = decoder_ortho_loss_weight
    sae_model_name = get_sae_name(cfg, sae_name_prefix)
    cfg["name"] = sae_model_name
    cfg["model_names"] = model_names
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
