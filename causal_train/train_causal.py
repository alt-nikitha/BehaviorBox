"""
Resume Amber training from a checkpoint using per-token weighted loss,
reading the numpy arrays produced by prepare_training_data.py.

Expects in --data_dir:
    token_ids.npy        int32   (N, L)
    pos_mask.npy         float32 (N, L)
    neg_mask.npy         float32 (N, L)

Pick one condition with --condition {pos,neg}. Saves only the final HF
model (no optimizer / fabric state) so the directory is directly usable
by lm-eval-harness.

Model architecture (Llama/OLMo/OLMo2/...) is auto-detected from the
checkpoint. LR schedule defaults to constant; pass --total_steps for cosine.

Usage (Amber):
    python train_causal.py \
        --data_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/symmetric_tasks \
        --condition pos \
        --resume_ckpt LLM360/Amber --resume_revision ckpt_300 \
        --workdir /data/user_data/nsrikant/bbox_data/training_checkpoints/symmetric_tasks/amber/pos_up

Usage (OLMo):
    python train_causal.py \
        --data_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/symmetric_tasks \
        --condition pos \
        --resume_ckpt allenai/OLMo-2-1124-7B \
        --workdir /data/user_data/nsrikant/bbox_data/training_checkpoints/symmetric_tasks/olmo/pos_up

Flip --condition neg for the negative-correlation upweighting run.
"""

import math
import os
import time
from datetime import datetime
from functools import partial

import fire
import lightning as L
import numpy as np
import torch
import tqdm
import wandb
from lightning.fabric.strategies import FSDPStrategy
from pytz import timezone
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer


TIMEZONE = timezone("EST")
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]

GRAD_NORM_CLIP = 1.0
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
RANDOM_SEED = 11111


def detect_decoder_layer_class(model):
    """Find the transformer decoder layer class for FSDP wrapping.
    Works for Llama (Amber), OLMo, OLMo2, Mistral, etc."""
    for attr in ("model", "transformer", "gpt_neox"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        layers = getattr(base, "layers", None) or getattr(base, "h", None)
        if layers is not None and len(layers) > 0:
            return type(layers[0])
    raise RuntimeError("Could not locate decoder layers on model")


def get_cosine_lr_decay_fn(total_steps, warmup_steps, learning_rate, end_learning_rate):
    def fn(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        if step > total_steps:
            return end_learning_rate
        ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return end_learning_rate + coeff * (learning_rate - end_learning_rate)
    return fn


def get_grad_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.detach().data.norm(2).item() ** 2
    return s ** 0.5


def save_model_only(fabric, tokenizer, model, save_dir):
    """Save HF model + tokenizer only. No optimizer / fabric state."""
    policy = FullStateDictConfig(
        offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
    with FSDP.state_dict_type(
            model, state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=policy):
        state_dict = model._forward_module.state_dict()
    if fabric.global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.module.save_pretrained(
            save_dir, state_dict=state_dict, safe_serialization=False)
    fabric.barrier()


def shard_indices(n, rank, world_size, micro_batch_size, shuffle, seed):
    idxes = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    n_trim = (n // micro_batch_size) * micro_batch_size
    return idxes[rank:n_trim:world_size]


def train_loop(fabric, tokenizer, model, optimizer, lr_schedule_fn,
               token_ids, loss_weights, indices,
               per_device_batch_size, accumulate_grad_batches,
               resume_step, run_wandb, workdir):
    step = resume_step
    bar = tqdm.trange(
        0, len(indices), per_device_batch_size,
        desc=f"train (bs={per_device_batch_size}, accum={accumulate_grad_batches})",
        disable=(fabric.global_rank != 0))

    for i in bar:
        t0 = time.time()
        batch_idx = indices[i:i + per_device_batch_size]
        if len(batch_idx) < per_device_batch_size:
            break

        lr = lr_schedule_fn(step)
        step += 1
        for g in optimizer.param_groups:
            g["lr"] = lr
        is_accum = (step % accumulate_grad_batches != 0)

        tok_batch = torch.from_numpy(token_ids[batch_idx]).to(
            fabric.device, dtype=torch.long, non_blocking=True)
        w_batch = torch.from_numpy(loss_weights[batch_idx]).to(
            fabric.device, dtype=torch.float32, non_blocking=True)
        input_ids = tok_batch[:, :-1]
        labels = tok_batch[:, 1:]
        loss_mask = w_batch[:, 1:]

        with fabric.no_backward_sync(model, enabled=is_accum):
            logits = model(input_ids).logits
            per_tok = torch.nn.functional.cross_entropy(
                logits.reshape((-1, logits.size(-1))),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            masked = per_tok * loss_mask
            n_active = (loss_mask > 0).sum()
            loss = masked.sum() / n_active if n_active > 0 else masked.sum()
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accum:
            gn = get_grad_norm(model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {"loss": loss.item(), "lr": lr, "step": step,
               "tok/s/gpu": int(input_ids.numel() / (time.time() - t0))}
        if not is_accum:
            log["grad_norm"] = gn
        if fabric.global_rank == 0:
            bar.set_postfix(log)
            if run_wandb:
                wandb.log(log)


def build_lr_fn(lr, end_lr, warmup_steps, total_steps):
    """Constant LR if total_steps is None; otherwise cosine-with-warmup."""
    if total_steps is None:
        return lambda step: lr
    return get_cosine_lr_decay_fn(total_steps, warmup_steps, lr, end_lr)


def main(data_dir,
         condition="pos",
         resume_ckpt="LLM360/Amber",
         resume_revision=None,
         workdir=None,
         n_devices=4,
         batch_size=10,
         accumulate_grad_batches=1,
         n_epochs=1,
         lr=1e-5,
         end_lr=1e-6,
         warmup_steps=0,
         total_steps=None,
         resume_step=0,
         use_safetensors=None,
         run_wandb=False,
         project_name=None):
    """
    Args:
        data_dir: dir with token_ids.npy / pos_mask.npy / neg_mask.npy
        condition: 'pos' (pos_mask.npy) or 'neg' (neg_mask.npy)
        resume_ckpt: HF id or local path (Amber, OLMo, etc.)
        resume_revision: optional HF revision (branch/tag)
        workdir: output dir for final HF model (no optimizer saved)
        lr/end_lr/warmup_steps/total_steps: LR schedule. If total_steps is
            None, uses constant lr.
        use_safetensors: if None, auto (True for OLMo-style, False for Amber).
    """
    assert condition in ("pos", "neg"), "condition must be pos|neg"
    if workdir is None:
        workdir = os.path.join(data_dir, f"model_{condition}_up")

    # Defer FSDP strategy setup until we know the layer class.
    # First, load model on rank 0 quickly to inspect arch — but we need
    # Fabric first for distributed. Trick: load model class via config first,
    # then instantiate inside fabric setup. Simpler: load on CPU, inspect,
    # then wrap with FSDP.

    # Amber stores weights as .bin; most OLMo variants use safetensors.
    if use_safetensors is None:
        use_safetensors = "amber" not in resume_ckpt.lower()

    print(f"Loading {resume_ckpt}" + (f"@{resume_revision}" if resume_revision else ""))
    tokenizer = AutoTokenizer.from_pretrained(
        resume_ckpt, revision=resume_revision)
    model = AutoModelForCausalLM.from_pretrained(
        resume_ckpt, revision=resume_revision,
        use_safetensors=use_safetensors,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    layer_cls = detect_decoder_layer_class(model)
    print(f"Detected decoder layer class: {layer_cls.__name__}")

    fabric = L.Fabric(
        accelerator="cuda", devices=n_devices, precision="bf16-mixed",
        strategy=FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls}),
            activation_checkpointing_policy={layer_cls},
            cpu_offload=False, limit_all_gathers=True))
    fabric.launch()
    fabric.seed_everything(RANDOM_SEED)

    if project_name is None:
        project_name = "causal_upweight"
    run_name = f"{os.path.basename(data_dir.rstrip('/'))}_{condition}_{DATE}"

    if fabric.global_rank == 0:
        os.makedirs(workdir, exist_ok=True)
        if run_wandb:
            wandb.init(project=project_name, name=run_name)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2), foreach=False)
    model, optimizer = fabric.setup(model, optimizer)
    torch.cuda.empty_cache()

    token_ids = np.load(os.path.join(data_dir, "token_ids.npy"), mmap_mode="r")
    mask_file = "pos_mask.npy" if condition == "pos" else "neg_mask.npy"
    loss_weights = np.load(os.path.join(data_dir, mask_file), mmap_mode="r")
    n_docs = token_ids.shape[0]
    if fabric.global_rank == 0:
        print(f"Data: {n_docs} docs × {token_ids.shape[1]} tokens "
              f"(weights: {mask_file})")

    lr_fn = build_lr_fn(lr, end_lr, warmup_steps, total_steps)
    if fabric.global_rank == 0:
        sched = "constant" if total_steps is None else f"cosine total={total_steps}"
        print(f"LR schedule: {sched}, start_lr={lr}, resume_step={resume_step}")

    global_micro_batch_size = batch_size * fabric.world_size

    for epoch in range(n_epochs):
        indices = shard_indices(
            n=n_docs, rank=fabric.global_rank, world_size=fabric.world_size,
            micro_batch_size=global_micro_batch_size, shuffle=True,
            seed=RANDOM_SEED + epoch)
        train_loop(
            fabric=fabric, tokenizer=tokenizer, model=model, optimizer=optimizer,
            lr_schedule_fn=lr_fn,
            token_ids=token_ids, loss_weights=loss_weights, indices=indices,
            per_device_batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            resume_step=resume_step + epoch * (len(indices) // batch_size),
            run_wandb=run_wandb, workdir=workdir)

    save_model_only(fabric, tokenizer, model, workdir)
    if fabric.global_rank == 0:
        print(f"Saved model (weights only) to {workdir}")


if __name__ == "__main__":
    fire.Fire(main)
