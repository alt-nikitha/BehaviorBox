"""
Train with exponential per-token weights produced by prepare_training_data.py.

Expected layout under `base_data_dir`:
    shared/
        token_ids.npy          (n_docs, L)  int32
        attention_mask.npy     (n_docs, L)  uint8
    {task}/
        pos_mask.npy           (n_docs, L)  float32
        neg_mask.npy           (n_docs, L)  float32

For the unweighted baseline, pass task='uniform' — loss_weights falls back to
shared/attention_mask.npy (active=1, pad=0). `condition` is ignored in that case.
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


def build_lr_fn(lr, end_lr, warmup_steps, total_steps):
    if total_steps is None:
        return lambda step: lr
    return get_cosine_lr_decay_fn(total_steps, warmup_steps, lr, end_lr)


@torch.no_grad()
def get_grad_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.detach().data.norm(2).item() ** 2
    return s ** 0.5


def save_model_only(fabric, tokenizer, model, save_dir):
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
               token_ids, attn_mask, loss_weights, indices,
               per_device_batch_size, accumulate_grad_batches,
               resume_step):
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

        tok_batch = torch.from_numpy(np.ascontiguousarray(token_ids[batch_idx])).to(
            fabric.device, dtype=torch.long, non_blocking=True)
        attn_batch = torch.from_numpy(np.ascontiguousarray(attn_mask[batch_idx])).to(
            fabric.device, dtype=torch.float32, non_blocking=True)
        w_batch = torch.from_numpy(np.ascontiguousarray(loss_weights[batch_idx])).to(
            fabric.device, dtype=torch.float32, non_blocking=True)

        input_ids = tok_batch[:, :-1]
        labels = tok_batch[:, 1:]
        loss_mask = w_batch[:, 1:] * attn_batch[:, 1:]
        attention_mask = attn_batch[:, :-1]

        with fabric.no_backward_sync(model, enabled=is_accum):
            logits = model(input_ids, attention_mask=attention_mask).logits
            per_tok = torch.nn.functional.cross_entropy(
                logits.reshape((-1, logits.size(-1))),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            masked = per_tok * loss_mask
            w_sum = loss_mask.sum()
            loss = masked.sum() / w_sum if w_sum > 0 else masked.sum()
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accum:
            gn = get_grad_norm(model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        if fabric.global_rank == 0:
            log = {"loss": loss.item(), "lr": lr, "step": step,
                   "tok/s/gpu": int(input_ids.numel() / (time.time() - t0))}
            if not is_accum:
                log["grad_norm"] = gn
            bar.set_postfix(log)


def main(base_data_dir,
         task,
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
         use_safetensors=None):
    """
    Args:
        base_data_dir: directory with shared/ and per-task subfolders
                       (e.g. causal_data/olmo/expweight_tasks).
        task: task subfolder (e.g. 'gsm8k') or 'uniform' for the unweighted baseline.
        condition: 'pos' or 'neg' — ignored when task == 'uniform'.
    """
    shared_dir = os.path.join(base_data_dir, "shared")
    assert os.path.isdir(shared_dir), f"shared/ not found at {shared_dir}"
    is_uniform = (task == "uniform")
    if not is_uniform:
        task_dir = os.path.join(base_data_dir, task)
        assert os.path.isdir(task_dir), f"Task dir not found at {task_dir}"
        assert condition in ("pos", "neg"), "condition must be pos|neg"

    if workdir is None:
        sub = "uniform" if is_uniform else f"{condition}_up"
        workdir = os.path.join(base_data_dir, "models", task, sub)

    if use_safetensors is None:
        use_safetensors = "amber" not in resume_ckpt.lower()

    tag = "uniform" if is_uniform else f"{task}/{condition}"
    print(f"[{tag}] Loading {resume_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(resume_ckpt, revision=resume_revision)
    model = AutoModelForCausalLM.from_pretrained(
        resume_ckpt, revision=resume_revision,
        use_safetensors=use_safetensors,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    layer_cls = detect_decoder_layer_class(model)

    fabric = L.Fabric(
        accelerator="cuda", devices=n_devices, precision="bf16-mixed",
        strategy=FSDPStrategy(
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls={layer_cls}),
            activation_checkpointing_policy={layer_cls}))
    fabric.launch()
    fabric.seed_everything(RANDOM_SEED)

    if fabric.global_rank == 0:
        os.makedirs(workdir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2))
    model, optimizer = fabric.setup(model, optimizer)

    token_ids = np.load(os.path.join(shared_dir, "token_ids.npy"), mmap_mode="r")
    attn_mask = np.load(os.path.join(shared_dir, "attention_mask.npy"), mmap_mode="r")
    if is_uniform:
        loss_weights = attn_mask
        weights_desc = "attention_mask (uniform baseline)"
    else:
        mask_file = "pos_mask.npy" if condition == "pos" else "neg_mask.npy"
        loss_weights = np.load(os.path.join(task_dir, mask_file), mmap_mode="r")
        weights_desc = mask_file

    n_docs = token_ids.shape[0]
    assert loss_weights.shape[0] == n_docs, \
        f"weights have {loss_weights.shape[0]} docs; shared has {n_docs}"
    if fabric.global_rank == 0:
        print(f"Task: {task} | docs: {n_docs} | weights: {weights_desc}")

    lr_fn = build_lr_fn(lr, end_lr, warmup_steps, total_steps)
    global_micro_batch_size = batch_size * fabric.world_size

    for epoch in range(n_epochs):
        indices = shard_indices(
            n=n_docs, rank=fabric.global_rank, world_size=fabric.world_size,
            micro_batch_size=global_micro_batch_size, shuffle=True,
            seed=RANDOM_SEED + epoch)

        train_loop(
            fabric=fabric, tokenizer=tokenizer, model=model, optimizer=optimizer,
            lr_schedule_fn=lr_fn, token_ids=token_ids, attn_mask=attn_mask,
            loss_weights=loss_weights, indices=indices,
            per_device_batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            resume_step=resume_step)

    save_model_only(fabric, tokenizer, model, workdir)
    if fabric.global_rank == 0:
        print(f"Saved model to {workdir}")


if __name__ == "__main__":
    fire.Fire(main)
