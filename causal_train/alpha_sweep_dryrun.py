"""
Single-GPU dry-run sanity check across tasks × alphas.

For a fixed batch sampled once, runs the same forward+backward used by
train_causal.py and reports weighted loss, unweighted baseline loss,
one-step grad norm, and mask statistics. Masks are computed on-the-fly
from {task}_bestc.npz — no files are written or overwritten.

Usage:
    python alpha_sweep_dryrun.py \
        --data_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/expweight_tasks_4 \
        --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/task_mapped_results \
        --resume_ckpt LLM360/Amber --resume_revision ckpt_300 \
        --alphas 1,2,4,8 --tau 0.6 --batch_size 2 \
        --out /home/nsrikant/BehaviorBoxNew/causal_train/alpha_sweep_results.json
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Dir with shared/{token_ids,attention_mask,global_word_ids,meta}.")
    p.add_argument("--task_mapped_dir", required=True,
                   help="Dir with {task}_bestc.npz files.")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated tasks. Default: auto-detect from task_mapped_dir.")
    p.add_argument("--resume_ckpt", default="LLM360/Amber")
    p.add_argument("--resume_revision", default=None)
    p.add_argument("--use_safetensors", default=None,
                   help="true/false; default auto (false for amber).")
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--alphas", default="1,2,4,8")
    p.add_argument("--condition", default="pos", choices=["pos", "neg"])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Enable gradient checkpointing to fit on one GPU.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def build_corr_flat(npz_path, n_words_total):
    data = np.load(npz_path)
    w_idx = data["w_idx"]
    best_c = data["best_c"].astype(np.float32, copy=False)
    corr = np.zeros(n_words_total, dtype=np.float32)
    corr[w_idx] = best_c
    return corr


def compute_mask(gwid, attn, corr, tau, alpha, condition):
    idx = np.maximum(gwid, 0)
    c = corr[idx]
    c[gwid < 0] = 0.0
    if condition == "pos":
        w = np.exp(alpha * np.maximum(0.0, c - tau), dtype=np.float32)
    else:
        w = np.exp(alpha * np.maximum(0.0, -tau - c), dtype=np.float32)
    return w * attn.astype(np.float32)


def grad_norm(model):
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.detach().float().norm(2).item() ** 2
    return s ** 0.5


def auto_tasks(task_mapped_dir):
    paths = sorted(glob.glob(os.path.join(task_mapped_dir, "*_bestc.npz")))
    return [os.path.basename(p)[: -len("_bestc.npz")] for p in paths]


def main():
    args = parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else auto_tasks(args.task_mapped_dir))
    print(f"Tasks ({len(tasks)}): {tasks}")
    print(f"Alphas: {alphas}  tau: {args.tau}  condition: {args.condition}")

    shared = os.path.join(args.data_dir, "shared")
    with open(os.path.join(shared, "meta.json")) as f:
        meta = json.load(f)
    n_words_total = meta["n_words_total"]

    token_ids_all = np.load(os.path.join(shared, "token_ids.npy"), mmap_mode="r")
    attn_all = np.load(os.path.join(shared, "attention_mask.npy"), mmap_mode="r")
    gwid_all = np.load(os.path.join(shared, "global_word_ids.npy"), mmap_mode="r")
    n_docs = token_ids_all.shape[0]

    rng = np.random.default_rng(args.seed)
    # Prefer docs with decent attended length.
    batch_idx = rng.choice(n_docs, size=args.batch_size, replace=False)
    tok_np = np.asarray(token_ids_all[batch_idx])
    attn_np = np.asarray(attn_all[batch_idx])
    gwid_np = np.asarray(gwid_all[batch_idx])
    print(f"Batch: docs {batch_idx.tolist()}  "
          f"attended tokens/doc: {attn_np.sum(axis=1).tolist()}")

    if args.use_safetensors is None:
        use_safetensors = "amber" not in args.resume_ckpt.lower()
    else:
        use_safetensors = args.use_safetensors.lower() == "true"

    print(f"Loading {args.resume_ckpt}"
          + (f"@{args.resume_revision}" if args.resume_revision else ""))
    t0 = time.time()
    AutoTokenizer.from_pretrained(args.resume_ckpt, revision=args.resume_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.resume_ckpt, revision=args.resume_revision,
        use_safetensors=use_safetensors,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    tok = torch.from_numpy(tok_np).long().cuda()
    input_ids = tok[:, :-1]
    labels = tok[:, 1:]
    attn_t = torch.from_numpy(attn_np).cuda()
    base_mask = attn_t[:, 1:].float()

    # Baseline: unweighted cross-entropy on attended label tokens.
    with torch.no_grad():
        logits = model(input_ids).logits
        per_tok_base = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape).float()
        n_active = (base_mask > 0).sum().clamp_min(1)
        baseline_loss = (per_tok_base * base_mask).sum() / n_active
    del logits, per_tok_base
    torch.cuda.empty_cache()
    print(f"baseline_loss (attended, unweighted): {baseline_loss.item():.4f}")

    results = []
    for task in tasks:
        npz = os.path.join(args.task_mapped_dir, f"{task}_bestc.npz")
        if not os.path.exists(npz):
            print(f"[{task}] SKIP: {npz} missing")
            continue
        corr = build_corr_flat(npz, n_words_total)

        for alpha in alphas:
            w_np = compute_mask(gwid_np, attn_np, corr, args.tau, alpha, args.condition)
            w_t = torch.from_numpy(w_np).cuda()
            loss_mask = w_t[:, 1:]

            pos_boosted = int(((w_t > 1.0) & (attn_t > 0)).sum().item())
            mask_max = float(loss_mask.max().item())
            s1 = float(loss_mask.sum().item())
            s2 = float((loss_mask * loss_mask).sum().item())
            ess = (s1 * s1 / s2) if s2 > 0 else 0.0
            n_active_w = int((loss_mask > 0).sum().item())

            model.zero_grad(set_to_none=True)
            logits = model(input_ids).logits
            per_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            masked = per_tok * loss_mask
            n_act_t = (loss_mask > 0).sum().clamp_min(1)
            loss = masked.sum() / n_act_t
            loss.backward()
            gn = grad_norm(model)
            model.zero_grad(set_to_none=True)
            del logits, per_tok, masked, loss_mask, w_t
            torch.cuda.empty_cache()

            row = {
                "task": task,
                "alpha": alpha,
                "weighted_loss": float(loss.item()),
                "baseline_loss": float(baseline_loss.item()),
                "ratio": float(loss.item() / max(baseline_loss.item(), 1e-9)),
                "grad_norm": gn,
                "mask_max": mask_max,
                "pos_boosted_tokens": pos_boosted,
                "n_active_weighted": n_active_w,
                "ess": ess,
            }
            results.append(row)
            print(f"[{task:<22} alpha={alpha:>4g}] "
                  f"loss={row['weighted_loss']:.4f} "
                  f"(base {baseline_loss.item():.4f}, {row['ratio']:.2f}x) "
                  f"gn={gn:7.2f} max_w={mask_max:6.2f} "
                  f"boost={pos_boosted:>6} ess={ess:8.0f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "config": {
                    "data_dir": args.data_dir,
                    "resume_ckpt": args.resume_ckpt,
                    "resume_revision": args.resume_revision,
                    "tau": args.tau,
                    "alphas": alphas,
                    "condition": args.condition,
                    "batch_size": args.batch_size,
                    "batch_idx": batch_idx.tolist(),
                    "seed": args.seed,
                },
                "baseline_loss": float(baseline_loss.item()),
                "results": results,
            }, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
