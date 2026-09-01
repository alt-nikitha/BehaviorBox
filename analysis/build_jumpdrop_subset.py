"""Build an SAE-training subset balanced across (biggest-jump, biggest-drop) trajectory
shapes, over ALL corpus tokens.

Each token's z-normed 11-checkpoint probability trajectory is reduced to two indices:
  jump = interval of its biggest rise (argmax of consecutive diffs)
  drop = interval of its biggest fall (argmin of consecutive diffs)
giving a 10x10 grid (90 valid cells, jump != drop). We then sample uniformly across
cells, so the SAE allocates capacity evenly across shape families instead of the
early-saturation-dominated default.

NOTE (see session findings): this balances the *trajectory-shape* axis, which is
decoupled from semantic content — good for an unbiased shape-covering SAE / the
timing-match premise, but it does NOT concentrate task-relevant content. For content
targeting, stratify on the embedding block instead (see cluster_stratify in train_sae).

Stage 1 (compute jump/drop for every row) reads the 57GB memmap once (~1 min) and
caches the per-token arrays; reruns with a different --per-cell skip the read.

Usage:
  python build_jumpdrop_subset.py                       # max-balanced (min-cell per cell)
  python build_jumpdrop_subset.py --per-cell 10472      # ~942k total
  python build_jumpdrop_subset.py --dist-only           # just print the distribution
"""
import argparse, json, pickle, time, numpy as np

CACHE_DIR = "/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen"
MEMMAP_INFO = CACHE_DIR + "/ofw=0.5_pznorm=0.01/cached_data_info.json"
WORD_IDS = CACHE_DIR + "/word_ids.pkl"
ARR_CACHE = CACHE_DIR + "/jumpdrop_full.npz"          # cached per-token jump/drop/valid
STEPS = [1, 2, 4, 8, 16, 32, 68, 103, 154, 205, 256]
ARR = [STEPS[i + 1] for i in range(10)]               # arrival checkpoint (k) per interval


def compute_jumpdrop(chunk=3_000_000):
    """Return (jump, drop, valid) int8/bool arrays over all memmap rows; cache to disk."""
    import os
    if os.path.exists(ARR_CACHE):
        z = np.load(ARR_CACHE)
        print(f"loaded cached jump/drop from {ARR_CACHE}")
        return z["jump"], z["drop"], z["valid"]
    info = json.load(open(MEMMAP_INFO))
    N = info["shape"][0]
    mm = np.memmap(info["filename"], dtype=np.float16, mode="r", shape=tuple(info["shape"]))
    jump = np.empty(N, np.int8); drop = np.empty(N, np.int8); valid = np.zeros(N, bool)
    t = time.time()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = np.asarray(mm[s:e, -11:], dtype=np.float32)          # prob block = last 11 cols
        mu = b.mean(1, keepdims=True); sd = b.std(1, keepdims=True)
        z = (b - mu) / (sd + 1e-9); d = np.diff(z, axis=1)
        jump[s:e] = d.argmax(1); drop[s:e] = d.argmin(1); valid[s:e] = sd[:, 0] > 1e-6
        print(f"  {e:,}/{N:,} ({time.time()-t:.0f}s)", flush=True)
    np.savez(ARR_CACHE, jump=jump, drop=drop, valid=valid)
    print(f"cached -> {ARR_CACHE}  valid={valid.sum():,}/{N:,}")
    return jump, drop, valid


def print_distribution(jump, drop, valid):
    cell = jump[valid].astype(np.int64) * 10 + drop[valid].astype(np.int64)
    M = np.bincount(cell, minlength=100).reshape(10, 10)
    hdr = "jump\\drop|" + "".join(f"{a:>8}" for a in ARR) + "  |   TOTAL"
    print("\n" + hdr); print("-" * len(hdr))
    for i in range(10):
        print(f"{ARR[i]:>8}|" + "".join(f"{M[i,k]:>8}" for k in range(10)) + f"  |{M[i].sum():>8}")
    print("-" * len(hdr))
    print(f"{'TOTAL':>8}|" + "".join(f"{M[:,k].sum():>8}" for k in range(10)) + f"  |{M.sum():>8}")
    off = M.copy(); np.fill_diagonal(off, 0)
    print(f"\nvalid={M.sum():,}  min cell={off[off>0].min():,}  "
          f"median={int(np.median(off[off>0])):,}  max={off.max():,}")
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=None,
                    help="tokens per (jump,drop) cell. Default: min cell size (max-balanced).")
    ap.add_argument("--out", default="/home/nsrikant/BehaviorBoxNew/analysis/sae_sample_by_jumpdrop_uniform.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dist-only", action="store_true", help="print distribution and exit.")
    a = ap.parse_args()

    jump, drop, valid = compute_jumpdrop()
    M = print_distribution(jump, drop, valid)
    if a.dist_only:
        return

    off = M.copy(); np.fill_diagonal(off, 0)
    per = a.per_cell or int(off[off > 0].min())
    print(f"\nper_cell = {per:,}  -> total {per*90:,}")
    cellid = jump.astype(np.int32) * 10 + drop.astype(np.int32); cellid[~valid] = -1
    wids = np.asarray(pickle.load(open(WORD_IDS, "rb")))
    rng = np.random.default_rng(a.seed)
    rows, jk, dk = [], [], []
    for j in range(10):
        for d in range(10):
            if j == d:
                continue
            idx = np.where(cellid == j * 10 + d)[0]
            take = min(len(idx), per)
            sel = rng.choice(idx, take, replace=False)
            rows.append(sel); jk.append(np.full(take, ARR[j])); dk.append(np.full(take, ARR[d]))
    rows = np.concatenate(rows); jk = np.concatenate(jk); dk = np.concatenate(dk)
    perm = rng.permutation(len(rows)); rows, jk, dk = rows[perm], jk[perm], dk[perm]
    sw = wids[rows]
    with open(a.out, "w") as f:
        for w, j, d in zip(sw.tolist(), jk.tolist(), dk.tolist()):
            f.write(json.dumps({"word_id": str(w), "jump": int(j), "drop": int(d)}) + "\n")
    print(f"WROTE {len(rows):,} tokens -> {a.out}")


if __name__ == "__main__":
    main()
