"""Precompute a 2D (min_r x max_l1) grid of EXCLUSIVE worst-passing tokens per family, for
an interactive slider visualization. "Exclusive" means: a token counts for family i only if
it clears the filter for i and NO other family (see predict_family_from_embedding_highconf.py
for why this matters — L1-nearest assignment alone doesn't guarantee that).

For each (min_r, max_l1) grid point and each family: pool size (how many tokens are
exclusive members under that filter) and the single WORST-passing token's actual curve
(lowest r among exclusive members) — so a slider can show, at any strictness, both how much
data survives and how bad the boundary case still looks.

Usage:
  python dump_exclusive_slider_grid.py --out slider_grid.json
"""
import argparse, json, pickle, time
from pathlib import Path

import numpy as np

from task_shape_content_clusters import CACHE_DIR, open_memmap
from build_family_subset import build_families
from predict_family_from_embedding_highconf import compute_r_dist_all, FAM_LABELS

WORD_IDS = CACHE_DIR / "word_ids.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-min", type=float, default=0.30)
    ap.add_argument("--r-max", type=float, default=0.90)
    ap.add_argument("--r-step", type=float, default=0.05)
    ap.add_argument("--l1-min", type=float, default=0.30)
    ap.add_argument("--l1-max", type=float, default=0.90)
    ap.add_argument("--l1-step", type=float, default=0.05)
    ap.add_argument("--exclude-family", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="slider_grid.json")
    a = ap.parse_args()

    fam_names, F, members = build_families(0.45)
    nfam = F.shape[0]
    print(f"{nfam} families: " + ", ".join(f"[{i}] {FAM_LABELS.get(i,i)}" for i in range(nfam)))

    R, D, valid = compute_r_dist_all(F)  # (N, nfam) each, cached full-corpus scan
    mm, edim, n_out = open_memmap()
    wids = np.asarray(pickle.load(open(WORD_IDS, "rb"))) if WORD_IDS.exists() else None

    r_grid = np.round(np.arange(a.r_min, a.r_max + 1e-9, a.r_step), 3)
    l1_grid = np.round(np.arange(a.l1_min, a.l1_max + 1e-9, a.l1_step), 3)
    fams = [f for f in range(nfam) if f not in set(a.exclude_family)]
    print(f"grid: {len(r_grid)} r-steps x {len(l1_grid)} L1-steps x {len(fams)} families "
          f"= {len(r_grid)*len(l1_grid)*len(fams)} cells")

    t0 = time.time()
    out = {
        "steps": [1, 2, 4, 8, 16, 32, 68, 103, 154, 205, 256],
        "r_grid": r_grid.tolist(),
        "l1_grid": l1_grid.tolist(),
        "families": {},
    }
    for f in fams:
        out["families"][str(f)] = {
            "name": FAM_LABELS.get(f, f),
            "family_curve": F[f].tolist(),
            "grid": [],  # grid[ri][li] = {n, r, dist, token_curve, word_id} or null
        }

    for ri, rt in enumerate(r_grid):
        clears = (R > rt)
        for li, lt in enumerate(l1_grid):
            c = clears & (D < lt) & valid[:, None]
            for f in range(nfam):
                if f in a.exclude_family:
                    continue
                c[:, f] = c[:, f]  # no-op, kept for clarity
            for f in set(a.exclude_family):
                c[:, f] = False
            n_clear = c.sum(1)
            mask = n_clear == 1
            labels = c.argmax(1)
            for f in fams:
                idxs = np.where(mask & (labels == f))[0]
                cell_list = out["families"][str(f)]["grid"]
                if len(cell_list) <= ri:
                    cell_list.append([])
                if len(idxs) == 0:
                    cell_list[ri].append(None)
                    continue
                worst_pos = idxs[np.argmin(R[idxs, f])]
                b = np.asarray(mm[worst_pos, -n_out:], dtype=np.float32)
                mu, sd = b.mean(), b.std()
                z = ((b - mu) / (sd + 1e-9)).tolist()
                cell_list[ri].append({
                    "n": int(len(idxs)),
                    "r": float(R[worst_pos, f]),
                    "dist": float(D[worst_pos, f]),
                    "word_id": str(wids[worst_pos]) if wids is not None else str(worst_pos),
                    "token_curve": z,
                })
        print(f"  r={rt:.2f} done ({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
