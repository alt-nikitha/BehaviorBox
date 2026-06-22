"""Per-sample probability trajectories from the preprocessed memmap.

The precomputed JSONs only carry per-feature `median_probs`/`std_probs` curves.
Per-sample curves live in the input-features cache:
  ~/.cache/<run>/<part>/<variant>/preprocessed_data.dat   (memmap)
  ~/.cache/<run>/<part>/<variant>/cached_data_info.json
  <data_dir>/input_features/file_to_doc.csv               (doc → row index)

This module maps a sample's `word_id` ("doc_<docid>_<word>" or "<docid>_<word>")
to its memmap row and exposes the last `output_feature_dim` columns of that row
as the sample's per-checkpoint output trajectory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE = Path("/home/nsrikant/.cache/n_only_early_and_late_olmo3/olmo_256000_unseen/ofw=0.8_znorm")


@lru_cache(maxsize=8)
def _load_index(cache_dir: str):
    """Returns (memmap, doc_offset, n_outputs, model_names)."""
    cache = Path(cache_dir)
    info = json.loads((cache / "cached_data_info.json").read_text())
    shape = tuple(info["shape"])
    n_outputs = int(info["output_feature_dim"])
    model_names = list(info["model_names"])
    data_dir = Path(info["data_dir"])

    dtype_str = info.get("dtype", "np16")
    dtype = {"np16": np.float16, "np32": np.float32, "float16": np.float16,
             "float32": np.float32}.get(dtype_str, np.float16)
    mm = np.memmap(info["filename"], dtype=dtype, mode="r", shape=shape)

    # The memmap is built from the *output* features. The index for those is
    # under output_features/<step>/file_to_doc.csv (any step has the same
    # doc order and counts). The input_features csv has different
    # ordering/coverage and won't line up.
    out_dir = data_dir / "output_features"
    step_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    if not step_dirs:
        raise RuntimeError(f"no output_features step dirs under {out_dir}")
    f2d_path = step_dirs[0] / "file_to_doc.csv"
    f2d = pd.read_csv(f2d_path)
    offsets = np.cumsum(f2d["num_words"].values) - f2d["num_words"].values
    doc_offset = dict(zip(f2d["doc_id"].astype(int).tolist(),
                          offsets.astype(np.int64).tolist()))
    return mm, doc_offset, n_outputs, model_names


def _parse_word_id(word_id) -> tuple[int, int] | None:
    """Returns (doc_id, word_idx) or None if unparseable."""
    s = str(word_id)
    parts = s.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def get_sample_curve(word_id, cache_dir=DEFAULT_CACHE):
    """Returns the sample's per-checkpoint output vector (length n_outputs)
    as a float32 numpy array, or None if the word_id can't be resolved."""
    parsed = _parse_word_id(word_id)
    if parsed is None:
        return None
    doc_id, w = parsed
    mm, doc_offset, n_outputs, _ = _load_index(str(cache_dir))
    if doc_id not in doc_offset:
        return None
    row = doc_offset[doc_id] + w
    if row < 0 or row >= mm.shape[0]:
        return None
    return np.asarray(mm[row, -n_outputs:], dtype=np.float32)


def _z(v):
    sd = v.std()
    if sd < 1e-9:
        return None
    return (v - v.mean()) / sd


def sample_distance_to(target_z_curve, word_id, mode="area",
                       cache_dir=DEFAULT_CACHE):
    """Distance between a sample's z-normalized output trajectory and a target
    curve already in z-form. `target_z_curve` is (idxs, z_values) or just a
    1D vector when full alignment is assumed.

    `mode` is "area" (trapezoid of |diff|) or "mse" (mean squared diff).
    Returns None if the sample or target is degenerate / unresolvable."""
    raw = get_sample_curve(word_id, cache_dir=cache_dir)
    if raw is None:
        return None
    sz = _z(raw)
    if sz is None:
        return None

    if isinstance(target_z_curve, tuple) and len(target_z_curve) == 2:
        idxs, tvals = target_z_curve
        if len(idxs) == 0:
            return None
        # Sample curve has one value per checkpoint (0..n_outputs-1). Align
        # to the target's idx set.
        if max(idxs) >= sz.shape[0]:
            return None
        sub = sz[np.array(idxs, dtype=int)]
        diff = sub - np.asarray(tvals, dtype=np.float32)
    else:
        tv = np.asarray(target_z_curve, dtype=np.float32)
        if tv.shape[0] != sz.shape[0]:
            return None
        diff = sz - tv

    if diff.size < 2:
        return None
    if mode == "mse":
        return float(np.mean(diff ** 2))
    return float(np.trapezoid(np.abs(diff)))
