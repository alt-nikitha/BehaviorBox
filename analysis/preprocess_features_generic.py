"""
Generic preprocess_features script for computing correlations between
SAE feature trajectories and any lm-evaluation-harness task.

Usage:
    python preprocess_features_generic.py --task arc_challenge
    python preprocess_features_generic.py --task hellaswag --metric acc_norm,none
    python preprocess_features_generic.py --task all
"""

import json, re, glob, argparse, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress, ConstantInputWarning, NearConstantInputWarning
import os
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)


# -----------------------------
# Model family configurations
# -----------------------------
# Each family has: eval_results_dir, checkpoints_file, column_prefix, dataset_folders
MODEL_FAMILIES = {
    "olmo3": {
        "eval_results_dir": "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3",
        "checkpoints_file": "/home/nsrikant/BehaviorBoxNew/checkpoints_info/olmo3_7b_checkpoints.txt",
        "column_prefix": "olmo3-",
        "dataset_folders": {
            # "OLMo3-7b-256k": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.7_N=3000_k=50_lp=None",
            # "OLMo3-7b-256k-k200": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.7_N=3000_k=200_lp=None",
            # "OLMo3-7b-256k-16000-k200-0.5": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.5_N=16000_k=200_lp=None"
            # "OLMo3-7b-256k-16000-k200-0.7": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.7_N=16000_k=200_lp=None"
            # "OLMo3-7b-256k-12000-k25-0.8": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.8_N=12000_k=25_lp=None"
            # "OLMo3-7b-256k-3000-k25-0.8-delta": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.8_delta_N=3000_k=25_lp=None",
            # "OLMo3-7b-256k-3000-k50-0.8-delta-ortho": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.8_delta_ortho=0.001_N=3000_k=50_lp=None"
            # "OLMo3-7b-256k-3000-k10-0.8-delta-znorm-odlw=auto-varfilt=0.2": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000/n_moreearly_olmo3_seed=42_ofw=0.8_varfilt=0.2_N=3000_k=10_lp=None_znorm_odlw=auto",
            
            "OLMo3-7b-256k-3000-k25-0.8-early-and-late-checkpoints-odlw-znorm":"/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late/n_only_early_and_late_olmo3_seed=42_ofw=0.8_N=3000_k=25_lp=None_znorm_odlw=auto"





        },
    },
    # "amber": {
    #     "eval_results_dir": "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_amber",
    #     "checkpoints_file": "/home/nsrikant/BehaviorBoxNew/checkpoints_info/amber_checkpoints.txt",
    #     "column_prefix": "amber-",
    #     "dataset_folders": {
    #         # "Amber-300-12000-k25-0.8": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.8_N=12000_k=25_lp=None",
    #         # "Amber-300-3000-k50-0.7": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.7_N=3000_k=50_lp=None",
    #         # "Amber-300-k200": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.7_N=3000_k=200_lp=None",
    #         # "Amber-300-16000-k200-0.5": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.5_N=16000_k=200_lp=None"
    #         # "Amber-300-16000-k200-0.7": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.7_N=16000_k=200_lp=None"
    #         "Amber-300-3000-k25-0.8-late-checkpoints":"/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_skip1_seed=42_ofw=0.8_N=3000_k=25_lp=None_znorm_odlw=auto"
    #     },
    # },
}

# Preferred metrics per task type (ordered by preference)
METRIC_PREFERENCE = [
    "acc_norm,none",
    "acc,none",
    "exact_match,strict-match",
    "exact_match,get-answer",
    "exact_match,flexible-extract",
    "exact,none",
    "em,none",
    "f1,none",
    "perplexity,none",
]


# -----------------------------
# Load checkpoints
# -----------------------------
def load_checkpoint_names(checkpoints_file):
    with open(checkpoints_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def checkpoint_to_column(checkpoint_name, column_prefix):
    """Convert eval checkpoint name to activation CSV column name."""
    return column_prefix + checkpoint_name


# -----------------------------
# Load eval results
# -----------------------------
def detect_metric(results_dict, task_key):
    """Auto-detect the best metric for a given task from the results."""
    task_results = results_dict.get(task_key, {})
    for metric in METRIC_PREFERENCE:
        if metric in task_results:
            return metric
    # Fallback: first non-stderr, non-alias metric
    for k in task_results:
        if "stderr" not in k and k != "alias":
            return k
    return None


def load_eval_results(task_name, checkpoints, eval_results_dir, metric_override=None):
    """
    Load evaluation results for a task across all checkpoints.

    Returns:
        overall_perf: list of float (one per checkpoint, None if missing)
        metric_used: str
    """
    overall_perf = []
    metric_used = metric_override
    result_key = None  # actual key in results dict (may differ from folder name)

    for checkpoint in checkpoints:
        result_dir = os.path.join(eval_results_dir, checkpoint, task_name)
        model_dirs = glob.glob(os.path.join(result_dir, "*/"))
        if not model_dirs:
            overall_perf.append(None)
            continue

        model_dir = model_dirs[0]
        result_files = glob.glob(os.path.join(model_dir, "results_*.json"))
        if not result_files:
            overall_perf.append(None)
            continue

        with open(result_files[0], "r") as f:
            data = json.load(f)

        results = data.get("results", {})

        # Resolve the actual result key (folder name may differ from result key)
        # e.g., folder "gsm8k" -> result key "gsm8k_cot"
        if result_key is None:
            if task_name in results:
                result_key = task_name
            elif len(results) == 1:
                result_key = list(results.keys())[0]
            else:
                group_subtasks = data.get("group_subtasks", {})
                for k in group_subtasks:
                    if k in results:
                        result_key = k
                        break
                if result_key is None:
                    result_key = list(results.keys())[0]

        if metric_used is None:
            metric_used = detect_metric(results, result_key)
            if metric_used is None:
                for k in results:
                    metric_used = detect_metric(results, k)
                    if metric_used:
                        break

        task_results = results.get(result_key, {})
        val = task_results.get(metric_used)
        if val is not None:
            overall_perf.append(float(val))
        else:
            overall_perf.append(None)

    if result_key and result_key != task_name:
        print(f"  Note: folder '{task_name}' resolved to result key '{result_key}'")
    return overall_perf, metric_used


# -----------------------------
# Correlation helpers (from original)
# -----------------------------
def calculate_residual_correlation(values, perf_values):
    """Calculate residual correlation after removing linear training progress trend."""
    if len(values) != len(perf_values) or len(values) < 3:
        return None, None
    try:
        values_arr = np.array(values, dtype=float)
        perf_arr = np.array(perf_values, dtype=float)
        if np.std(values_arr) == 0 or np.std(perf_arr) == 0:
            return None, None
        slope_f, int_f, _, _, _ = linregress(range(len(values_arr)), values_arr)
        feature_residuals = values_arr - (slope_f * np.arange(len(values_arr)) + int_f)
        slope_p, int_p, _, _, _ = linregress(range(len(perf_arr)), perf_arr)
        perf_residuals = perf_arr - (slope_p * np.arange(len(perf_arr)) + int_p)
        if np.std(feature_residuals) == 0 or np.std(perf_residuals) == 0:
            return None, None
        pearson_corr, _ = stats.pearsonr(feature_residuals, perf_residuals)
        spearman_corr, _ = stats.spearmanr(feature_residuals, perf_residuals)
        return float(pearson_corr), float(spearman_corr)
    except Exception:
        return None, None


def calculate_partial_correlation(values, perf_values, baseline_values):
    """Calculate partial correlation controlling for baseline median probs.

    Regresses both feature values and task performance on the baseline
    (median prob across all features), then correlates the residuals.
    This removes the effect of general model capability improvement.
    """
    if len(values) != len(perf_values) or len(values) != len(baseline_values) or len(values) < 3:
        return None, None
    try:
        values_arr = np.array(values, dtype=float)
        perf_arr = np.array(perf_values, dtype=float)
        baseline_arr = np.array(baseline_values, dtype=float)
        if np.std(values_arr) == 0 or np.std(perf_arr) == 0 or np.std(baseline_arr) == 0:
            return None, None

        # Regress feature values on baseline, get residuals
        slope_f, int_f, _, _, _ = linregress(baseline_arr, values_arr)
        feature_residuals = values_arr - (slope_f * baseline_arr + int_f)

        # Regress task performance on baseline, get residuals
        slope_p, int_p, _, _, _ = linregress(baseline_arr, perf_arr)
        perf_residuals = perf_arr - (slope_p * baseline_arr + int_p)

        if np.std(feature_residuals) == 0 or np.std(perf_residuals) == 0:
            return None, None
        pearson_corr, _ = stats.pearsonr(feature_residuals, perf_residuals)
        spearman_corr, _ = stats.spearmanr(feature_residuals, perf_residuals)
        return float(pearson_corr), float(spearman_corr)
    except Exception:
        return None, None


def calculate_diff_correlation(values, perf_values):
    """Pearson/Spearman correlation of consecutive-checkpoint diffs.

    Differencing removes monotonic time trend, so this directly answers
    "when this feature's prob jumps, does task perf jump too?"
    """
    if len(values) != len(perf_values) or len(values) < 3:
        return None, None
    try:
        v = np.asarray(values, dtype=float)
        p = np.asarray(perf_values, dtype=float)
        dv = np.diff(v)
        dp = np.diff(p)
        if np.std(dv) == 0 or np.std(dp) == 0:
            return None, None
        pearson, _ = stats.pearsonr(dv, dp)
        spearman, _ = stats.spearmanr(dv, dp)
        return float(pearson), float(spearman)
    except Exception:
        return None, None


def calculate_partial_diff_correlation(values, perf_values, baseline_values):
    """Diff correlation controlling for diffs in the baseline series.

    Regresses both feature diffs and perf diffs on the baseline diff,
    then correlates the residuals. Isolates feature-specific jumps that
    track perf jumps beyond what general capability gains explain.
    """
    if (len(values) != len(perf_values) or len(values) != len(baseline_values)
            or len(values) < 3):
        return None, None
    try:
        v = np.asarray(values, dtype=float)
        p = np.asarray(perf_values, dtype=float)
        b = np.asarray(baseline_values, dtype=float)
        dv = np.diff(v)
        dp = np.diff(p)
        db = np.diff(b)
        if np.std(dv) == 0 or np.std(dp) == 0 or np.std(db) == 0:
            return None, None
        slope_f, int_f, _, _, _ = linregress(db, dv)
        v_res = dv - (slope_f * db + int_f)
        slope_p, int_p, _, _, _ = linregress(db, dp)
        p_res = dp - (slope_p * db + int_p)
        if np.std(v_res) == 0 or np.std(p_res) == 0:
            return None, None
        pearson, _ = stats.pearsonr(v_res, p_res)
        spearman, _ = stats.spearmanr(v_res, p_res)
        return float(pearson), float(spearman)
    except Exception:
        return None, None


EMPTY_CORRS = {
    "pearson_corr": None, "spearman_corr": None,
    "residual_pearson_corr": None, "residual_spearman_corr": None,
    "partial_pearson_corr": None, "partial_spearman_corr": None,
    "diff_pearson_corr": None, "diff_spearman_corr": None,
    "partial_diff_pearson_corr": None, "partial_diff_spearman_corr": None,
}


def calculate_correlations(values, overall_perf, baseline_median_probs=None):
    """Calculate raw, residual, partial, and diff Pearson/Spearman correlations.

    Returns (corrs_dict, overall_perf_subset).
    """
    perf_subset = overall_perf[:len(values)]
    if len(overall_perf) != len(values):
        return dict(EMPTY_CORRS), perf_subset

    valid_pairs = [(v, p, i) for i, (v, p) in enumerate(zip(values, perf_subset))
                   if p is not None]
    if len(valid_pairs) < 2:
        return dict(EMPTY_CORRS), perf_subset

    try:
        valid_values = [p[0] for p in valid_pairs]
        valid_perf = [p[1] for p in valid_pairs]
        valid_indices = [p[2] for p in valid_pairs]
        if np.std(valid_values) == 0 or np.std(valid_perf) == 0:
            return dict(EMPTY_CORRS), perf_subset

        out = dict(EMPTY_CORRS)
        raw_pearson, _ = stats.pearsonr(valid_values, valid_perf)
        raw_spearman, _ = stats.spearmanr(valid_values, valid_perf)
        out["pearson_corr"] = float(raw_pearson)
        out["spearman_corr"] = float(raw_spearman)

        res_p, res_s = calculate_residual_correlation(valid_values, valid_perf)
        out["residual_pearson_corr"] = res_p
        out["residual_spearman_corr"] = res_s

        diff_p, diff_s = calculate_diff_correlation(valid_values, valid_perf)
        out["diff_pearson_corr"] = diff_p
        out["diff_spearman_corr"] = diff_s

        if baseline_median_probs is not None:
            valid_baseline = [baseline_median_probs[i] for i in valid_indices]
            part_p, part_s = calculate_partial_correlation(
                valid_values, valid_perf, valid_baseline)
            out["partial_pearson_corr"] = part_p
            out["partial_spearman_corr"] = part_s

            pdiff_p, pdiff_s = calculate_partial_diff_correlation(
                valid_values, valid_perf, valid_baseline)
            out["partial_diff_pearson_corr"] = pdiff_p
            out["partial_diff_spearman_corr"] = pdiff_s

        return out, perf_subset
    except Exception:
        return dict(EMPTY_CORRS), perf_subset


# -----------------------------
# Shared utility functions
# -----------------------------
def convert_to_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def max_jump(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return (0, -1)
    diffs = np.diff(vals)
    jump_idx = int(np.argmax(diffs))
    jump_val = float(diffs[jump_idx])
    return (jump_val, jump_idx)


def parse_array_field(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=float)
    s = str(x).strip().strip('"')
    try:
        return np.array(json.loads(s), dtype=float)
    except Exception:
        pass
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return np.array([float(n) for n in nums], dtype=float)


def classify_trend(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 3:
        return "other"
    if n >= 5:
        smoothed = np.convolve(vals, np.ones(3) / 3, mode='valid')
        smoothed = np.concatenate([[vals[0]], smoothed, [vals[-1]]])
    else:
        smoothed = vals.copy()
    vals_min, vals_max = smoothed.min(), smoothed.max()
    rng = vals_max - vals_min
    if rng < 1e-6:
        return "other"
    normalized = (smoothed - vals_min) / rng
    start_val = normalized[0]
    end_val = normalized[-1]
    peak_idx = np.argmax(normalized)
    peak_val = normalized[peak_idx]
    trough_idx = np.argmin(normalized)
    trough_val = normalized[trough_idx]
    overall_change = end_val - start_val
    diffs = np.diff(normalized)
    pos_steps = np.sum(diffs > 0.01)
    neg_steps = np.sum(diffs < -0.01)
    window_size = max(2, int(n * 0.3))
    final_window = normalized[-window_size:]
    final_std = np.std(final_window)
    is_final_stagnant = final_std < 0.08

    if 1 < peak_idx < n - 2:
        rise = peak_val - start_val
        fall = peak_val - end_val
        if rise > 0.3 and fall > 0.3:
            return "increase-decrease"
    if 1 < trough_idx < n - 2:
        fall = start_val - trough_val
        rise = end_val - trough_val
        if fall > 0.3 and rise > 0.3:
            return "decrease-increase"
    if overall_change > 0.2 and pos_steps > neg_steps:
        if is_final_stagnant and overall_change > 0.3:
            pre_stagnant = normalized[:-(window_size - 1)]
            if len(pre_stagnant) > 1 and np.max(pre_stagnant) - pre_stagnant[0] > 0.3:
                return "increase-stagnate"
        return "increasing"
    if overall_change < -0.2 and neg_steps > pos_steps:
        if is_final_stagnant and overall_change < -0.3:
            pre_stagnant = normalized[:-(window_size - 1)]
            if len(pre_stagnant) > 1 and pre_stagnant[0] - np.min(pre_stagnant) > 0.3:
                return "decrease-stagnate"
        return "decreasing"
    return "other"


# -----------------------------
# Main processing
# -----------------------------
def load_and_process_dataset(folder, dataset_name, checkpoints, task_name,
                             overall_perf, metric_used, column_prefix):
    """Load SAE feature data and compute correlations with task performance."""
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Folder: {folder}")

    model_columns = [checkpoint_to_column(cp, column_prefix) for cp in checkpoints]

    # Load feature labels
    with open(os.path.join(folder, "feature_labels_validated/gemini-gemini-2.5-pro.json"), "r") as f:

    
    # with open(os.path.join(folder, "feature_labels_validated/azure-gpt-5.4-mini.json"), "r") as f:
    # with open(os.path.join(folder, "feature_labels_validated/neulab-claude-sonnet-4-20250514.json"), "r") as f:
        raw = json.load(f)

    # Load activations
    act_csv_path = os.path.join(folder, "top-50_activations.csv")
    act_df = pd.read_csv(act_csv_path)

    # Verify columns exist
    available_cols = set(act_df.columns)
    valid_columns = [c for c in model_columns if c in available_cols]
    valid_checkpoints_mask = [c in available_cols for c in model_columns]
    valid_checkpoints = [cp for cp, ok in zip(checkpoints, valid_checkpoints_mask) if ok]

    if len(valid_columns) == 0:
        print(f"ERROR: No matching columns found in activation CSV.")
        print(f"  Expected columns like: {model_columns[:3]}")
        print(f"  Available columns: {sorted(available_cols)[:10]}")
        return []

    print(f"Matched {len(valid_columns)}/{len(model_columns)} checkpoint columns")

    # Filter performance to valid checkpoints only
    valid_overall_perf = [overall_perf[i] for i, ok in enumerate(valid_checkpoints_mask) if ok]

    # Load embeddings
    emb_df_path = os.path.join(folder, "feature_sample_centroid-metrics.csv")
    emb_df = pd.read_csv(emb_df_path)

    # Load word samples
    word_json_path = os.path.join(folder, "top-50_words_in_context.json")
    with open(word_json_path, "r") as f:
        word_samples = json.load(f)

    # Pre-group dataframes by feature to avoid repeated full-scan filtering
    act_df["feature"] = act_df["feature"].astype(str)
    act_grouped = dict(list(act_df.groupby("feature")))

    emb_df["feature"] = emb_df["feature"].astype(str)
    emb_grouped = dict(list(emb_df.groupby("feature")))

    feature_to_cos = (
        emb_df[["feature", "sample_centroid_cos_sim"]]
        .set_index("feature")["sample_centroid_cos_sim"]
        .to_dict()
    )

    processed_features = []
    coherent_features = {fid: feat for fid, feat in raw.items()}
    print(f"Processing {len(coherent_features)}/{len(raw)} coherent features...")

    # Compute baseline median probs: median across all coherent features at each checkpoint
    all_median_probs = []
    for fid, feat in coherent_features.items():
        mp = parse_array_field(feat.get("Median Probs", "[]"))[:len(valid_columns)]
        if len(mp) == len(valid_columns):
            all_median_probs.append(mp)
    if all_median_probs:
        baseline_median_probs = np.median(np.array(all_median_probs), axis=0).tolist()
        print(f"Baseline median probs (across {len(all_median_probs)} features): {[f'{v:.4f}' for v in baseline_median_probs]}")
    else:
        baseline_median_probs = None

    for fid, feat in tqdm(coherent_features.items(), desc="Features"):
        avg_ranks = parse_array_field(feat.get("Mean Ranks", "[]"))
        avg_probs = parse_array_field(feat.get("Avg Probs", "[]"))
        median_probs = parse_array_field(feat.get("Median Probs", "[]"))

        # Trim to valid checkpoints count
        n_ckpts = len(valid_columns)
        avg_ranks = avg_ranks[:n_ckpts]
        avg_probs = avg_probs[:n_ckpts]
        median_probs = median_probs[:n_ckpts]

        # Trends
        trend_ranks = classify_trend(avg_ranks)
        trend_avg_probs = classify_trend(avg_probs)
        trend_median_probs = classify_trend(median_probs)

        # Jumps
        jump_ranks = max_jump(avg_ranks)
        jump_avg_probs = max_jump(avg_probs)
        jump_median_probs = max_jump(median_probs)

        # Get feature activations (from pre-grouped dict)
        feat_acts_all = act_grouped.get(str(fid), pd.DataFrame()).reset_index(drop=True)

        # Error ranges
        higher_errors = []
        lower_errors = []
        mean_probs_list = []
        std_probs_list = []
        for col in valid_columns:
            logprobs = feat_acts_all[col].values
            probs = np.exp(logprobs)
            lower_errors.append(float(np.min(probs)))
            higher_errors.append(float(np.max(probs)))
            mean_probs_list.append(float(np.mean(probs)))
            std_probs_list.append(float(np.std(probs)))

        max_error = float(np.max(np.array(std_probs_list, dtype=float)))

        # Ranges
        ranges = {}
        for metric_name, metric_values in [
            ("Ranks", avg_ranks), ("Avg Probs", avg_probs), ("Median Probs", median_probs)
        ]:
            if len(metric_values) > 0:
                ranges[metric_name] = float(np.max(metric_values) - np.min(metric_values))
            else:
                ranges[metric_name] = 0.0

        # Overall correlation: median_probs vs task performance
        corrs = dict(EMPTY_CORRS)
        overall_perf_subset = []
        if len(median_probs) > 0:
            corrs, overall_perf_subset = calculate_correlations(
                median_probs.tolist(), valid_overall_perf, baseline_median_probs)

        # Samples (from pre-grouped dict)
        combined = feat_acts_all.copy()
        feat_emb_df = emb_grouped.get(str(fid), pd.DataFrame())
        if len(feat_emb_df) > 0:
            feat_emb = feat_emb_df["sample_centroid_cos_sim"].values
            if len(feat_emb) < len(combined):
                cos_series = pd.Series(feat_emb, index=combined.index[:len(feat_emb)])
            else:
                cos_series = pd.Series(feat_emb[:len(combined)], index=combined.index)
            combined["cos_sim"] = cos_series
            combined = combined.sort_values("cos_sim", ascending=False, na_position="last")

        samples_data = []
        for idx, r in combined.iterrows():
            wid = str(r.get("word_id", ""))
            word_info = word_samples.get(wid, {"before": "", "word": "", "after": ""})
            activation_val = r.get("act_value")
            cos_sim_val = r.get("cos_sim")
            samples_data.append({
                "word_id": wid,
                "activation": float(activation_val) if not pd.isna(activation_val) else None,
                "cos_sim": float(cos_sim_val) if not pd.isna(cos_sim_val) else None,
                "before": word_info.get("before", ""),
                "word": word_info.get("word", ""),
                "after": word_info.get("after", ""),
            })

        feature_data = {
            "feature_id": fid,
            "description": feat.get("Description", ""),
            "winning_rank": feat.get("Winning Rank"),
            "model": feat.get("Model"),
            "sample_centroid_cos_sim": feature_to_cos.get(str(fid)),

            # Raw values
            "avg_ranks": avg_ranks.tolist(),
            "avg_probs": avg_probs.tolist(),
            "median_probs": median_probs.tolist(),
            "lower_errors": lower_errors,
            "higher_errors": higher_errors,
            "mean_probs": mean_probs_list,
            "std_probs": std_probs_list,
            "overall_performance": overall_perf_subset,

            # Trends
            "trend_ranks": trend_ranks,
            "trend_avg_probs": trend_avg_probs,
            "trend_median_probs": trend_median_probs,

            # Jumps
            "jump_ranks_value": jump_ranks[0],
            "jump_ranks_idx": jump_ranks[1],
            "jump_avg_probs_value": jump_avg_probs[0],
            "jump_avg_probs_idx": jump_avg_probs[1],
            "jump_median_probs_value": jump_median_probs[0],
            "jump_median_probs_idx": jump_median_probs[1],

            # Ranges
            "range_ranks": ranges.get("Ranks", 0),
            "range_avg_probs": ranges.get("Avg Probs", 0),
            "range_median_probs": ranges.get("Median Probs", 0),

            # Error
            "max_error": max_error,

            # Correlations - RAW
            "pearson_corr": corrs["pearson_corr"],
            "spearman_corr": corrs["spearman_corr"],

            # Correlations - RESIDUAL (detrended by linear time index)
            "residual_pearson_corr": corrs["residual_pearson_corr"],
            "residual_spearman_corr": corrs["residual_spearman_corr"],

            # Correlations - PARTIAL (controlling for baseline median probs across all features)
            "partial_pearson_corr": corrs["partial_pearson_corr"],
            "partial_spearman_corr": corrs["partial_spearman_corr"],

            # Correlations - DIFF (consecutive-checkpoint changes)
            "diff_pearson_corr": corrs["diff_pearson_corr"],
            "diff_spearman_corr": corrs["diff_spearman_corr"],

            # Correlations - PARTIAL DIFF (diff corr controlling for baseline diff)
            "partial_diff_pearson_corr": corrs["partial_diff_pearson_corr"],
            "partial_diff_spearman_corr": corrs["partial_diff_spearman_corr"],

            # Samples
            "samples": samples_data,
        }

        feature_data = convert_to_json_serializable(feature_data)
        processed_features.append(feature_data)

    return processed_features


def get_available_tasks(eval_results_dir):
    """List all tasks available in the eval results directory."""
    tasks = set()
    for checkpoint_dir in glob.glob(os.path.join(eval_results_dir, "*")):
        if os.path.isdir(checkpoint_dir):
            for task_dir in glob.glob(os.path.join(checkpoint_dir, "*")):
                if os.path.isdir(task_dir):
                    tasks.add(os.path.basename(task_dir))
    return sorted(tasks)


def process_task(task_name, family_config, metric_override=None):
    """Process a single task across all dataset folders for a model family."""
    eval_results_dir = family_config["eval_results_dir"]
    checkpoints = load_checkpoint_names(family_config["checkpoints_file"])
    column_prefix = family_config["column_prefix"]
    dataset_folders = family_config["dataset_folders"]

    print(f"\n{'='*80}")
    print(f"Task: {task_name} | Checkpoints: {checkpoints}")
    print(f"{'='*80}")

    # Load eval results
    overall_perf, metric_used = load_eval_results(
        task_name, checkpoints, eval_results_dir, metric_override
    )
    print(f"Metric: {metric_used}")
    print(f"Overall perf across checkpoints: {overall_perf}")

    # Check we have at least some valid performance data
    if all(p is None for p in overall_perf):
        print(f"WARNING: No performance data found for task '{task_name}'. Skipping.")
        return

    output_dir = f"/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_{task_name}"
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, folder in dataset_folders.items():
        if not os.path.exists(folder):
            print(f"Skipping {dataset_name}: folder not found")
            continue

        try:
            processed_data = load_and_process_dataset(
                folder, dataset_name, checkpoints, task_name,
                overall_perf, metric_used, column_prefix
            )

            output_file = os.path.join(output_dir, f"{dataset_name}.json")
            with open(output_file, "w") as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "folder": folder,
                    "task_name": task_name,
                    "metric": metric_used,
                    "checkpoints": checkpoints,
                    "overall_performance": overall_perf,
                    "features": processed_data,
                }, f, indent=2)
            # Note: baseline_median_probs is computed inside load_and_process_dataset

            print(f"Saved {len(processed_data)} features to {output_file}")

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    families = list(MODEL_FAMILIES.keys())
    parser = argparse.ArgumentParser(description="Generic feature preprocessing with eval task correlation")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (e.g., arc_challenge, hellaswag) or 'all' for all tasks")
    parser.add_argument("--family", type=str, default="all", choices=families + ["all"],
                        help=f"Model family to process (default: all). Choices: {families}")
    parser.add_argument("--metric", type=str, default=None,
                        help="Override metric (e.g., 'acc_norm,none'). Auto-detected if not specified.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks and exit")
    args = parser.parse_args()

    selected_families = families if args.family == "all" else [args.family]

    if args.list_tasks:
        for fname in selected_families:
            cfg = MODEL_FAMILIES[fname]
            tasks = get_available_tasks(cfg["eval_results_dir"])
            print(f"[{fname}] Available tasks:")
            for t in tasks:
                print(f"  {t}")
        return

    if args.task is None:
        parser.error("--task is required (or use --list-tasks)")

    for fname in selected_families:
        cfg = MODEL_FAMILIES[fname]
        print(f"\n{'#'*80}")
        print(f"# Model family: {fname}")
        print(f"{'#'*80}")

        if args.task == "all":
            tasks = get_available_tasks(cfg["eval_results_dir"])
            print(f"Processing all {len(tasks)} tasks: {tasks}")
            for task in tasks:
                process_task(task, cfg, args.metric)
        else:
            process_task(args.task, cfg, args.metric)

    print(f"\n{'='*80}")
    print("Preprocessing complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
