# import json, re
# import numpy as np
# import pandas as pd
# from scipy import stats
# import os
# from tqdm import tqdm

# # -----------------------------
# # Dataset folder configuration
# # -----------------------------
# DATASET_FOLDERS = {
#     # "Pythia-BLIMPtrainedonpile": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_blimp_trained_on_pile",
#     # "Pythia-BLIMPtrainedonpile_ofw0.9": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_blimp_trained_on_pile_ablation/eval_blimp_train_pile_seed=42_ofw=0.9_N=3000_k=50_lp=None"
#     # "Pythia-BLIMPtrainedonblimp": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_blimp_ablation/eval_blimp_seed=42_ofw=0.7_N=3000_k=50_lp=None",
#     # "Pythia-BLIMPtrainedonblimp_6400": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_blimp_ablation/eval_blimp_seed=42_ofw=0.7_N=6400_k=50_lp=None",
#     "Pythia-piletrainedonpile": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile/n_moreearly_larger_pile_seed=42_ofw=0.7_N=3000_k=50_lp=None",
#     "Pythia-piletrainedonpile_6400": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile/n_moreearly_larger_pile_seed=42_ofw=0.7_N=6400_k=50_lp=None",
#     "Pythia6.9b-piletrainedonpile": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile_6_9b_pythia/n_moreearly_larger_pile_6_9b_pythia_3000_seed=42_ofw=0.7_N=3000_k=50_lp=None",
#     "Pythia6.9b-piletrainedonpile_6400": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile_6_9b_pythia/n_moreearly_larger_pile_6_9b_pythia_seed=42_ofw=0.7_N=6400_k=50_lp=None"
# }

# performance_dataset_name = "blimp_full"
# calculate_corr = True
# correlation_folder = f"/home/nsrikant/BehaviorBoxNew/results/{performance_dataset_name}"
# PERFORMANCE_METRIC = "overall_accuracy"

# model_names = [
#     "pythia-160m-step1",
#     "pythia-160m-step2",
#     "pythia-160m-step4",
#     "pythia-160m-step8",
#     "pythia-160m-step16",
#     "pythia-160m-step32",
#     "pythia-160m-step64",
#     "pythia-160m-step128",
#     "pythia-160m-step256",
#     "pythia-160m-step512",
#     "pythia-160m-step1000",
#     "pythia-160m-step10000",
#     "pythia-160m-step70000",
#     "pythia-160m-step100000",
#     "pythia-160m"
# ]

# OUTPUT_DIR = f"/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_{performance_dataset_name}"


# from scipy import stats
# import os
# import pandas as pd

# def load_subset_performance_data(model_names, correlation_folder, performance_dataset_name):

#     subset_performance_data = {}
    
#     for model_name in model_names:
#         perf_file = os.path.join(correlation_folder, model_name, 
#                                 f"{performance_dataset_name}_summary_seq_logprob.json")
        
#         if not os.path.exists(perf_file):
#             subset_performance_data[model_name] = {}
#             continue
        
        
#         with open(perf_file, 'r') as f:
#             perf_data = json.load(f)
#         subset_acc = {}
        
#         # Find subset column
        
        
        
        
#         for subset_name, score in perf_data["per_split_accuracy"].items():
#             subset_acc[subset_name] = float(score)
            
    
#         subset_performance_data[model_name] = subset_acc
        
       
    
#     return subset_performance_data


# def calculate_subset_correlations(feature_performance, model_names, subset_performance_data):
    
#     subset_correlations = {}
    
#     # Filter out None values from feature_performance
#     valid_indices = [i for i, fp in enumerate(feature_performance) 
#                     if fp is not None and i < len(model_names)]
    
    
#     valid_feature_perf = [feature_performance[i] for i in valid_indices]
#     valid_model_names = [model_names[i] for i in valid_indices]
    
#     # Get all subsets available
#     all_subsets = set()
#     for model_name in valid_model_names:
#         subset_data = subset_performance_data.get(model_name, {})
#         all_subsets.update(subset_data.keys())
    
#     # Correlate feature with each subset
#     for subset in all_subsets:
#         subset_perf_values = []
        
#         for model_name in valid_model_names:
#             subset_data = subset_performance_data.get(model_name, {})
#             subset_acc = subset_data.get(subset)
            
#             if subset_acc is not None:
#                 subset_perf_values.append(subset_acc)
#             else:
#                 subset_perf_values = None
#                 break
        
#         # Only correlate if we have complete data
#         if subset_perf_values is None or len(subset_perf_values) != len(valid_feature_perf):
#             continue
        
#         try:
#             pearson_corr, p_value = stats.pearsonr(valid_feature_perf, subset_perf_values)
#             spearman_corr, p_value = stats.spearmanr(valid_feature_perf, subset_perf_values)
            
#             # Only include subsets with |correlation| > 0.5
#             if abs(pearson_corr) > 0.5:
#                 subset_correlations[subset] = {
#                     'pearson_correlation': float(pearson_corr),
#                     'spearman_correlation': float(spearman_corr),
#                     # 'p_value': float(p_value),
#                     'feature_performance_at_checkpoints': [float(v) for v in valid_feature_perf],
#                     'subset_performance_at_checkpoints': [float(v) for v in subset_perf_values]
#                 }
#         except:
#             pass
    
#     return subset_correlations


# def _make_model_names(base_name, steps=None):
#     """Return a list like ['{base}-step1', ..., '{base}']"""
#     if steps is None:
#         steps = [1,2,4,8,16,32,64,128,256,512,1000,10000,70000,100000]
#     names = [f"{base_name}-step{int(s)}" for s in steps]
#     names.append(base_name)
#     return names


# def get_model_names_for_dataset(dataset_name: str):
#     """Choose model name list based on dataset_name heuristics.

#     Currently recognizes 6.9b / 6_9b datasets and falls back to the
#     original `model_names` list for everything else.
#     """
#     dn = str(dataset_name).lower()
#     # Pythia 6.9b (often named with underscore in filesystem)
#     if "6.9b" in dn or "6_9b" in dn or "6.9" in dn:
#         return _make_model_names("pythia-6_9b")
#     # explicit 160m dataset -> keep existing 160m naming
#     if "160m" in dn:
#         return _make_model_names("pythia-160m")
#     # default: use the global model_names defined above
#     return list(model_names)

# # -----------------------------
# # Helper functions
# # -----------------------------
# def convert_to_json_serializable(obj):
#     """Convert numpy types to native Python types for JSON serialization"""
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {key: convert_to_json_serializable(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_json_serializable(item) for item in obj]
#     elif isinstance(obj, tuple):
#         return tuple(convert_to_json_serializable(item) for item in obj)
#     elif pd.isna(obj):
#         return None
#     else:
#         return obj
# def max_jump(values):
#     """Returns tuple (jump_value, jump_index)"""
#     vals = np.asarray(values, dtype=float)
#     if len(vals) < 2:
#         return (0, -1)
#     diffs = np.diff(vals)
#     jump_idx = np.argmax(diffs)
#     jump_val = diffs[jump_idx]
#     return (jump_val, jump_idx)

# def parse_array_field(x):
#     if isinstance(x, (list, tuple, np.ndarray)):
#         return np.array(x, dtype=float)
#     s = str(x).strip().strip('"')
#     try:
#         return np.array(json.loads(s), dtype=float)
#     except Exception:
#         pass
#     nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
#     return np.array([float(n) for n in nums], dtype=float)

# # def classify_trend(values):
# #     vals = np.asarray(values, dtype=float)
# #     n = len(vals)
# #     if n < 2:
# #         return "other"

# #     vals_min, vals_max = vals.min(), vals.max()
# #     rng = vals_max - vals_min
# #     if rng < 1e-8:
# #         return "other"
# #     normalized = (vals - vals_min) / rng

# #     peak_idx = np.argmax(normalized)
# #     trough_idx = np.argmin(normalized)
    
# #     if 1 < peak_idx < n - 1:
# #         rise = normalized[peak_idx] - normalized[0]
# #         fall = normalized[peak_idx] - normalized[-1]
# #         if rise > 0.25 and fall > 0.25:
# #             return "increase-decrease"
    
# #     if 1 < trough_idx < n - 1:
# #         fall = normalized[0] - normalized[trough_idx]
# #         rise = normalized[-1] - normalized[trough_idx]
# #         if fall > 0.25 and rise > 0.25:
# #             return "decrease-increase"
    
# #     overall_change = normalized[-1] - normalized[0]
# #     diffs = np.diff(normalized)
# #     pos_changes = np.sum(diffs > 0.05)
# #     neg_changes = np.sum(diffs < -0.05)

# #     if overall_change > 0.2 and pos_changes >= neg_changes:
# #         return "increasing"

# #     if overall_change < -0.2 and neg_changes >= pos_changes:
# #         return "decreasing"

# #     return "other"

# def classify_trend(values, debug=False):
#     """
#     Improved trend classification with better robustness.
    
#     Strategy:
#     1. Smooth values slightly to handle noise
#     2. Use relative changes rather than absolute thresholds
#     3. Look at both local and global patterns
#     4. Prioritize clear patterns over ambiguous ones
#     """
#     vals = np.asarray(values, dtype=float)
#     n = len(vals)
    
#     if n < 3:
#         return "other"
    
#     # Smooth slightly to reduce noise (moving average window of 3)
#     if n >= 5:
#         smoothed = np.convolve(vals, np.ones(3)/3, mode='valid')
#         # Pad to keep same length
#         smoothed = np.concatenate([[vals[0]], smoothed, [vals[-1]]])
#     else:
#         smoothed = vals.copy()
    
#     # Normalize to 0-1 range
#     vals_min, vals_max = smoothed.min(), smoothed.max()
#     rng = vals_max - vals_min
    
#     if rng < 1e-6:  # Essentially flat
#         return "other"
    
#     normalized = (smoothed - vals_min) / rng
    
#     # Calculate key metrics
#     start_val = normalized[0]
#     end_val = normalized[-1]
#     peak_idx = np.argmax(normalized)
#     peak_val = normalized[peak_idx]
#     trough_idx = np.argmin(normalized)
#     trough_val = normalized[trough_idx]
    
#     # Overall directional change
#     overall_change = end_val - start_val
    
#     # Calculate monotonicity (how consistently it moves in one direction)
#     diffs = np.diff(normalized)
#     pos_steps = np.sum(diffs > 0.01)  # Small threshold for noise
#     neg_steps = np.sum(diffs < -0.01)
#     zero_steps = len(diffs) - pos_steps - neg_steps
    
#     # Stagnation detection (last 30% of checkpoints)
#     window_size = max(2, int(n * 0.3))
#     final_window = normalized[-window_size:]
#     final_std = np.std(final_window)
#     is_final_stagnant = final_std < 0.08  # Tighter threshold
    
#     if debug:
#         print(f"  Values: {normalized}")
#         print(f"  Overall change: {overall_change:.3f}")
#         print(f"  Peak at {peak_idx}: {peak_val:.3f}")
#         print(f"  Trough at {trough_idx}: {trough_val:.3f}")
#         print(f"  Pos steps: {pos_steps}, Neg steps: {neg_steps}, Zero: {zero_steps}")
#         print(f"  Final stagnant: {is_final_stagnant} (std={final_std:.3f})")
    
#     # === Pattern 1: Increase-Decrease (peak in middle) ===
#     if 1 < peak_idx < n - 2:  # Peak not at edges (with margin)
#         rise = peak_val - start_val
#         fall = peak_val - end_val
        
#         # Strong rise then strong fall
#         if rise > 0.3 and fall > 0.3:
#             if debug:
#                 print(f"  -> increase-decrease (rise={rise:.3f}, fall={fall:.3f})")
#             return "increase-decrease"
    
#     # === Pattern 2: Decrease-Increase (trough in middle) ===
#     if 1 < trough_idx < n - 2:  # Trough not at edges (with margin)
#         fall = start_val - trough_val
#         rise = end_val - trough_val
        
#         # Strong fall then strong rise
#         if fall > 0.3 and rise > 0.3:
#             if debug:
#                 print(f"  -> decrease-increase (fall={fall:.3f}, rise={rise:.3f})")
#             return "decrease-increase"
    
#     # === Pattern 3: Increasing (with or without stagnation) ===
#     if overall_change > 0.2 and pos_steps > neg_steps:
#         # Check if it's increase-stagnate
#         if is_final_stagnant and overall_change > 0.3:
#             # Make sure the increase happened before stagnation
#             pre_stagnant = normalized[:-(window_size-1)]
#             if len(pre_stagnant) > 1 and np.max(pre_stagnant) - pre_stagnant[0] > 0.3:
#                 if debug:
#                     print(f"  -> increase-stagnate (change={overall_change:.3f}, final_std={final_std:.3f})")
#                 return "increase-stagnate"
        
#         # Just increasing
#         if debug:
#             print(f"  -> increasing (change={overall_change:.3f})")
#         return "increasing"
    
#     # === Pattern 4: Decreasing (with or without stagnation) ===
#     if overall_change < -0.2 and neg_steps > pos_steps:
#         # Check if it's decrease-stagnate
#         if is_final_stagnant and overall_change < -0.3:
#             # Make sure the decrease happened before stagnation
#             pre_stagnant = normalized[:-(window_size-1)]
#             if len(pre_stagnant) > 1 and pre_stagnant[0] - np.min(pre_stagnant) > 0.3:
#                 if debug:
#                     print(f"  -> decrease-stagnate (change={overall_change:.3f}, final_std={final_std:.3f})")
#                 return "decrease-stagnate"
        
#         # Just decreasing
#         if debug:
#             print(f"  -> decreasing (change={overall_change:.3f})")
#         return "decreasing"
    
#     # === Pattern 5: Other (unclear pattern) ===
#     if debug:
#         print(f"  -> other")
#     return "other"

# def load_overall_performance(model_names=None, dataset_name="blimp_full"):
#     """Load overall performance metrics for all models ONCE"""
#     print("Loading overall performance data...")
#     if model_names is None:
#         model_names = globals().get("model_names", [])
#     overall_perf = []
#     for model_name in model_names:
#         try:
#             perf_file = os.path.join(correlation_folder, model_name, f"{dataset_name}_summary_seq_logprob.json")
#             with open(perf_file, "r") as fp:
#                 c_json = json.load(fp)
#             # Ensure it's a native Python float
#             overall_perf.append(float(c_json[PERFORMANCE_METRIC])/100.0)
#         except (FileNotFoundError, KeyError):
#             overall_perf.append(None)  # Use None instead of np.nan
#     print(f"Loaded overall performance for {len(overall_perf)} models")
#     return overall_perf

# def load_all_performance_data(model_names=None, dataset_name="blimp_full"):
#     """Load all performance CSVs ONCE and index by doc_id"""
#     print("Loading feature performance data for all models...")
#     if model_names is None:
#         model_names = globals().get("model_names", [])
#     performance_data = {}
    
#     for model_name in tqdm(model_names, desc="Loading performance CSVs"):
#         try:
#             perf_file = os.path.join(correlation_folder, model_name, f"{dataset_name}_results_seq_logprob.csv")
#             if not os.path.exists(perf_file):
#                 performance_data[model_name] = None
#                 continue
            
#             perf_df = pd.read_csv(perf_file)
            
#             # Create a lookup dictionary for faster access
#             # Map doc_id -> (good_logprob, bad_logprob)
#             doc_lookup = {}
#             for _, row in perf_df.iterrows():
#                 good_id = row['good_doc_id']
#                 bad_id = row['bad_doc_id']
#                 good_logprob = row['good_seq_logprob']
#                 bad_logprob = row['bad_seq_logprob']
                
#                 # Both good and bad doc_ids can be used to look up this pair
#                 doc_lookup[good_id] = (good_logprob, bad_logprob)
#                 doc_lookup[bad_id] = (good_logprob, bad_logprob)
            
#             performance_data[model_name] = doc_lookup
            
#         except Exception as e:
#             print(f"Error loading {model_name}: {e}")
#             performance_data[model_name] = None
    
#     print(f"Loaded performance data for {len(performance_data)} models")
#     return performance_data

# def calculate_correlations(values, overall_perf):
#     """Calculate Pearson and Spearman correlations with preloaded performance metrics"""
#     if len(overall_perf) != len(values):
#         return None, None, overall_perf[:len(values)]
    
#     # Filter out None values
#     valid_pairs = [(v, p) for v, p in zip(values, overall_perf[:len(values)]) if p is not None]
    
#     if len(valid_pairs) < 2:
#         return None, None, overall_perf[:len(values)]
    
#     try:
#         valid_values = [p[0] for p in valid_pairs]
#         valid_perf = [p[1] for p in valid_pairs]
        
#         pearson_corr, _ = stats.pearsonr(valid_values, valid_perf)
#         spearman_corr, _ = stats.spearmanr(valid_values, valid_perf)
        
#         # Convert to native Python floats
#         return float(pearson_corr), float(spearman_corr), overall_perf[:len(values)]
#     except:
#         return None, None, overall_perf[:len(values)]

# def calculate_feature_performance(feature_samples_df, performance_data, feature_id="unknown"):
#     """Calculate average accuracy using preloaded performance data"""
#     doc_ids_raw = set(feature_samples_df['word_id'].astype(str))
    
#     # Parse doc_ids
#     doc_ids = set()
#     for doc_id in doc_ids_raw:
#         parts = doc_id.rsplit('_', 1)
#         if len(parts) == 2 and parts[1].isdigit():
#             doc_ids.add(parts[0])
#         else:
#             doc_ids.add(doc_id)
    
#     performance_per_model = []
#     # model_names can be derived from performance_data keys if needed
#     model_names = list(performance_data.keys()) if performance_data is not None else []
    
#     for model_name in model_names:
#         doc_lookup = performance_data.get(model_name)
        
#         if doc_lookup is None:
#             performance_per_model.append(None)  # Use None instead of np.nan
#             continue
        
#         # Find all matching samples
#         matches = []
#         for doc_id in doc_ids:
#             if doc_id in doc_lookup:
#                 good_logprob, bad_logprob = doc_lookup[doc_id]
#                 matches.append(1 if good_logprob > bad_logprob else 0)
        
#         if len(matches) == 0:
#             performance_per_model.append(None)  # Use None instead of np.nan
#         else:
#             accuracy = float(sum(matches) / len(matches))  # Convert to native float
#             performance_per_model.append(accuracy)
    
#     return performance_per_model

# def load_and_process_dataset(folder, dataset_name):
#     """Load and process all data for a single dataset"""
#     print(f"\nProcessing dataset: {dataset_name}")
#     print(f"Folder: {folder}")
#     # Choose model names based on dataset
#     model_names = get_model_names_for_dataset(dataset_name)
    
#     # Load feature labels
#     with open(os.path.join(folder, "feature_labels_validated/neulab-claude-sonnet-4-20250514.json"), "r") as f:
#         raw = json.load(f)
    
#     # Load activations
#     act_csv_path = os.path.join(folder, "top-50_activations.csv")
#     act_df = pd.read_csv(act_csv_path)
    
#     # Load embeddings
#     emb_df_path = os.path.join(folder, "feature_sample_centroid-metrics.csv")
#     emb_df = pd.read_csv(emb_df_path)
    
#     # Load word samples
#     word_json_path = os.path.join(folder, "top-50_words_in_context.json")
#     with open(word_json_path, "r") as f:
#         word_samples = json.load(f)
    
#     # Map cosine similarities
#     feature_to_cos = (
#         emb_df[["feature", "sample_centroid_cos_sim"]]
#         .assign(feature=lambda x: x["feature"].astype(str))
#         .set_index("feature")["sample_centroid_cos_sim"]
#         .to_dict()
#     )
    
#     # BATCH LOAD: Load all performance data once (dataset-specific model list)
#     overall_perf = load_overall_performance(model_names, performance_dataset_name) if calculate_corr else []
#     performance_data = load_all_performance_data(model_names, performance_dataset_name)

#     subset_performance_data = load_subset_performance_data(
#        model_names, 
#        correlation_folder, 
#        performance_dataset_name
#    )
    
#     # overall_perf_mmlu = load_overall_performance(model_names, "mmlu") if calculate_corr else []
#     # performance_data_mmlu = load_all_performance_data(model_names, "mmlu")
    
#     processed_features = []
    
#     print(f"Processing {len(raw)} features...")
#     for fid, feat in tqdm(raw.items(), desc="Features"):
#         if int(feat["Score"]) <= 0: #no features with less than zero score from llm
#             continue
#         avg_ranks = parse_array_field(feat.get("Mean Ranks", "[]"))
#         avg_probs = parse_array_field(feat.get("Avg Probs", "[]"))
#         median_probs = parse_array_field(feat.get("Median Probs", "[]"))
        
#         # Classify trends
#         trend_ranks = classify_trend(avg_ranks)
#         trend_avg_probs = classify_trend(avg_probs)
#         trend_median_probs = classify_trend(median_probs)
        
#         # Calculate jumps
#         jump_ranks = max_jump(avg_ranks)
#         jump_avg_probs = max_jump(avg_probs)
#         jump_median_probs = max_jump(median_probs)
        
#         # Get feature activations
#         feat_acts_all = act_df[act_df["feature"].astype(str) == str(fid)].reset_index(drop=True)
        
#         # Calculate error ranges
#         higher_errors = []
#         lower_errors = []
#         mean_probs = []
#         std_probs = []
#         for model in model_names:
#             logprobs = feat_acts_all[model].values
#             probs = np.exp(logprobs)
#             min_prob = float(np.min(probs))
#             max_prob = float(np.max(probs))
#             lower_errors.append(min_prob)
#             higher_errors.append(max_prob)
#             mean_probs.append(float(np.mean(probs)))
#             std_probs.append(float(np.std(probs))) 

#         # max_error = float(np.max(np.array(higher_errors) - np.array(lower_errors)))
#         max_error = float(np.max(np.array(std_probs, dtype=float))) 

        
#         # Calculate probability ranges for each metric
#         ranges = {}
#         for metric_name, metric_values in [
#             ("Ranks", avg_ranks),
#             ("Avg Probs", avg_probs),
#             ("Median Probs", median_probs)
#         ]:
#             if len(metric_values) > 0:
#                 min_val = float(np.min(metric_values[:len(model_names)]))
#                 max_val = float(np.max(metric_values[:len(model_names)]))
#                 ranges[metric_name] = float(max_val - min_val)
#             else:
#                 ranges[metric_name] = 0.0
        
#         # Calculate correlations
#         pearson_corr = None
#         spearman_corr = None
#         overall_perf_subset = []
        
#         if calculate_corr and len(median_probs) > 0:
#             pearson_corr, spearman_corr, overall_perf_subset = calculate_correlations(
#                 median_probs[:len(model_names)], overall_perf
#             )
        
#         # Calculate feature performance using preloaded data
#         feature_performance = calculate_feature_performance(feat_acts_all, performance_data, fid)
        

#         subset_corr_results = calculate_subset_correlations(
#                 median_probs,
#                 model_names,
#                 subset_performance_data
#             )
#         # Calculate correlations between feature performance and other metrics
#         feat_perf_vs_probs_pearson = None
#         feat_perf_vs_probs_spearman = None
#         feat_perf_vs_overall_pearson = None
#         feat_perf_vs_overall_spearman = None
        
#         # Check if we have valid performance data (not all None)
#         has_performance_data = not all(p is None for p in feature_performance[:len(median_probs)])
        
#         if has_performance_data:
#             # Filter out None values for correlation calculation
#             valid_pairs = [(fp, mp) for fp, mp in zip(feature_performance[:len(median_probs)], 
#                                                        median_probs[:len(model_names)]) 
#                           if fp is not None]
            
#             if len(valid_pairs) >= 2:
#                 feat_perf_clean = [p[0] for p in valid_pairs]
#                 feat_probs_clean = [p[1] for p in valid_pairs]
                
#                 try:
#                     # Feature performance vs feature probs
#                     pearson, _ = stats.pearsonr(feat_perf_clean, feat_probs_clean)
#                     spearman, _ = stats.spearmanr(feat_perf_clean, feat_probs_clean)
#                     feat_perf_vs_probs_pearson = float(pearson)
#                     feat_perf_vs_probs_spearman = float(spearman)
                    
#                     # Feature performance vs overall performance
#                     if len(overall_perf_subset) > 0:
#                         valid_triples = [(fp, op) for fp, op in zip(feature_performance[:len(median_probs)],
#                                                                      overall_perf_subset[:len(median_probs)])
#                                        if fp is not None and op is not None]
                        
#                         if len(valid_triples) >= 2:
#                             feat_perf_clean2 = [t[0] for t in valid_triples]
#                             overall_perf_clean = [t[1] for t in valid_triples]
                            
#                             pearson, _ = stats.pearsonr(feat_perf_clean2, overall_perf_clean)
#                             spearman, _ = stats.spearmanr(feat_perf_clean2, overall_perf_clean)
#                             feat_perf_vs_overall_pearson = float(pearson)
#                             feat_perf_vs_overall_spearman = float(spearman)
#                 except:
#                     pass
        
#         # Process samples
#         combined = feat_acts_all.copy()
#         if len(feature_to_cos) > 0:
#             feat_emb = emb_df[emb_df["feature"].astype(str) == str(fid)]["sample_centroid_cos_sim"].values
#             if len(feat_emb) < len(combined):
#                 cos_series = pd.Series(feat_emb, index=combined.index[:len(feat_emb)])
#             else:
#                 cos_series = pd.Series(feat_emb[:len(combined)], index=combined.index)
#             combined["cos_sim"] = cos_series
#             combined = combined.sort_values("cos_sim", ascending=False, na_position="last")
        
        
            

#         # Extract subset and good/bad distributions
#         samples_data = []
#         subset_counts = {}
#         subset_good_counts = {}  # Track good samples per subset
#         subset_bad_counts = {}   # Track bad samples per subset
#         good_count = 0
#         bad_count = 0
        
#         for idx, r in combined.iterrows():
#             wid = str(r.get("word_id", ""))
#             if any(["blimp", "mmlu", "rlhf" in str(dataset_name).lower()]):
#                 subset_goodbad = wid.split("_")[:-2]
#                 subset = "_".join(subset_goodbad[:-1])
#                 goodbad = subset_goodbad[-1] if len(subset_goodbad) > 0 else ""
            
#                 # Count distributions
#                 subset_counts[subset] = subset_counts.get(subset, 0) + 1
                
#                 if goodbad == "good":
#                     good_count += 1
#                     subset_good_counts[subset] = subset_good_counts.get(subset, 0) + 1
#                 elif goodbad == "bad":
#                     bad_count += 1
#                     subset_bad_counts[subset] = subset_bad_counts.get(subset, 0) + 1
#             else:
#                 subset = "N/A"
#                 goodbad = "N/A"
                
                
#             word_info = word_samples.get(wid, {"before": "", "word": "", "after": ""})
            
#             # Convert to native Python types
#             activation_val = r.get('act_value')
#             cos_sim_val = r.get("cos_sim")
            
#             samples_data.append({
#                 "word_id": wid,
#                 "activation": float(activation_val) if not pd.isna(activation_val) else None,
#                 "cos_sim": float(cos_sim_val) if not pd.isna(cos_sim_val) else None,
#                 "subset": subset,
#                 "good_bad": goodbad,
#                 "before": word_info.get("before", ""),
#                 "word": word_info.get("word", ""),
#                 "after": word_info.get("after", "")
#             })
        
        

#         # Store feature data
#         feature_data = {
#             "feature_id": fid,
#             "description": feat.get("Description", ""),
#             "winning_rank": feat.get("Winning Rank"),
#             "model": feat.get("Model"),
#             "sample_centroid_cos_sim": feature_to_cos.get(str(fid)),
            
#             # Raw values
#             "avg_ranks": avg_ranks.tolist(),
#             "avg_probs": avg_probs.tolist(),
#             "median_probs": median_probs.tolist(),
#             "lower_errors": lower_errors,
#             "higher_errors": higher_errors,
#             "mean_probs": mean_probs,     # <-- new
#             "std_probs": std_probs,     
#             "feature_performance": feature_performance,
#             "overall_performance": overall_perf_subset,
            
#             # Trends
#             "trend_ranks": trend_ranks,
#             "trend_avg_probs": trend_avg_probs,
#             "trend_median_probs": trend_median_probs,
            
#             # Jumps
#             "jump_ranks_value": jump_ranks[0],
#             "jump_ranks_idx": jump_ranks[1],
#             "jump_avg_probs_value": jump_avg_probs[0],
#             "jump_avg_probs_idx": jump_avg_probs[1],
#             "jump_median_probs_value": jump_median_probs[0],
#             "jump_median_probs_idx": jump_median_probs[1],
            
#             # Ranges
#             "range_ranks": ranges.get("Ranks", 0),
#             "range_avg_probs": ranges.get("Avg Probs", 0),
#             "range_median_probs": ranges.get("Median Probs", 0),
            
#             # Error
#             "max_error": max_error,
            
#             # Correlations
#             "pearson_corr": pearson_corr,
#             "spearman_corr": spearman_corr,
#             "feat_perf_vs_probs_pearson": feat_perf_vs_probs_pearson,
#             "feat_perf_vs_probs_spearman": feat_perf_vs_probs_spearman,
#             "feat_perf_vs_overall_pearson": feat_perf_vs_overall_pearson,
#             "feat_perf_vs_overall_spearman": feat_perf_vs_overall_spearman,
#             "has_performance_data": has_performance_data,
            
#             # Distributions
#             "subset_counts": subset_counts,
#             "subset_good_counts": subset_good_counts,
#             "subset_bad_counts": subset_bad_counts,
#             "good_count": good_count,
#             "bad_count": bad_count,
            
#             "subset_correlations": subset_corr_results,
#             # Samples
#             "samples": samples_data
#         }
        
#         # Convert all numpy types to native Python types for JSON serialization
#         feature_data = convert_to_json_serializable(feature_data)
        
#         processed_features.append(feature_data)
    
#     return processed_features

# def main():
#     """Main preprocessing function"""
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     for dataset_name, folder in DATASET_FOLDERS.items():
#         print(f"\n{'='*80}")
#         print(f"Processing dataset: {dataset_name}")
#         print(f"{'='*80}")
        
#         try:
#             processed_data = load_and_process_dataset(folder, dataset_name)
            
#             # Save to JSON
#             output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}.json")
#             with open(output_file, "w") as f:
#                 json.dump({
#                     "dataset_name": dataset_name,
#                     "folder": folder,
#                     "model_names": get_model_names_for_dataset(dataset_name),
#                     "features": processed_data
#                 }, f, indent=2)
            
#             print(f"\nSaved {len(processed_data)} features to {output_file}")
            
#         except Exception as e:
#             print(f"Error processing {dataset_name}: {e}")
#             import traceback
#             traceback.print_exc()
        
    
#     print(f"\n{'='*80}")
#     print(f"Preprocessing complete! Data saved to {OUTPUT_DIR}/")
#     print(f"{'='*80}")

# if __name__ == "__main__":
#     main()



import json, re
import numpy as np
import pandas as pd
from scipy import stats
import os
from tqdm import tqdm

# Add residual correlation function at the top
from scipy.stats import linregress

def calculate_residual_correlation(values, perf_values):
    """
    Calculate residual correlation after removing training progress effect.
    
    Steps:
    1. Remove linear trend from feature values
    2. Remove linear trend from performance values
    3. Correlate the residuals
    """
    
    if len(values) != len(perf_values) or len(values) < 3:
        return None, None
    
    try:
        values_arr = np.array(values, dtype=float)
        perf_arr = np.array(perf_values, dtype=float)
        
        # Remove training progress effect from feature trajectory
        slope_f, int_f, _, _, _ = linregress(range(len(values_arr)), values_arr)
        feature_residuals = values_arr - (slope_f * np.arange(len(values_arr)) + int_f)
        
        # Remove training progress effect from performance trajectory
        slope_p, int_p, _, _, _ = linregress(range(len(perf_arr)), perf_arr)
        perf_residuals = perf_arr - (slope_p * np.arange(len(perf_arr)) + int_p)
        
        # Correlate residuals
        pearson_corr, _ = stats.pearsonr(feature_residuals, perf_residuals)
        spearman_corr, _ = stats.spearmanr(feature_residuals, perf_residuals)
        
        return float(pearson_corr), float(spearman_corr)
    except:
        return None, None

# -----------------------------
# Dataset folder configuration
# -----------------------------
DATASET_FOLDERS = {
    "Pythia-piletrainedonpile": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile/n_moreearly_larger_pile_seed=42_ofw=0.7_N=3000_k=50_lp=None",
    # "Pythia-piletrainedonpile_6400": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile/n_moreearly_larger_pile_seed=42_ofw=0.7_N=6400_k=50_lp=None",
    "Pythia6.9b-piletrainedonpile": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile_6_9b_pythia/n_moreearly_larger_pile_6_9b_pythia_3000_seed=42_ofw=0.7_N=3000_k=50_lp=None",
    # "Pythia6.9b-piletrainedonpile_6400": "/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_larger_pile_6_9b_pythia/n_moreearly_larger_pile_6_9b_pythia_seed=42_ofw=0.7_N=6400_k=50_lp=None"
}

performance_dataset_name = "blimp_full"
calculate_corr = True
correlation_folder = f"/home/nsrikant/BehaviorBoxNew/results/{performance_dataset_name}"
PERFORMANCE_METRIC = "overall_accuracy"

model_names = [
    "pythia-160m-step1",
    "pythia-160m-step2",
    "pythia-160m-step4",
    "pythia-160m-step8",
    "pythia-160m-step16",
    "pythia-160m-step32",
    "pythia-160m-step64",
    "pythia-160m-step128",
    "pythia-160m-step256",
    "pythia-160m-step512",
    "pythia-160m-step1000",
    "pythia-160m-step10000",
    "pythia-160m-step70000",
    "pythia-160m-step100000",
    "pythia-160m"
]

OUTPUT_DIR = f"/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_{performance_dataset_name}"


from scipy import stats
import os
import pandas as pd

def load_subset_performance_data(model_names, correlation_folder, performance_dataset_name):

    subset_performance_data = {}
    
    for model_name in model_names:
        perf_file = os.path.join(correlation_folder, model_name, 
                                f"{performance_dataset_name}_summary_seq_logprob.json")
        
        if not os.path.exists(perf_file):
            subset_performance_data[model_name] = {}
            continue
        
        
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)
        subset_acc = {}
        
        for subset_name, score in perf_data["per_split_accuracy"].items():
            subset_acc[subset_name] = float(score)
            
    
        subset_performance_data[model_name] = subset_acc
        
       
    
    return subset_performance_data


def calculate_subset_correlations(feature_performance, model_names, subset_performance_data):
    """Calculate both RAW and RESIDUAL correlations for subsets"""
    
    subset_correlations = {}
    
    # Filter out None values from feature_performance
    valid_indices = [i for i, fp in enumerate(feature_performance) 
                    if fp is not None and i < len(model_names)]
    
    
    valid_feature_perf = [feature_performance[i] for i in valid_indices]
    valid_model_names = [model_names[i] for i in valid_indices]
    
    # Get all subsets available
    all_subsets = set()
    for model_name in valid_model_names:
        subset_data = subset_performance_data.get(model_name, {})
        all_subsets.update(subset_data.keys())
    
    # Correlate feature with each subset
    for subset in all_subsets:
        subset_perf_values = []
        
        for model_name in valid_model_names:
            subset_data = subset_performance_data.get(model_name, {})
            subset_acc = subset_data.get(subset)
            
            if subset_acc is not None:
                subset_perf_values.append(subset_acc)
            else:
                subset_perf_values = None
                break
        
        # Only correlate if we have complete data
        if subset_perf_values is None or len(subset_perf_values) != len(valid_feature_perf):
            continue
        
        try:
            # RAW CORRELATION
            raw_pearson_corr, _ = stats.pearsonr(valid_feature_perf, subset_perf_values)
            raw_spearman_corr, _ = stats.spearmanr(valid_feature_perf, subset_perf_values)
            
            # RESIDUAL CORRELATION
            residual_pearson_corr, residual_spearman_corr = calculate_residual_correlation(
                valid_feature_perf, subset_perf_values
            )
            
            # Only include subsets with |correlation| > 0.3 (using residual)
            threshold_corr = residual_pearson_corr if residual_pearson_corr is not None else raw_pearson_corr
            if threshold_corr is not None and abs(threshold_corr) > 0.3:
                subset_correlations[subset] = {
                    'raw_pearson_correlation': float(raw_pearson_corr),
                    'raw_spearman_correlation': float(raw_spearman_corr),
                    'residual_pearson_correlation': float(residual_pearson_corr) if residual_pearson_corr is not None else None,
                    'residual_spearman_correlation': float(residual_spearman_corr) if residual_spearman_corr is not None else None,
                    'feature_performance_at_checkpoints': [float(v) for v in valid_feature_perf],
                    'subset_performance_at_checkpoints': [float(v) for v in subset_perf_values]
                }
        except:
            pass
    
    return subset_correlations


def _make_model_names(base_name, steps=None):
    """Return a list like ['{base}-step1', ..., '{base}']"""
    if steps is None:
        steps = [1,2,4,8,16,32,64,128,256,512,1000,10000,70000,100000]
    names = [f"{base_name}-step{int(s)}" for s in steps]
    names.append(base_name)
    return names


def get_model_names_for_dataset(dataset_name: str):
    """Choose model name list based on dataset_name heuristics."""
    dn = str(dataset_name).lower()
    # Pythia 6.9b (often named with underscore in filesystem)
    if "6.9b" in dn or "6_9b" in dn or "6.9" in dn:
        return _make_model_names("pythia-6_9b")
    # explicit 160m dataset -> keep existing 160m naming
    if "160m" in dn:
        return _make_model_names("pythia-160m")
    # default: use the global model_names defined above
    return list(model_names)

# -----------------------------
# Helper functions
# -----------------------------
def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
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
    """Returns tuple (jump_value, jump_index)"""
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return (0, -1)
    diffs = np.diff(vals)
    jump_idx = np.argmax(diffs)
    jump_val = diffs[jump_idx]
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

def classify_trend(values, debug=False):
    """Classify trend patterns"""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    
    if n < 3:
        return "other"
    
    # Smooth slightly to reduce noise (moving average window of 3)
    if n >= 5:
        smoothed = np.convolve(vals, np.ones(3)/3, mode='valid')
        # Pad to keep same length
        smoothed = np.concatenate([[vals[0]], smoothed, [vals[-1]]])
    else:
        smoothed = vals.copy()
    
    # Normalize to 0-1 range
    vals_min, vals_max = smoothed.min(), smoothed.max()
    rng = vals_max - vals_min
    
    if rng < 1e-6:  # Essentially flat
        return "other"
    
    normalized = (smoothed - vals_min) / rng
    
    # Calculate key metrics
    start_val = normalized[0]
    end_val = normalized[-1]
    peak_idx = np.argmax(normalized)
    peak_val = normalized[peak_idx]
    trough_idx = np.argmin(normalized)
    trough_val = normalized[trough_idx]
    
    # Overall directional change
    overall_change = end_val - start_val
    
    # Calculate monotonicity
    diffs = np.diff(normalized)
    pos_steps = np.sum(diffs > 0.01)
    neg_steps = np.sum(diffs < -0.01)
    zero_steps = len(diffs) - pos_steps - neg_steps
    
    # Stagnation detection (last 30% of checkpoints)
    window_size = max(2, int(n * 0.3))
    final_window = normalized[-window_size:]
    final_std = np.std(final_window)
    is_final_stagnant = final_std < 0.08
    
    # === Pattern 1: Increase-Decrease ===
    if 1 < peak_idx < n - 2:
        rise = peak_val - start_val
        fall = peak_val - end_val
        
        if rise > 0.3 and fall > 0.3:
            return "increase-decrease"
    
    # === Pattern 2: Decrease-Increase ===
    if 1 < trough_idx < n - 2:
        fall = start_val - trough_val
        rise = end_val - trough_val
        
        if fall > 0.3 and rise > 0.3:
            return "decrease-increase"
    
    # === Pattern 3: Increasing ===
    if overall_change > 0.2 and pos_steps > neg_steps:
        if is_final_stagnant and overall_change > 0.3:
            pre_stagnant = normalized[:-(window_size-1)]
            if len(pre_stagnant) > 1 and np.max(pre_stagnant) - pre_stagnant[0] > 0.3:
                return "increase-stagnate"
        return "increasing"
    
    # === Pattern 4: Decreasing ===
    if overall_change < -0.2 and neg_steps > pos_steps:
        if is_final_stagnant and overall_change < -0.3:
            pre_stagnant = normalized[:-(window_size-1)]
            if len(pre_stagnant) > 1 and pre_stagnant[0] - np.min(pre_stagnant) > 0.3:
                return "decrease-stagnate"
        return "decreasing"
    
    # === Pattern 5: Other ===
    return "other"

def load_overall_performance(model_names=None, dataset_name="blimp_full"):
    """Load overall performance metrics for all models ONCE"""
    print("Loading overall performance data...")
    if model_names is None:
        model_names = globals().get("model_names", [])
    overall_perf = []
    for model_name in model_names:
        try:
            perf_file = os.path.join(correlation_folder, model_name, f"{dataset_name}_summary_seq_logprob.json")
            with open(perf_file, "r") as fp:
                c_json = json.load(fp)
            # Ensure it's a native Python float
            overall_perf.append(float(c_json[PERFORMANCE_METRIC])/100.0)
        except (FileNotFoundError, KeyError):
            overall_perf.append(None)  # Use None instead of np.nan
    print(f"Loaded overall performance for {len(overall_perf)} models")
    return overall_perf

def load_all_performance_data(model_names=None, dataset_name="blimp_full"):
    """Load all performance CSVs ONCE and index by doc_id"""
    print("Loading feature performance data for all models...")
    if model_names is None:
        model_names = globals().get("model_names", [])
    performance_data = {}
    
    for model_name in tqdm(model_names, desc="Loading performance CSVs"):
        try:
            perf_file = os.path.join(correlation_folder, model_name, f"{dataset_name}_results_seq_logprob.csv")
            if not os.path.exists(perf_file):
                performance_data[model_name] = None
                continue
            
            perf_df = pd.read_csv(perf_file)
            
            # Create a lookup dictionary for faster access
            # Map doc_id -> (good_logprob, bad_logprob)
            doc_lookup = {}
            for _, row in perf_df.iterrows():
                good_id = row['good_doc_id']
                bad_id = row['bad_doc_id']
                good_logprob = row['good_seq_logprob']
                bad_logprob = row['bad_seq_logprob']
                
                # Both good and bad doc_ids can be used to look up this pair
                doc_lookup[good_id] = (good_logprob, bad_logprob)
                doc_lookup[bad_id] = (good_logprob, bad_logprob)
            
            performance_data[model_name] = doc_lookup
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            performance_data[model_name] = None
    
    print(f"Loaded performance data for {len(performance_data)} models")
    return performance_data

def calculate_correlations(values, overall_perf):
    """Calculate both RAW and RESIDUAL Pearson and Spearman correlations"""
    if len(overall_perf) != len(values):
        return None, None, None, None, overall_perf[:len(values)]
    
    # Filter out None values
    valid_pairs = [(v, p) for v, p in zip(values, overall_perf[:len(values)]) if p is not None]
    
    if len(valid_pairs) < 2:
        return None, None, None, None, overall_perf[:len(values)]
    
    try:
        valid_values = [p[0] for p in valid_pairs]
        valid_perf = [p[1] for p in valid_pairs]
        
        # RAW CORRELATION
        raw_pearson, _ = stats.pearsonr(valid_values, valid_perf)
        raw_spearman, _ = stats.spearmanr(valid_values, valid_perf)
        
        # RESIDUAL CORRELATION
        residual_pearson, residual_spearman = calculate_residual_correlation(valid_values, valid_perf)
        
        # Convert to native Python floats
        return float(raw_pearson), float(raw_spearman), float(residual_pearson) if residual_pearson is not None else None, float(residual_spearman) if residual_spearman is not None else None, overall_perf[:len(values)]
    except:
        return None, None, None, None, overall_perf[:len(values)]

def calculate_feature_performance(feature_samples_df, performance_data, feature_id="unknown"):
    """Calculate average accuracy using preloaded performance data"""
    doc_ids_raw = set(feature_samples_df['word_id'].astype(str))
    
    # Parse doc_ids
    doc_ids = set()
    for doc_id in doc_ids_raw:
        parts = doc_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            doc_ids.add(parts[0])
        else:
            doc_ids.add(doc_id)
    
    performance_per_model = []
    model_names = list(performance_data.keys()) if performance_data is not None else []
    
    for model_name in model_names:
        doc_lookup = performance_data.get(model_name)
        
        if doc_lookup is None:
            performance_per_model.append(None)
            continue
        
        # Find all matching samples
        matches = []
        for doc_id in doc_ids:
            if doc_id in doc_lookup:
                good_logprob, bad_logprob = doc_lookup[doc_id]
                matches.append(1 if good_logprob > bad_logprob else 0)
        
        if len(matches) == 0:
            performance_per_model.append(None)
        else:
            accuracy = float(sum(matches) / len(matches))
            performance_per_model.append(accuracy)
    
    return performance_per_model

def load_and_process_dataset(folder, dataset_name):
    """Load and process all data for a single dataset"""
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Folder: {folder}")
    # Choose model names based on dataset
    model_names = get_model_names_for_dataset(dataset_name)
    
    # Load feature labels
    with open(os.path.join(folder, "feature_labels_validated/neulab-claude-sonnet-4-20250514.json"), "r") as f:
        raw = json.load(f)
    
    # Load activations
    act_csv_path = os.path.join(folder, "top-50_activations.csv")
    act_df = pd.read_csv(act_csv_path)
    
    # Load embeddings
    emb_df_path = os.path.join(folder, "feature_sample_centroid-metrics.csv")
    emb_df = pd.read_csv(emb_df_path)
    
    # Load word samples
    word_json_path = os.path.join(folder, "top-50_words_in_context.json")
    with open(word_json_path, "r") as f:
        word_samples = json.load(f)
    
    # Map cosine similarities
    feature_to_cos = (
        emb_df[["feature", "sample_centroid_cos_sim"]]
        .assign(feature=lambda x: x["feature"].astype(str))
        .set_index("feature")["sample_centroid_cos_sim"]
        .to_dict()
    )
    
    # BATCH LOAD: Load all performance data once (dataset-specific model list)
    overall_perf = load_overall_performance(model_names, performance_dataset_name) if calculate_corr else []
    performance_data = load_all_performance_data(model_names, performance_dataset_name)

    subset_performance_data = load_subset_performance_data(
       model_names, 
       correlation_folder, 
       performance_dataset_name
   )
    
    processed_features = []
    
    print(f"Processing {len(raw)} features...")
    for fid, feat in tqdm(raw.items(), desc="Features"):
        if int(feat["Score"]) <= 0:
            continue
        avg_ranks = parse_array_field(feat.get("Mean Ranks", "[]"))
        avg_probs = parse_array_field(feat.get("Avg Probs", "[]"))
        median_probs = parse_array_field(feat.get("Median Probs", "[]"))
        
        # Classify trends
        trend_ranks = classify_trend(avg_ranks)
        trend_avg_probs = classify_trend(avg_probs)
        trend_median_probs = classify_trend(median_probs)
        
        # Calculate jumps
        jump_ranks = max_jump(avg_ranks)
        jump_avg_probs = max_jump(avg_probs)
        jump_median_probs = max_jump(median_probs)
        
        # Get feature activations
        feat_acts_all = act_df[act_df["feature"].astype(str) == str(fid)].reset_index(drop=True)
        
        # Calculate error ranges
        higher_errors = []
        lower_errors = []
        mean_probs = []
        std_probs = []
        for model in model_names:
            logprobs = feat_acts_all[model].values
            probs = np.exp(logprobs)
            min_prob = float(np.min(probs))
            max_prob = float(np.max(probs))
            lower_errors.append(min_prob)
            higher_errors.append(max_prob)
            mean_probs.append(float(np.mean(probs)))
            std_probs.append(float(np.std(probs))) 

        max_error = float(np.max(np.array(std_probs, dtype=float))) 

        
        # Calculate probability ranges for each metric
        ranges = {}
        for metric_name, metric_values in [
            ("Ranks", avg_ranks),
            ("Avg Probs", avg_probs),
            ("Median Probs", median_probs)
        ]:
            if len(metric_values) > 0:
                min_val = float(np.min(metric_values[:len(model_names)]))
                max_val = float(np.max(metric_values[:len(model_names)]))
                ranges[metric_name] = float(max_val - min_val)
            else:
                ranges[metric_name] = 0.0
        
        # Calculate correlations (now includes both RAW and RESIDUAL)
        pearson_corr = None
        spearman_corr = None
        residual_pearson_corr = None
        residual_spearman_corr = None
        overall_perf_subset = []
        
        if calculate_corr and len(median_probs) > 0:
            pearson_corr, spearman_corr, residual_pearson_corr, residual_spearman_corr, overall_perf_subset = calculate_correlations(
                median_probs[:len(model_names)], overall_perf
            )
        
        # Calculate feature performance using preloaded data
        feature_performance = calculate_feature_performance(feat_acts_all, performance_data, fid)
        
        # Calculate subset correlations (now includes both RAW and RESIDUAL)
        subset_corr_results = calculate_subset_correlations(
                median_probs,
                model_names,
                subset_performance_data
            )
        
        # Calculate correlations between feature performance and other metrics
        feat_perf_vs_probs_pearson = None
        feat_perf_vs_probs_spearman = None
        feat_perf_vs_probs_residual_pearson = None
        feat_perf_vs_probs_residual_spearman = None
        feat_perf_vs_overall_pearson = None
        feat_perf_vs_overall_spearman = None
        feat_perf_vs_overall_residual_pearson = None
        feat_perf_vs_overall_residual_spearman = None
        
        # Check if we have valid performance data (not all None)
        has_performance_data = not all(p is None for p in feature_performance[:len(median_probs)])
        
        if has_performance_data:
            # Filter out None values for correlation calculation
            valid_pairs = [(fp, mp) for fp, mp in zip(feature_performance[:len(median_probs)], 
                                                       median_probs[:len(model_names)]) 
                          if fp is not None]
            
            if len(valid_pairs) >= 2:
                feat_perf_clean = [p[0] for p in valid_pairs]
                feat_probs_clean = [p[1] for p in valid_pairs]
                
                try:
                    # RAW: Feature performance vs feature probs
                    pearson, _ = stats.pearsonr(feat_perf_clean, feat_probs_clean)
                    spearman, _ = stats.spearmanr(feat_perf_clean, feat_probs_clean)
                    feat_perf_vs_probs_pearson = float(pearson)
                    feat_perf_vs_probs_spearman = float(spearman)
                    
                    # RESIDUAL: Feature performance vs feature probs
                    res_pearson, res_spearman = calculate_residual_correlation(feat_perf_clean, feat_probs_clean)
                    feat_perf_vs_probs_residual_pearson = float(res_pearson) if res_pearson is not None else None
                    feat_perf_vs_probs_residual_spearman = float(res_spearman) if res_spearman is not None else None
                    
                    # Feature performance vs overall performance
                    if len(overall_perf_subset) > 0:
                        valid_triples = [(fp, op) for fp, op in zip(feature_performance[:len(median_probs)],
                                                                     overall_perf_subset[:len(median_probs)])
                                       if fp is not None and op is not None]
                        
                        if len(valid_triples) >= 2:
                            feat_perf_clean2 = [t[0] for t in valid_triples]
                            overall_perf_clean = [t[1] for t in valid_triples]
                            
                            # RAW
                            pearson, _ = stats.pearsonr(feat_perf_clean2, overall_perf_clean)
                            spearman, _ = stats.spearmanr(feat_perf_clean2, overall_perf_clean)
                            feat_perf_vs_overall_pearson = float(pearson)
                            feat_perf_vs_overall_spearman = float(spearman)
                            
                            # RESIDUAL
                            res_pearson, res_spearman = calculate_residual_correlation(feat_perf_clean2, overall_perf_clean)
                            feat_perf_vs_overall_residual_pearson = float(res_pearson) if res_pearson is not None else None
                            feat_perf_vs_overall_residual_spearman = float(res_spearman) if res_spearman is not None else None
                except:
                    pass
        
        # Process samples
        combined = feat_acts_all.copy()
        if len(feature_to_cos) > 0:
            feat_emb = emb_df[emb_df["feature"].astype(str) == str(fid)]["sample_centroid_cos_sim"].values
            if len(feat_emb) < len(combined):
                cos_series = pd.Series(feat_emb, index=combined.index[:len(feat_emb)])
            else:
                cos_series = pd.Series(feat_emb[:len(combined)], index=combined.index)
            combined["cos_sim"] = cos_series
            combined = combined.sort_values("cos_sim", ascending=False, na_position="last")

        # Extract subset and good/bad distributions
        samples_data = []
        subset_counts = {}
        subset_good_counts = {}
        subset_bad_counts = {}
        good_count = 0
        bad_count = 0
        
        for idx, r in combined.iterrows():
            wid = str(r.get("word_id", ""))
            if any(["blimp", "mmlu", "rlhf" in str(dataset_name).lower()]):
                subset_goodbad = wid.split("_")[:-2]
                subset = "_".join(subset_goodbad[:-1])
                goodbad = subset_goodbad[-1] if len(subset_goodbad) > 0 else ""
            
                # Count distributions
                subset_counts[subset] = subset_counts.get(subset, 0) + 1
                
                if goodbad == "good":
                    good_count += 1
                    subset_good_counts[subset] = subset_good_counts.get(subset, 0) + 1
                elif goodbad == "bad":
                    bad_count += 1
                    subset_bad_counts[subset] = subset_bad_counts.get(subset, 0) + 1
            else:
                subset = "N/A"
                goodbad = "N/A"
                
            word_info = word_samples.get(wid, {"before": "", "word": "", "after": ""})
            
            # Convert to native Python types
            activation_val = r.get('act_value')
            cos_sim_val = r.get("cos_sim")
            
            samples_data.append({
                "word_id": wid,
                "activation": float(activation_val) if not pd.isna(activation_val) else None,
                "cos_sim": float(cos_sim_val) if not pd.isna(cos_sim_val) else None,
                "subset": subset,
                "good_bad": goodbad,
                "before": word_info.get("before", ""),
                "word": word_info.get("word", ""),
                "after": word_info.get("after", "")
            })

        # Store feature data
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
            "mean_probs": mean_probs,
            "std_probs": std_probs,
            "feature_performance": feature_performance,
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
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            
            # Correlations - RESIDUAL
            "residual_pearson_corr": residual_pearson_corr,
            "residual_spearman_corr": residual_spearman_corr,
            
            # Feature performance correlations - RAW
            "feat_perf_vs_probs_pearson": feat_perf_vs_probs_pearson,
            "feat_perf_vs_probs_spearman": feat_perf_vs_probs_spearman,
            "feat_perf_vs_overall_pearson": feat_perf_vs_overall_pearson,
            "feat_perf_vs_overall_spearman": feat_perf_vs_overall_spearman,
            
            # Feature performance correlations - RESIDUAL
            "feat_perf_vs_probs_residual_pearson": feat_perf_vs_probs_residual_pearson,
            "feat_perf_vs_probs_residual_spearman": feat_perf_vs_probs_residual_spearman,
            "feat_perf_vs_overall_residual_pearson": feat_perf_vs_overall_residual_pearson,
            "feat_perf_vs_overall_residual_spearman": feat_perf_vs_overall_residual_spearman,
            
            "has_performance_data": has_performance_data,
            
            # Distributions
            "subset_counts": subset_counts,
            "subset_good_counts": subset_good_counts,
            "subset_bad_counts": subset_bad_counts,
            "good_count": good_count,
            "bad_count": bad_count,
            
            # Subset correlations (now includes both RAW and RESIDUAL)
            "subset_correlations": subset_corr_results,
            
            # Samples
            "samples": samples_data
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        feature_data = convert_to_json_serializable(feature_data)
        
        processed_features.append(feature_data)
    
    return processed_features

def main():
    """Main preprocessing function"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for dataset_name, folder in DATASET_FOLDERS.items():
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        try:
            processed_data = load_and_process_dataset(folder, dataset_name)
            
            # Save to JSON
            output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}.json")
            with open(output_file, "w") as f:
                json.dump({
                    "dataset_name": dataset_name,
                    "folder": folder,
                    "model_names": get_model_names_for_dataset(dataset_name),
                    "features": processed_data
                }, f, indent=2)
            
            print(f"\nSaved {len(processed_data)} features to {output_file}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
        
    
    print(f"\n{'='*80}")
    print(f"Preprocessing complete! Data saved to {OUTPUT_DIR}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()