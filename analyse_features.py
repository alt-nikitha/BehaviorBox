import pandas as pd

import numpy as np

def classify_trend_soft(arr, tol=1e-5):
    arr = np.array(arr)
    diffs = np.diff(arr)
    
    pos = np.sum(diffs > tol)
    neg = np.sum(diffs < -tol)
    zero = np.sum(np.abs(diffs) <= tol)
    
    n = len(diffs)
    
    # Roughly monotone
    if pos / n > 0.8 and neg / n < 0.1:
        return "roughly increasing"
    elif neg / n > 0.8 and pos / n < 0.1:
        return "roughly decreasing"
    
    # Roughly monotone with stagnant part
    elif pos / n > 0.5 and zero / n > 0.2 and neg / n < 0.1:
        return "roughly increasing to stagnant"
    elif neg / n > 0.5 and zero / n > 0.2 and pos / n < 0.1:
        return "roughly decreasing to stagnant"
    
    # Increasing then decreasing
    peak_idx = np.argmax(arr)
    if peak_idx > 0 and peak_idx < len(arr)-1:
        before_peak = arr[:peak_idx+1]
        after_peak = arr[peak_idx:]
        if np.sum(np.diff(before_peak) > -tol)/len(before_peak) > 0.6 and \
           np.sum(np.diff(after_peak) < tol)/len(after_peak) > 0.6:
            return "roughly increasing then decreasing"
    
    # Decreasing then increasing
    trough_idx = np.argmin(arr)
    if trough_idx > 0 and trough_idx < len(arr)-1:
        before_trough = arr[:trough_idx+1]
        after_trough = arr[trough_idx:]
        if np.sum(np.diff(before_trough) < tol)/len(before_trough) > 0.6 and \
           np.sum(np.diff(after_trough) > -tol)/len(after_trough) > 0.6:
            return "roughly decreasing then increasing"
    
    # If nothing matches
    return "other"


    
# import ast
# feature_metrics = pd.read_csv("/home/nsrikant/BehaviorBoxNew/sae_outputs/n_comparison/_seed=42_ofw=_N=3000_k=50_lp=None/feature_metrics-pythia-160m-step1000_pythia-160m-step10000_pythia-160m-step70000_pythia-160m-step100000_pythia-160m.csv")
# print(feature_metrics.columns)
# print(feature_metrics.head())

# feature_metrics['prob_means'] = feature_metrics['prob_means'].apply(lambda x: np.array([float(v) for v in x.strip("[]").split()]))



activations_df = pd.read_csv("/mnt/labshare/nsrikant/bbox_outputs/sae_outputs/n_comparison/_seed=42_ofw=_N=3000_k=50_lp=None/top-50_activations.csv")
print(activations_df.columns)
print(activations_df.head())
print(activations_df['pythia-160m-step1000'].to_list()[:5])
# for i,row in feature_metrics.iterrows():
#     prob_means = row["prob_means"]
#     print(prob_means, "->", classify_trend_soft(prob_means))

# for i,row in feature_metrics.iterrows():
#     print(row["prob_rankings"])
#     break

# print(feature_metrics["logprob_rankings"].to_list()[:10])

# increasing trend features - features where higher checkpoints have more prob

# stagnant features - features where there is an increase in probs till some point but it then remains constant

# increase decrease trend features - features where there is an increase and then sudden decrease