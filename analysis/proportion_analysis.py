import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

feature_file = "/home/nsrikant/BehaviorBoxNew/analysis/visualizations/dash/precomputed_data/Pythia-BLIMPtrainedonpile_ofw0.9.json"
# Load your feature data (list of dicts)
with open(feature_file, "r") as f:
    features = json.load(f)

# Focus only on trend_median_probs
trend_groups = defaultdict(list)
for feat in features["features"]:
    
    trend = feat.get("trend_median_probs")
    if trend is not None:
        trend_groups[trend].append(feat)

# Aggregate subset counts and good/bad proportions per trend
trend_summaries = {}

for trend, feats in trend_groups.items():
    subset_total = defaultdict(int)
    subset_good_total = defaultdict(int)
    subset_bad_total = defaultdict(int)
    
    for feat in feats:
        if feat["spearman_corr"] is None or feat["spearman_corr"]< 0.95:
            continue
        for subset, count in feat.get("subset_counts", {}).items():
            subset_total[subset] += count
            subset_good_total[subset] += feat.get("subset_good_counts", {}).get(subset, 0)
            subset_bad_total[subset] += feat.get("subset_bad_counts", {}).get(subset, 0)
    
    # Compute proportions
    subset_summary = {}
    for subset in subset_total:
        total = subset_total[subset]
        good = subset_good_total[subset]
        bad = subset_bad_total[subset]
        subset_summary[subset] = {
            "total": total,
            "good_prop": round(good / total, 3) if total > 0 else 0,
            "bad_prop": round(bad / total, 3) if total > 0 else 0
        }
    
    trend_summaries[trend] = subset_summary

# Save summary to JSON
with open("trend_median_probs_summary_0_9.json", "w") as f:
    json.dump(trend_summaries, f, indent=2)

print("Trend median probs summary saved to trend_median_probs_summary.json")

import os

output_dir = "trend_median_probs_plots_0_9"
os.makedirs(output_dir, exist_ok=True)

for trend, summary in trend_summaries.items():
    # Sort subsets by total count descending
    sorted_subsets = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
    subsets = [s[0] for s in sorted_subsets]
    totals = [s[1]['total'] for s in sorted_subsets]
    good_props = [s[1]['good_prop'] for s in sorted_subsets]
    bad_props = [s[1]['bad_prop'] for s in sorted_subsets]
    
    x = np.arange(len(subsets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Bar for total counts
    ax.bar(x - width/2, totals, width, label='Total count', color='lightgray')
    # Stacked bars for good/bad proportions
    ax.bar(x + width/2, [t*p for t,p in zip(totals, good_props)], width, label='Good count', color='green')
    ax.bar(x + width/2, [t*p for t,p in zip(totals, bad_props)], width,
           bottom=[t*p for t,p in zip(totals, good_props)], label='Bad count', color='red')
    
    ax.set_ylabel('Counts')
    ax.set_title(f'Subset distribution for trend_median_probs: {trend}')
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"trend_median_probs_{trend}.png")
    plt.savefig(filename)
    plt.close(fig)  # Close to free memory

print(f"Plots saved in folder: {output_dir}")