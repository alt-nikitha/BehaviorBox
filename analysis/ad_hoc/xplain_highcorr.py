import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

PYTHIA_MODEL = "160m"  # "160m" or "6.9b"

# Update with your actual path
if PYTHIA_MODEL == "160m":
    json_path = "/home/nsrikant/BehaviorBoxNew/analysis/topics/Pythia160m/data_with_topics.json"
else:
    json_path = "/home/nsrikant/BehaviorBoxNew/analysis/topics/Pythia6.9b/data_with_topics.json"

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"Loading {PYTHIA_MODEL} data from {json_path}...")
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"File not found: {json_path}")
    print("Please provide the correct path to your data file")
    exit(1)

features = data['features']
model_names = data['model_names']

print(f"Loaded {len(features)} features")

# ============================================================================
# GET ACQUISITION CHECKPOINTS AND PERFORMANCE CORRELATIONS
# ============================================================================

def get_acq_checkpoint(median_probs):
    """Get checkpoint where feature first reaches 0.5 median probability"""
    if not median_probs:
        return -1
    for idx, prob in enumerate(median_probs):
        if prob is not None and prob >= 0.5:
            return idx
    return -1

correlations = []

for feature in features:
    median_probs = feature.get('median_probs', [])
    overall_perf = feature.get('overall_performance', [])
    
    if not median_probs or not overall_perf:
        continue
    
    # Filter out None values and ensure same length
    min_len = min(len(median_probs), len(overall_perf))
    probs = [median_probs[i] for i in range(min_len) if median_probs[i] is not None]
    perf = overall_perf[:min_len]
    
    if len(probs) < 3 or len(perf) < 3:
        continue
    
    # Check variance
    prob_std = np.std(probs)
    perf_std = np.std(perf)
    
    if prob_std == 0 or perf_std == 0:
        continue  # Skip constant features
    
    acq_ckpt = get_acq_checkpoint(median_probs)
    
    try:
        pearson_r, pearson_p = pearsonr(probs, perf)
        spearman_r, spearman_p = spearmanr(probs, perf)
        
        correlations.append({
            'feature_id': feature.get('feature_id'),
            'description': feature.get('description', '')[:60],
            'acq_checkpoint': acq_ckpt,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        })
    except:
        pass

print(f"Analyzed {len(correlations)} features with valid data")

# ============================================================================
# RESULTS
# ============================================================================

if correlations:
    print(f"\n" + "="*80)
    print("CORRELATION: Feature Acquisition vs Feature's BLIMP Performance")
    print("="*80)
    print("(For each feature: does reaching higher prob correlate with better BLIMP perf?)")
    
    # Summary statistics
    pearson_rs = [c['pearson_r'] for c in correlations]
    spearman_rs = [c['spearman_r'] for c in correlations]
    
    print(f"\nPearson Correlation Distribution:")
    print(f"  Mean: {np.mean(pearson_rs):.4f}")
    print(f"  Std: {np.std(pearson_rs):.4f}")
    print(f"  Min: {np.min(pearson_rs):.4f}")
    print(f"  Max: {np.max(pearson_rs):.4f}")
    print(f"  Median: {np.median(pearson_rs):.4f}")
    
    print(f"\nSpearman Correlation Distribution:")
    print(f"  Mean: {np.mean(spearman_rs):.4f}")
    print(f"  Std: {np.std(spearman_rs):.4f}")
    print(f"  Min: {np.min(spearman_rs):.4f}")
    print(f"  Max: {np.max(spearman_rs):.4f}")
    print(f"  Median: {np.median(spearman_rs):.4f}")
    
    # Count by strength
    strong_pos = len([r for r in pearson_rs if r > 0.7])
    strong_neg = len([r for r in pearson_rs if r < -0.7])
    weak = len([r for r in pearson_rs if abs(r) <= 0.3])
    moderate = len([r for r in pearson_rs if 0.3 < abs(r) <= 0.7])
    
    print(f"\nPearson r Distribution:")
    print(f"  Strong positive (r > 0.7): {strong_pos}")
    print(f"  Strong negative (r < -0.7): {strong_neg}")
    print(f"  Moderate (0.3 < |r| <= 0.7): {moderate}")
    print(f"  Weak (|r| <= 0.3): {weak}")
    
    # Correlation between acquisition checkpoint and correlation strength
    acq_ckpts = np.array([c['acq_checkpoint'] for c in correlations if c['acq_checkpoint'] >= 0])
    corr_strengths = np.array([abs(c['pearson_r']) for c in correlations if c['acq_checkpoint'] >= 0])
    
    if len(acq_ckpts) > 2 and np.std(acq_ckpts) > 0 and np.std(corr_strengths) > 0:
        meta_pearson, meta_p = pearsonr(acq_ckpts, corr_strengths)
        print(f"\n" + "="*80)
        print("META-ANALYSIS: Does Early Acquisition → Higher Feature-Performance Correlation?")
        print("="*80)
        print(f"Correlation between acq_checkpoint and |feature-perf correlation|:")
        print(f"  Pearson r = {meta_pearson:.4f} (p={meta_p:.4e})")
        
        if abs(meta_pearson) > 0.5:
            print(f"\n⚠️  Features acquired earlier have stronger feature-perf correlations")
            print(f"  This explains why so many features correlate highly!")
        else:
            print(f"\n✅ No strong relationship: acquisition timing ≠ feature-perf correlation strength")
    
    # Show top features
    # print(f"\n" + "="*80)
    # print("Top 10 Features by Pearson Correlation Strength")
    # print("="*80)
    
    # sorted_corrs = sorted(correlations, key=lambda x: abs(x['pearson_r']), reverse=True)
    
    # for i, corr in enumerate(sorted_corrs[:10]):
    #     print(f"\n{i+1}. Feature {corr['feature_id']}")
    #     print(f"   Description: {corr['description']}")
    #     print(f"   Acquired: checkpoint {corr['acq_checkpoint']} ({model_names[corr['acq_checkpoint']] if corr['acq_checkpoint'] >= 0 else 'N/A'})")
    #     print(f"   Pearson r = {corr['pearson_r']:.4f} (p={corr['pearson_p']:.4e})")
    #     print(f"   Spearman ρ = {corr['spearman_r']:.4f} (p={corr['spearman_p']:.4e})")

else:
    print("ERROR: No valid features to analyze")