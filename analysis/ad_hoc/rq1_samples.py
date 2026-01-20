"""
Find features that activate on the same samples across Pythia-160m and Pythia-6.9b

Compares feature sample composition to identify semantically similar features
even if they have different feature IDs, then analyzes whether they were 
acquired in one model vs the other.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm

# Configuration
DATA_DIR = "/home/nsrikant/BehaviorBoxNew/analysis/visualizations/dash/precomputed_data"
OUTPUT_DIR = "/home/nsrikant/BehaviorBoxNew/analysis/rq1_sample_overlap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets
DATASETS = {
    "Pythia-160m": "Pythia-piletrainedonpile.json",
    "Pythia-6.9b": "Pythia6.9b-piletrainedonpile.json"
}

THRESHOLD = 0.5
MIN_OVERLAP = 0.3  # Minimum 30% sample overlap to consider features "similar"

def load_dataset(filepath):
    """Load precomputed feature data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def calculate_acquisition_checkpoint(median_probs, threshold=0.5):
    """Find first checkpoint where probability crosses threshold"""
    for i, prob in enumerate(median_probs):
        if prob >= threshold:
            return i
    return -1

def extract_feature_info(data):
    """Extract feature information including sample composition"""
    features = []
    
    for feature in data['features']:
        median_probs = feature['median_probs']
        
        if len(median_probs) == 0:
            continue
        
        # Get sample word_ids
        samples = feature.get('samples', [])
        word_ids = set([s['word_id'] for s in samples if 'word_id' in s])
        
        acq_checkpoint = calculate_acquisition_checkpoint(median_probs, THRESHOLD)
        
        features.append({
            'feature_id': int(feature['feature_id']) if isinstance(feature['feature_id'], str) else feature['feature_id'],
            'description': feature['description'],
            'trend': feature['trend_median_probs'],
            'acquired': acq_checkpoint >= 0,
            'acquisition_checkpoint': acq_checkpoint,
            'max_probability': max(median_probs) if median_probs else 0,
            'final_probability': median_probs[-1] if median_probs else 0,
            'word_ids': word_ids,
            'n_samples': len(word_ids),
            'good_count': feature.get('good_count', 0),
            'bad_count': feature.get('bad_count', 0)
        })
    
    return features

def calculate_sample_overlap(word_ids_1, word_ids_2):
    """Calculate Jaccard similarity between two sets of word_ids"""
    if len(word_ids_1) == 0 or len(word_ids_2) == 0:
        return 0.0
    
    intersection = len(word_ids_1 & word_ids_2)
    union = len(word_ids_1 | word_ids_2)
    
    return intersection / union if union > 0 else 0.0

def find_matching_features(features_160m, features_6_9b, min_overlap=0.3):
    """Find feature pairs with significant sample overlap"""
    
    print(f"\nFinding feature pairs with ≥{min_overlap*100}% sample overlap...")
    matches = []
    
    for feat_160m in tqdm(features_160m, desc="Comparing features"):
        for feat_6_9b in features_6_9b:
            overlap = calculate_sample_overlap(feat_160m['word_ids'], feat_6_9b['word_ids'])
            
            if overlap >= min_overlap:
                matches.append({
                    '160m_feature_id': feat_160m['feature_id'],
                    '160m_description': feat_160m['description'],
                    '160m_trend': feat_160m['trend'],
                    '160m_acquired': feat_160m['acquired'],
                    '160m_acq_checkpoint': feat_160m['acquisition_checkpoint'],
                    '160m_max_prob': feat_160m['max_probability'],
                    '160m_final_prob': feat_160m['final_probability'],
                    '6.9b_feature_id': feat_6_9b['feature_id'],
                    '6.9b_description': feat_6_9b['description'],
                    '6.9b_trend': feat_6_9b['trend'],
                    '6.9b_acquired': feat_6_9b['acquired'],
                    '6.9b_acq_checkpoint': feat_6_9b['acquisition_checkpoint'],
                    '6.9b_max_prob': feat_6_9b['max_probability'],
                    '6.9b_final_prob': feat_6_9b['final_probability'],
                    'sample_overlap': overlap,
                    'n_shared_samples': len(feat_160m['word_ids'] & feat_6_9b['word_ids']),
                    '160m_n_samples': feat_160m['n_samples'],
                    '6.9b_n_samples': feat_6_9b['n_samples']
                })
    
    return pd.DataFrame(matches)

def categorize_match(row):
    """Categorize match type based on acquisition status"""
    if row['160m_acquired'] and row['6.9b_acquired']:
        return 'Both Acquired'
    elif not row['160m_acquired'] and not row['6.9b_acquired']:
        return 'Both Not Acquired'
    elif row['160m_acquired'] and not row['6.9b_acquired']:
        return '160m Only'
    else:
        return '6.9b Only'

def analyze_matches(df_matches):
    """Analyze the matched features"""
    
    print(f"\n{'='*80}")
    print("MATCHED FEATURES ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nTotal matched feature pairs: {len(df_matches)}")
    print(f"Average sample overlap: {df_matches['sample_overlap'].mean():.3f}")
    print(f"Median sample overlap: {df_matches['sample_overlap'].median():.3f}")
    
    # Add match category
    df_matches['match_category'] = df_matches.apply(categorize_match, axis=1)
    
    # Category breakdown
    print(f"\n📊 Acquisition Status Distribution:")
    category_counts = df_matches['match_category'].value_counts()
    for cat, count in category_counts.items():
        pct = (count / len(df_matches)) * 100
        print(f"  {cat:<20}: {count:4d} ({pct:5.1f}%)")
    
    # Save by category
    for category in ['Both Acquired', 'Both Not Acquired', '160m Only', '6.9b Only']:
        df_cat = df_matches[df_matches['match_category'] == category]
        
        if len(df_cat) == 0:
            continue
        
        # Sort by overlap
        df_cat = df_cat.sort_values('sample_overlap', ascending=False)
        
        filename = f"matched_{category.replace(' ', '_').lower()}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_cat.to_csv(filepath, index=False)
        
        print(f"\n  ✅ Saved {len(df_cat)} matches: {filename}")
        
        # Show top 5
        if len(df_cat) > 0:
            print(f"\n  Top 5 matches for '{category}':")
            for _, row in df_cat.head(5).iterrows():
                print(f"    160m #{int(row['160m_feature_id']):4d} ({row['160m_trend']:20s}) ↔ "
                      f"6.9b #{int(row['6.9b_feature_id']):4d} ({row['6.9b_trend']:20s}) | "
                      f"Overlap: {row['sample_overlap']:.3f}")
                print(f"      160m: {row['160m_description'][:70]}")
                print(f"      6.9b: {row['6.9b_description'][:70]}")
    
    # Trend analysis
    print(f"\n{'='*80}")
    print("TREND COMBINATION ANALYSIS")
    print(f"{'='*80}")
    
    trend_combos = df_matches.groupby(['160m_trend', '6.9b_trend']).size().reset_index(name='count')
    trend_combos = trend_combos.sort_values('count', ascending=False)
    
    print(f"\nMost common trend combinations:")
    for _, row in trend_combos.head(10).iterrows():
        print(f"  {row['160m_trend']:20s} (160m) ↔ {row['6.9b_trend']:20s} (6.9b): {row['count']:3d} pairs")
    
    csv_path = os.path.join(OUTPUT_DIR, 'trend_combinations.csv')
    trend_combos.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}")
    
    # Acquisition disagreement analysis
    print(f"\n{'='*80}")
    print("ACQUISITION DISAGREEMENT ANALYSIS")
    print(f"{'='*80}")
    
    disagreements = df_matches[df_matches['match_category'].isin(['160m Only', '6.9b Only'])]
    
    if len(disagreements) > 0:
        print(f"\n{len(disagreements)} feature pairs where only one model acquired:")
        
        # 160m acquired but 6.9b didn't
        df_160m_wins = disagreements[disagreements['match_category'] == '160m Only'].copy()
        df_160m_wins = df_160m_wins.sort_values('sample_overlap', ascending=False)
        
        if len(df_160m_wins) > 0:
            print(f"\n  160m acquired but 6.9b didn't: {len(df_160m_wins)} pairs")
            print(f"  Top examples:")
            for _, row in df_160m_wins.head(5).iterrows():
                acq_ckpt = int(row['160m_acq_checkpoint']) if row['160m_acq_checkpoint'] >= 0 else -1
                print(f"    160m #{int(row['160m_feature_id']):4d} (acq@{acq_ckpt:2d}) ↔ "
                      f"6.9b #{int(row['6.9b_feature_id']):4d} (max={row['6.9b_max_prob']:.3f}) | "
                      f"Overlap: {row['sample_overlap']:.3f}")
                print(f"      {row['160m_description'][:80]}")
        
        # 6.9b acquired but 160m didn't
        df_6_9b_wins = disagreements[disagreements['match_category'] == '6.9b Only'].copy()
        df_6_9b_wins = df_6_9b_wins.sort_values('sample_overlap', ascending=False)
        
        if len(df_6_9b_wins) > 0:
            print(f"\n  6.9b acquired but 160m didn't: {len(df_6_9b_wins)} pairs")
            print(f"  Top examples:")
            for _, row in df_6_9b_wins.head(5).iterrows():
                acq_ckpt = int(row['6.9b_acq_checkpoint']) if row['6.9b_acq_checkpoint'] >= 0 else -1
                print(f"    6.9b #{int(row['6.9b_feature_id']):4d} (acq@{acq_ckpt:2d}) ↔ "
                      f"160m #{int(row['160m_feature_id']):4d} (max={row['160m_max_prob']:.3f}) | "
                      f"Overlap: {row['sample_overlap']:.3f}")
                print(f"      {row['6.9b_description'][:80]}")
    
    # Performance comparison for "Both Not Acquired"
    print(f"\n{'='*80}")
    print("BOTH NOT ACQUIRED - PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    df_both_not = df_matches[df_matches['match_category'] == 'Both Not Acquired']
    
    if len(df_both_not) > 0:
        print(f"\n{len(df_both_not)} feature pairs where both failed to acquire:")
        
        # Calculate performance gap
        df_both_not = df_both_not.copy()
        df_both_not['max_prob_diff'] = df_both_not['6.9b_max_prob'] - df_both_not['160m_max_prob']
        
        print(f"\n  Average max probability:")
        print(f"    160m: {df_both_not['160m_max_prob'].mean():.3f}")
        print(f"    6.9b: {df_both_not['6.9b_max_prob'].mean():.3f}")
        print(f"    Difference: {df_both_not['max_prob_diff'].mean():.3f}")
        
        # Features where 6.9b did much better (even though still failed)
        df_6_9b_better = df_both_not[df_both_not['max_prob_diff'] > 0.1].sort_values('max_prob_diff', ascending=False)
        
        if len(df_6_9b_better) > 0:
            print(f"\n  {len(df_6_9b_better)} pairs where 6.9b got significantly closer (>0.1 difference):")
            for _, row in df_6_9b_better.head(5).iterrows():
                print(f"    Overlap: {row['sample_overlap']:.3f} | "
                      f"160m: {row['160m_max_prob']:.3f} → 6.9b: {row['6.9b_max_prob']:.3f} "
                      f"(+{row['max_prob_diff']:.3f})")
                print(f"      {row['160m_description'][:80]}")
        
        # Save detailed comparison
        csv_path = os.path.join(OUTPUT_DIR, 'both_not_acquired_comparison.csv')
        df_both_not.sort_values('max_prob_diff', ascending=False).to_csv(csv_path, index=False)
        print(f"\n  ✅ Saved detailed comparison: {csv_path}")
    
    return df_matches

def create_summary_statistics(df_matches):
    """Create summary statistics table"""
    
    summary = []
    
    for match_cat in ['Both Acquired', 'Both Not Acquired', '160m Only', '6.9b Only']:
        df_cat = df_matches[df_matches['match_category'] == match_cat]
        
        if len(df_cat) == 0:
            continue
        
        summary.append({
            'Category': match_cat,
            'N Pairs': len(df_cat),
            'Avg Overlap': f"{df_cat['sample_overlap'].mean():.3f}",
            'Avg 160m Max Prob': f"{df_cat['160m_max_prob'].mean():.3f}",
            'Avg 6.9b Max Prob': f"{df_cat['6.9b_max_prob'].mean():.3f}",
            'Most Common 160m Trend': df_cat['160m_trend'].mode()[0] if len(df_cat) > 0 else 'N/A',
            'Most Common 6.9b Trend': df_cat['6.9b_trend'].mode()[0] if len(df_cat) > 0 else 'N/A'
        })
    
    df_summary = pd.DataFrame(summary)
    
    csv_path = os.path.join(OUTPUT_DIR, 'SUMMARY_matched_features.csv')
    df_summary.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))
    print(f"\n✅ Saved: {csv_path}")
    
    return df_summary

def main():
    """Main analysis"""
    print("="*80)
    print("FEATURE SAMPLE OVERLAP ANALYSIS")
    print("Finding features that activate on same samples across models")
    print("="*80)
    
    # Load data
    print(f"\nLoading datasets...")
    data_160m = load_dataset(os.path.join(DATA_DIR, DATASETS["Pythia-160m"]))
    data_6_9b = load_dataset(os.path.join(DATA_DIR, DATASETS["Pythia-6.9b"]))
    
    # Extract feature info
    print(f"\nExtracting feature information...")
    features_160m = extract_feature_info(data_160m)
    features_6_9b = extract_feature_info(data_6_9b)
    
    print(f"\n  Pythia-160m: {len(features_160m)} features")
    print(f"  Pythia-6.9b: {len(features_6_9b)} features")
    
    # Find matches
    df_matches = find_matching_features(features_160m, features_6_9b, min_overlap=MIN_OVERLAP)
    
    if len(df_matches) == 0:
        print("\n❌ No matching features found!")
        return
    
    # Save all matches
    all_matches_path = os.path.join(OUTPUT_DIR, 'all_matched_features.csv')
    df_matches.to_csv(all_matches_path, index=False)
    print(f"\n✅ Saved all matches: {all_matches_path}")
    
    # Analyze matches
    df_matches = analyze_matches(df_matches)
    
    # Create summary
    create_summary_statistics(df_matches)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print("="*80)
    
    print("\n📄 Generated files:")
    print("  - all_matched_features.csv              # All feature pairs with overlap")
    print("  - matched_both_acquired.csv             # Both models acquired")
    print("  - matched_both_not_acquired.csv         # Both models failed")
    print("  - matched_160m_only.csv                 # Only 160m acquired")
    print("  - matched_6.9b_only.csv                 # Only 6.9b acquired")
    print("  - both_not_acquired_comparison.csv      # Detailed comparison of failures")
    print("  - trend_combinations.csv                # Trend pattern analysis")
    print("  - SUMMARY_matched_features.csv          # Summary statistics")

if __name__ == "__main__":
    main()