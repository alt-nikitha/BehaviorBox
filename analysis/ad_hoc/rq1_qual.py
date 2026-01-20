"""
RQ1 Qualitative Analysis: Feature Acquisition Order and Grouping

For each model:
1. Acquired features (>0.5): Grouped by similarity, ordered by acquisition, separated by trend
2. Not acquired features (increasing trend): Grouped by similarity, ordered by max_prob (descending)
3. Cross-model comparison: Common features with acquisition and trend comparison
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import random

# Configuration
DATA_DIR = "/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data"
OUTPUT_DIR = "/home/nsrikant/BehaviorBoxNew/analysis/rq1_qualitative"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "Pythia-160m": "Pythia-piletrainedonpile.json",
    "Pythia-6.9b": "Pythia6.9b-piletrainedonpile.json"
}

THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.6  # 60% sample overlap to group features

random.seed(42)  # For reproducible random selection of representative features

def load_dataset(filepath):
    """Load JSON data"""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_features(data):
    """Extract feature information"""
    features = []
    
    for feature in data['features']:
        median_probs = feature.get('median_probs', [])
        
        if len(median_probs) == 0:
            continue
        
        # Get acquisition checkpoint
        acq_checkpoint = -1
        for i, prob in enumerate(median_probs):
            if prob >= THRESHOLD:
                acq_checkpoint = i
                break
        
        # Get sample word_ids
        samples = feature.get('samples', [])
        word_ids = frozenset([s['word_id'] for s in samples if 'word_id' in s])
        
        if len(word_ids) == 0:
            continue
        
        features.append({
            'feature_id': int(feature['feature_id']) if isinstance(feature['feature_id'], str) else feature['feature_id'],
            'description': feature['description'],
            'trend': feature['trend_median_probs'],
            'acquired': acq_checkpoint >= 0,
            'acq_checkpoint': acq_checkpoint,
            'max_prob': float(max(median_probs)),
            'final_prob': float(median_probs[-1]),
            'word_ids': word_ids,
            'n_samples': len(word_ids),
            'median_probs': median_probs
        })
    
    return features

def calculate_overlap(word_ids_1, word_ids_2):
    """Calculate Jaccard similarity"""
    if len(word_ids_1) == 0 or len(word_ids_2) == 0:
        return 0.0
    
    intersection = len(word_ids_1 & word_ids_2)
    union = len(word_ids_1 | word_ids_2)
    
    return intersection / union if union > 0 else 0.0

def group_similar_features(features, similarity_threshold=0.6):
    """
    Group features that activate on similar samples.
    Returns list of groups, each group has a representative and members.
    """
    
    if len(features) == 0:
        return []
    
    # Create graph of similar features
    similar_pairs = []
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features[i+1:], i+1):
            overlap = calculate_overlap(feat1['word_ids'], feat2['word_ids'])
            if overlap >= similarity_threshold:
                similar_pairs.append((i, j, overlap))
    
    # Build connected components (groups)
    feature_to_group = {}
    groups = []
    
    for i in range(len(features)):
        if i not in feature_to_group:
            # Start new group
            group_members = [i]
            feature_to_group[i] = len(groups)
            
            # Find all connected features
            queue = [i]
            while queue:
                current = queue.pop(0)
                for idx1, idx2, overlap in similar_pairs:
                    if idx1 == current and idx2 not in feature_to_group:
                        feature_to_group[idx2] = len(groups)
                        group_members.append(idx2)
                        queue.append(idx2)
                    elif idx2 == current and idx1 not in feature_to_group:
                        feature_to_group[idx1] = len(groups)
                        group_members.append(idx1)
                        queue.append(idx1)
            
            # Select random representative
            representative_idx = random.choice(group_members)
            
            groups.append({
                'representative_idx': representative_idx,
                'representative': features[representative_idx],
                'member_indices': group_members,
                'n_members': len(group_members)
            })
    
    return groups

def process_acquired_features(features, model_name):
    """
    Process acquired features: group by trend, then by similarity, order by acquisition
    """
    
    print(f"\n{'='*60}")
    print(f"Processing Acquired Features: {model_name}")
    print(f"{'='*60}")
    
    acquired = [f for f in features if f['acquired']]
    print(f"Total acquired features: {len(acquired)}")
    
    # Group by trend
    by_trend = defaultdict(list)
    for f in acquired:
        by_trend[f['trend']].append(f)
    
    # Process each trend
    for trend in sorted(by_trend.keys()):
        trend_features = by_trend[trend]
        print(f"\n  Trend: {trend} ({len(trend_features)} features)")
        
        # Group similar features
        groups = group_similar_features(trend_features, SIMILARITY_THRESHOLD)
        print(f"    Grouped into {len(groups)} feature groups")
        
        # Sort groups by acquisition checkpoint of representative
        groups_sorted = sorted(groups, key=lambda g: g['representative']['acq_checkpoint'])
        
        # Create output rows
        rows = []
        for group in groups_sorted:
            rep = group['representative']
            rows.append({
                'feature_group_id': len(rows) + 1,
                'representative_feature_id': rep['feature_id'],
                'description': rep['description'],
                'n_members': group['n_members'],
                'acq_checkpoint': rep['acq_checkpoint'],
                'final_prob': rep['final_prob'],
                'trend': rep['trend'],
                'all_feature_ids': ','.join([str(trend_features[idx]['feature_id']) for idx in group['member_indices']])
            })
        
        # Save to CSV
        df = pd.DataFrame(rows)
        filename = f"acquired_{trend}_{model_name}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"    Saved: {filename}")
        
        # Show first 5 groups
        print(f"    First 5 groups:")
        for _, row in df.head(5).iterrows():
            print(f"      Group {row['feature_group_id']:2d}: {row['description'][:60]} "
                  f"(n={row['n_members']}, acq@{row['acq_checkpoint']})")

def process_not_acquired_features(features, model_name):
    """
    Process not acquired features (increasing trend only): 
    group by similarity, order by max_prob descending
    """
    
    print(f"\n{'='*60}")
    print(f"Processing Not Acquired Features (increasing): {model_name}")
    print(f"{'='*60}")
    
    not_acquired = [f for f in features if not f['acquired'] and f['trend'] == 'increasing']
    print(f"Total not acquired 'increasing' features: {len(not_acquired)}")
    
    if len(not_acquired) == 0:
        print("  No features to process")
        return
    
    # Group similar features
    groups = group_similar_features(not_acquired, SIMILARITY_THRESHOLD)
    print(f"  Grouped into {len(groups)} feature groups")
    
    # Sort groups by max_prob of representative (descending)
    groups_sorted = sorted(groups, key=lambda g: g['representative']['max_prob'], reverse=True)
    
    # Create output rows
    rows = []
    for group in groups_sorted:
        rep = group['representative']
        rows.append({
            'feature_group_id': len(rows) + 1,
            'representative_feature_id': rep['feature_id'],
            'description': rep['description'],
            'n_members': group['n_members'],
            'max_prob': rep['max_prob'],
            'final_prob': rep['final_prob'],
            'gap_to_threshold': THRESHOLD - rep['max_prob'],
            'all_feature_ids': ','.join([str(not_acquired[idx]['feature_id']) for idx in group['member_indices']])
        })
    
    # Save to CSV
    df = pd.DataFrame(rows)
    filename = f"not_acquired_increasing_{model_name}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filename}")
    
    # Show first 5 groups
    print(f"  First 5 groups (highest potential):")
    for _, row in df.head(5).iterrows():
        print(f"    Group {row['feature_group_id']:2d}: {row['description'][:60]} "
              f"(n={row['n_members']}, max={row['max_prob']:.3f}, gap={row['gap_to_threshold']:.3f})")

def compare_models(features_160m, features_6_9b):
    """
    Find common features across models and compare acquisition and trends
    """
    
    print(f"\n{'='*60}")
    print(f"Cross-Model Comparison")
    print(f"{'='*60}")
    
    # Find all pairs with significant overlap
    matches = []
    
    print("Finding matching features...")
    for feat_160m in features_160m:
        for feat_6_9b in features_6_9b:
            overlap = calculate_overlap(feat_160m['word_ids'], feat_6_9b['word_ids'])
            
            if overlap >= SIMILARITY_THRESHOLD:
                matches.append({
                    '160m_feature_id': feat_160m['feature_id'],
                    '160m_description': feat_160m['description'],
                    '160m_trend': feat_160m['trend'],
                    '160m_acquired': feat_160m['acquired'],
                    '160m_acq_checkpoint': feat_160m['acq_checkpoint'] if feat_160m['acquired'] else -1,
                    '160m_max_prob': feat_160m['max_prob'],
                    '6.9b_feature_id': feat_6_9b['feature_id'],
                    '6.9b_description': feat_6_9b['description'],
                    '6.9b_trend': feat_6_9b['trend'],
                    '6.9b_acquired': feat_6_9b['acquired'],
                    '6.9b_acq_checkpoint': feat_6_9b['acq_checkpoint'] if feat_6_9b['acquired'] else -1,
                    '6.9b_max_prob': feat_6_9b['max_prob'],
                    'sample_overlap': overlap,
                    'n_shared_samples': len(feat_160m['word_ids'] & feat_6_9b['word_ids'])
                })
    
    if len(matches) == 0:
        print("No matching features found")
        return
    
    df_matches = pd.DataFrame(matches)
    print(f"Found {len(df_matches)} matched feature pairs")
    
    # Categorize matches
    def categorize(row):
        if row['160m_acquired'] and row['6.9b_acquired']:
            return 'Both Acquired'
        elif not row['160m_acquired'] and not row['6.9b_acquired']:
            return 'Both Not Acquired'
        elif row['160m_acquired']:
            return '160m Only'
        else:
            return '6.9b Only'
    
    df_matches['category'] = df_matches.apply(categorize, axis=1)
    
    # Add additional comparisons
    df_matches['trend_match'] = df_matches['160m_trend'] == df_matches['6.9b_trend']
    df_matches['acq_diff'] = df_matches.apply(
        lambda row: row['6.9b_acq_checkpoint'] - row['160m_acq_checkpoint'] 
        if row['category'] == 'Both Acquired' else None, 
        axis=1
    )
    
    print("\nDistribution by category:")
    for cat, count in df_matches['category'].value_counts().items():
        pct = (count / len(df_matches)) * 100
        print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Save by category
    for category in ['Both Acquired', 'Both Not Acquired', '160m Only', '6.9b Only']:
        df_cat = df_matches[df_matches['category'] == category].copy()
        
        if len(df_cat) == 0:
            continue
        
        # Sort appropriately
        if category == 'Both Acquired':
            df_cat = df_cat.sort_values('160m_acq_checkpoint')
        elif category == 'Both Not Acquired':
            df_cat = df_cat.sort_values('6.9b_max_prob', ascending=False)
        elif category == '160m Only':
            df_cat = df_cat.sort_values('160m_acq_checkpoint')
        else:  # 6.9b Only
            df_cat = df_cat.sort_values('6.9b_acq_checkpoint')
        
        filename = f"comparison_{category.replace(' ', '_').lower()}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_cat.to_csv(filepath, index=False)
        print(f"\n  Saved {len(df_cat)} matches: {filename}")
        
        # Show examples
        print(f"  First 3 examples:")
        for _, row in df_cat.head(3).iterrows():
            if category == 'Both Acquired':
                print(f"    {row['160m_feature_id']:4d} ↔ {row['6.9b_feature_id']:4d} | "
                      f"Acq: 160m@{row['160m_acq_checkpoint']:2d} 6.9b@{row['6.9b_acq_checkpoint']:2d} | "
                      f"Trends: {row['160m_trend']:15s} / {row['6.9b_trend']:15s}")
            elif category == 'Both Not Acquired':
                print(f"    {row['160m_feature_id']:4d} ↔ {row['6.9b_feature_id']:4d} | "
                      f"Max: 160m={row['160m_max_prob']:.3f} 6.9b={row['6.9b_max_prob']:.3f} | "
                      f"Trends: {row['160m_trend']:15s} / {row['6.9b_trend']:15s}")
            elif category == '160m Only':
                print(f"    {row['160m_feature_id']:4d} ↔ {row['6.9b_feature_id']:4d} | "
                      f"160m acq@{row['160m_acq_checkpoint']:2d} but 6.9b max={row['6.9b_max_prob']:.3f}")
            else:  # 6.9b Only
                print(f"    {row['160m_feature_id']:4d} ↔ {row['6.9b_feature_id']:4d} | "
                      f"6.9b acq@{row['6.9b_acq_checkpoint']:2d} but 160m max={row['160m_max_prob']:.3f}")
            print(f"      {row['160m_description'][:70]}")
    
    # Save overall comparison summary
    summary_rows = []
    for cat in ['Both Acquired', 'Both Not Acquired', '160m Only', '6.9b Only']:
        df_cat = df_matches[df_matches['category'] == cat]
        if len(df_cat) == 0:
            continue
        
        summary_rows.append({
            'category': cat,
            'n_matches': len(df_cat),
            'avg_overlap': df_cat['sample_overlap'].mean(),
            'trend_agreement': (df_cat['trend_match'].sum() / len(df_cat)) * 100 if len(df_cat) > 0 else 0,
            'most_common_160m_trend': df_cat['160m_trend'].mode()[0] if len(df_cat) > 0 else 'N/A',
            'most_common_6.9b_trend': df_cat['6.9b_trend'].mode()[0] if len(df_cat) > 0 else 'N/A'
        })
    
    df_summary = pd.DataFrame(summary_rows)
    filepath = os.path.join(OUTPUT_DIR, 'comparison_summary.csv')
    df_summary.to_csv(filepath, index=False)
    print(f"\n  Saved summary: comparison_summary.csv")
    print("\n  Summary:")
    print(df_summary.to_string(index=False))

def main():
    """Main analysis"""
    
    print("="*80)
    print("RQ1 QUALITATIVE ANALYSIS")
    print("Feature Acquisition Order with Grouping")
    print("="*80)
    
    all_features = {}
    
    # Process each model
    for model_name, filename in DATASETS.items():
        filepath = os.path.join(DATA_DIR, filename)
        
        print(f"\n{'='*80}")
        print(f"Loading: {model_name}")
        print(f"{'='*80}")
        
        data = load_dataset(filepath)
        features = extract_features(data)
        all_features[model_name] = features
        
        print(f"Total features: {len(features)}")
        print(f"Acquired: {sum(1 for f in features if f['acquired'])}")
        print(f"Not acquired: {sum(1 for f in features if not f['acquired'])}")
        
        # Process acquired features (all trends)
        process_acquired_features(features, model_name)
        
        # Process not acquired features (increasing trend only)
        process_not_acquired_features(features, model_name)
    
    # Cross-model comparison
    compare_models(all_features['Pythia-160m'], all_features['Pythia-6.9b'])
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("="*80)
    
    print("\n📄 Generated files:")
    print("\n  Acquired features (by trend, per model):")
    print("    - acquired_{trend}_{model}.csv")
    print("\n  Not acquired features (increasing trend, per model):")
    print("    - not_acquired_increasing_{model}.csv")
    print("\n  Cross-model comparison:")
    print("    - comparison_both_acquired.csv")
    print("    - comparison_both_not_acquired.csv")
    print("    - comparison_160m_only.csv")
    print("    - comparison_6.9b_only.csv")
    print("    - comparison_summary.csv")

if __name__ == "__main__":
    main()