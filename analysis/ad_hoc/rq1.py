"""
RQ1 Analysis: Feature Acquisition CSV Tables by Trend Type

For features that cross 0.5:
- CSV per trend, sorted by acquisition checkpoint

For features that never cross 0.5:
- CSV per trend, sorted by max_jump
- CSV per trend, sorted by maximum probability reached
"""

import json
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# Configuration
DATA_DIR = "/home/nsrikant/BehaviorBoxNew/analysis/visualizations/dash/precomputed_data"
OUTPUT_DIR = "/home/nsrikant/BehaviorBoxNew/analysis/rq1_csv_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets to analyze
DATASETS = {
    "Pythia-160m": "Pythia-piletrainedonpile.json",
    "Pythia-6.9b": "Pythia6.9b-piletrainedonpile.json"
}

# Acquisition threshold
THRESHOLD = 0.5

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

def calculate_max_jump(median_probs):
    """Calculate maximum jump and its location"""
    if len(median_probs) < 2:
        return 0, -1
    
    diffs = np.diff(median_probs)
    max_jump_idx = np.argmax(diffs)
    max_jump_value = diffs[max_jump_idx]
    
    return max_jump_value, max_jump_idx

def process_features(data, model_names):
    """Process all features and categorize them"""
    
    acquired_features = defaultdict(list)  # trend -> list of features
    not_acquired_features = defaultdict(list)  # trend -> list of features
    
    for feature in data['features']:
        median_probs = feature['median_probs']
        
        if len(median_probs) == 0:
            continue
        
        trend = feature['trend_median_probs']
        acq_checkpoint = calculate_acquisition_checkpoint(median_probs, THRESHOLD)
        max_jump_value, max_jump_idx = calculate_max_jump(median_probs)
        max_prob = max(median_probs)
        
        # Find checkpoint where max prob was reached
        max_prob_checkpoint = next((i for i, p in enumerate(median_probs) if p == max_prob), -1)
        
        feature_data = {
            'feature_id': feature['feature_id'],
            'description': feature['description'],
            'trend': trend,
            'acquisition_checkpoint': acq_checkpoint,
            'acquisition_checkpoint_name': model_names[acq_checkpoint] if acq_checkpoint >= 0 and acq_checkpoint < len(model_names) else 'Never',
            'max_jump_value': max_jump_value,
            'max_jump_checkpoint': max_jump_idx,
            'max_jump_checkpoint_name': model_names[max_jump_idx] if max_jump_idx >= 0 and max_jump_idx < len(model_names) else 'N/A',
            'max_probability': max_prob,
            'max_prob_checkpoint': max_prob_checkpoint,
            'max_prob_checkpoint_name': model_names[max_prob_checkpoint] if max_prob_checkpoint >= 0 and max_prob_checkpoint < len(model_names) else 'N/A',
            'final_probability': median_probs[-1] if median_probs else 0,
            'median_probs_list': str(median_probs),  # For reference
            'good_count': feature.get('good_count', 0),
            'bad_count': feature.get('bad_count', 0),
            'subset_counts': str(feature.get('subset_counts', {}))
        }
        
        if acq_checkpoint >= 0:
            acquired_features[trend].append(feature_data)
        else:
            not_acquired_features[trend].append(feature_data)
    
    return acquired_features, not_acquired_features

def save_acquired_features_csvs(acquired_features, dataset_name):
    """Save CSV for each trend with acquired features sorted by acquisition order"""
    
    print(f"\n{'='*60}")
    print(f"ACQUIRED FEATURES (≥{THRESHOLD}) - {dataset_name}")
    print(f"{'='*60}")
    
    for trend in sorted(acquired_features.keys()):
        features = acquired_features[trend]
        
        if len(features) == 0:
            continue
        
        # Sort by acquisition checkpoint
        features_sorted = sorted(features, key=lambda x: x['acquisition_checkpoint'])
        
        # Create DataFrame
        df = pd.DataFrame(features_sorted)
        
        # Select and order columns
        columns = [
            'feature_id',
            'description',
            'acquisition_checkpoint',
            'acquisition_checkpoint_name',
            'final_probability',
            'max_probability',
            'good_count',
            'bad_count',
            'subset_counts'
        ]
        
        df = df[columns]
        
        # Save
        filename = f'acquired_{trend}_{dataset_name}.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"✅ {trend:25s} | {len(features):4d} features | {filename}")
    
    # Save combined file with all acquired features
    all_acquired = []
    for trend, features in acquired_features.items():
        all_acquired.extend(features)
    
    if len(all_acquired) > 0:
        df_all = pd.DataFrame(sorted(all_acquired, key=lambda x: x['acquisition_checkpoint']))
        columns = [
            'feature_id',
            'description',
            'trend',
            'acquisition_checkpoint',
            'acquisition_checkpoint_name',
            'final_probability',
            'max_probability',
            'good_count',
            'bad_count',
            'subset_counts'
        ]
        df_all = df_all[columns]
        
        filename = f'acquired_ALL_TRENDS_{dataset_name}.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_all.to_csv(filepath, index=False)
        print(f"\n✅ ALL TRENDS COMBINED     | {len(all_acquired):4d} features | {filename}")

def save_not_acquired_maxjump_csvs(not_acquired_features, dataset_name):
    """Save CSV for each trend with non-acquired features sorted by jump index (when jump occurred)"""
    
    print(f"\n{'='*60}")
    print(f"NOT ACQUIRED (by jump index/checkpoint) - {dataset_name}")
    print(f"{'='*60}")
    
    for trend in sorted(not_acquired_features.keys()):
        features = not_acquired_features[trend]
        
        if len(features) == 0:
            continue
        
        # Sort by max_jump_checkpoint (ascending - earliest jumps first)
        features_sorted = sorted(features, key=lambda x: x['max_jump_checkpoint'])
        
        # Create DataFrame
        df = pd.DataFrame(features_sorted)
        
        # Select and order columns
        columns = [
            'feature_id',
            'description',
            'max_jump_value',
            'max_jump_checkpoint',
            'max_jump_checkpoint_name',
            'max_probability',
            'final_probability',
            'good_count',
            'bad_count',
            'subset_counts'
        ]
        
        df = df[columns]
        
        # Save
        filename = f'not_acquired_JUMPINDEX_{trend}_{dataset_name}.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"✅ {trend:25s} | {len(features):4d} features | {filename}")
    
    # Save combined file
    all_not_acquired = []
    for trend, features in not_acquired_features.items():
        all_not_acquired.extend(features)
    
    if len(all_not_acquired) > 0:
        df_all = pd.DataFrame(sorted(all_not_acquired, key=lambda x: x['max_jump_checkpoint']))
        columns = [
            'feature_id',
            'description',
            'trend',
            'max_jump_value',
            'max_jump_checkpoint',
            'max_jump_checkpoint_name',
            'max_probability',
            'final_probability',
            'good_count',
            'bad_count',
            'subset_counts'
        ]
        df_all = df_all[columns]
        
        filename = f'not_acquired_JUMPINDEX_ALL_TRENDS_{dataset_name}.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_all.to_csv(filepath, index=False)
        print(f"\n✅ ALL TRENDS COMBINED     | {len(all_not_acquired):4d} features | {filename}")

def save_not_acquired_maxprob_csvs(not_acquired_features, dataset_name):
    """Save CSV for each trend with non-acquired features sorted by max_probability"""
    
    print(f"\n{'='*60}")
    print(f"NOT ACQUIRED (by max probability) - {dataset_name}")
    print(f"{'='*60}")
    
    for trend in sorted(not_acquired_features.keys()):
        features = not_acquired_features[trend]
        
        if len(features) == 0:
            continue
        
        # Sort by max_probability (descending - highest first)
        features_sorted = sorted(features, key=lambda x: x['max_probability'], reverse=True)
        
        # Create DataFrame
        df = pd.DataFrame(features_sorted)
        
        # Select and order columns
        columns = [
            'feature_id',
            'description',
            'max_probability',
            'max_prob_checkpoint',
            'max_prob_checkpoint_name',
            'max_jump_value',
            'final_probability',
            'good_count',
            'bad_count',
            'subset_counts'
        ]
        
        df = df[columns]
        
        # Save
        filename = f'not_acquired_MAXPROB_{trend}_{dataset_name}.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"✅ {trend:25s} | {len(features):4d} features | {filename}")
    
    # Save combined file
    all_not_acquired = []
    for trend, features in not_acquired_features.items():
        all_not_acquired.extend(features)
    
    if len(all_not_acquired) > 0:
        df_all = pd.DataFrame(sorted(all_not_acquired, key=lambda x: x['max_probability'], reverse=True))
        columns = [
            'feature_id',
            'description',
            'trend',
            'max_probability',
            'max_prob_checkpoint',
            'max_prob_checkpoint_name',
            'max_jump_value',
            'final_probability',
            'good_count',
            'bad_count',
            'subset_counts'
        ]
        df_all = df_all[columns]
        
        filename = f'not_acquired_MAXPROB_ALL_TRENDS_{dataset_name}.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_all.to_csv(filepath, index=False)
        print(f"\n✅ ALL TRENDS COMBINED     | {len(all_not_acquired):4d} features | {filename}")

def create_summary_table(acquired_features, not_acquired_features, dataset_name):
    """Create summary statistics table"""
    
    rows = []
    
    # All trends
    all_trends = set(list(acquired_features.keys()) + list(not_acquired_features.keys()))
    
    for trend in sorted(all_trends):
        acq = acquired_features.get(trend, [])
        not_acq = not_acquired_features.get(trend, [])
        
        row = {
            'Trend': trend,
            'Acquired (≥0.5)': len(acq),
            'Not Acquired (<0.5)': len(not_acq),
            'Total': len(acq) + len(not_acq),
            'Acquisition Rate': f"{len(acq)/(len(acq)+len(not_acq))*100:.1f}%" if (len(acq)+len(not_acq)) > 0 else "N/A"
        }
        
        if len(acq) > 0:
            acq_checkpoints = [f['acquisition_checkpoint'] for f in acq]
            row['Median Acq Ckpt'] = f"{np.median(acq_checkpoints):.1f}"
            row['Earliest Acq Ckpt'] = min(acq_checkpoints)
            row['Latest Acq Ckpt'] = max(acq_checkpoints)
        else:
            row['Median Acq Ckpt'] = 'N/A'
            row['Earliest Acq Ckpt'] = 'N/A'
            row['Latest Acq Ckpt'] = 'N/A'
        
        if len(not_acq) > 0:
            max_probs = [f['max_probability'] for f in not_acq]
            max_jumps = [f['max_jump_value'] for f in not_acq]
            row['Avg Max Prob (not acq)'] = f"{np.mean(max_probs):.3f}"
            row['Avg Max Jump (not acq)'] = f"{np.mean(max_jumps):.3f}"
        else:
            row['Avg Max Prob (not acq)'] = 'N/A'
            row['Avg Max Jump (not acq)'] = 'N/A'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add totals row
    total_acq = sum(len(v) for v in acquired_features.values())
    total_not_acq = sum(len(v) for v in not_acquired_features.values())
    totals = {
        'Trend': 'TOTAL',
        'Acquired (≥0.5)': total_acq,
        'Not Acquired (<0.5)': total_not_acq,
        'Total': total_acq + total_not_acq,
        'Acquisition Rate': f"{total_acq/(total_acq+total_not_acq)*100:.1f}%" if (total_acq+total_not_acq) > 0 else "N/A",
        'Median Acq Ckpt': '',
        'Earliest Acq Ckpt': '',
        'Latest Acq Ckpt': '',
        'Avg Max Prob (not acq)': '',
        'Avg Max Jump (not acq)': ''
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    
    # Save
    csv_path = os.path.join(OUTPUT_DIR, f'SUMMARY_TABLE_{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE - {dataset_name}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\n✅ Saved: {csv_path}")
    
    return df

def main():
    """Main RQ1 CSV generation"""
    print("="*80)
    print("RQ1 CSV TABLES: Feature Acquisition by Trend Type")
    print("="*80)
    print(f"Acquisition threshold: {THRESHOLD}")
    print("="*80)
    
    for dataset_name, filename in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")
        
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
        
        # Load data
        data = load_dataset(filepath)
        model_names = data['model_names']
        
        print(f"Total checkpoints: {len(model_names)}")
        
        # Process features
        acquired_features, not_acquired_features = process_features(data, model_names)
        
        total_acq = sum(len(v) for v in acquired_features.values())
        total_not_acq = sum(len(v) for v in not_acquired_features.values())
        
        print(f"\n📊 Features acquired (≥{THRESHOLD}): {total_acq}")
        print(f"📊 Features not acquired (<{THRESHOLD}): {total_not_acq}")
        
        # Create summary table
        create_summary_table(acquired_features, not_acquired_features, dataset_name)
        
        # Save CSVs for acquired features
        save_acquired_features_csvs(acquired_features, dataset_name)
        
        # Save CSVs for non-acquired features (by max jump)
        save_not_acquired_maxjump_csvs(not_acquired_features, dataset_name)
        
        # Save CSVs for non-acquired features (by max probability)
        save_not_acquired_maxprob_csvs(not_acquired_features, dataset_name)
    
    print("\n" + "="*80)
    print("✅ RQ1 CSV GENERATION COMPLETE!")
    print(f"📁 All CSV files saved to: {OUTPUT_DIR}/")
    print("="*80)
    
    print("\n📄 File naming convention:")
    print("  ACQUIRED features:")
    print("    - acquired_{trend}_{dataset}.csv")
    print("    - acquired_ALL_TRENDS_{dataset}.csv")
    print("")
    print("  NOT ACQUIRED features (sorted by jump index - when jump occurred):")
    print("    - not_acquired_JUMPINDEX_{trend}_{dataset}.csv")
    print("    - not_acquired_JUMPINDEX_ALL_TRENDS_{dataset}.csv")
    print("")
    print("  NOT ACQUIRED features (sorted by max probability):")
    print("    - not_acquired_MAXPROB_{trend}_{dataset}.csv")
    print("    - not_acquired_MAXPROB_ALL_TRENDS_{dataset}.csv")
    print("")
    print("  SUMMARY:")
    print("    - SUMMARY_TABLE_{dataset}.csv")

if __name__ == "__main__":
    main()