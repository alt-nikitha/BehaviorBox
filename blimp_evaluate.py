"""
Optimized code to evaluate blimp from pre-computed logprobs stored in parquet/pickle format
Key optimizations:
1. Pre-build word_id to logprob dictionaries once per file
2. Pre-build doc_id indices for faster filtering
3. Use vectorized operations where possible
4. Add progress bars for better feedback
"""

import click
import json
import pandas as pd
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Union, Tuple
import numpy as np
from tqdm import tqdm


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_parquet_files_by_mapping(
    file_to_doc_df: pd.DataFrame,
    features_dir: str,
    file_type: str = "input"
) -> Dict[str, pd.DataFrame]:
    """
    Load all necessary parquet files based on file_to_doc mapping
    """
    features_dir = Path(features_dir)
    unique_files = file_to_doc_df['file'].unique()
    
    print(f"\nLoading {len(unique_files)} {file_type}_features files...")
    
    features_cache = {}
    for file in tqdm(unique_files, desc=f"Loading {file_type} files"):
        parquet_path = features_dir / file
        
        if not parquet_path.exists():
            print(f"Warning: File not found: {parquet_path}")
            continue
        
        try:
            df = pd.read_parquet(parquet_path)
            features_cache[file] = df
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    return features_cache


def build_input_indices(input_features_cache: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Pre-build indices for fast doc_id lookup in input features
    Returns: {filename: {doc_id: sorted_word_ids_array}}
    """
    print("\nBuilding input feature indices...")
    indices = {}
    
    for file_name, df in tqdm(input_features_cache.items(), desc="Building input indices"):
        file_index = {}
        # Group by doc_id once
        for doc_id, group in df.groupby('doc_id'):
            # Sort by word_id and extract as numpy array
            file_index[doc_id] = group.sort_values('word_id')['word_id'].values
        indices[file_name] = file_index
    
    return indices


def build_output_indices(output_features_cache: Dict[str, pd.DataFrame], 
                         logprob_column: str = "logprobs") -> Dict[str, Dict[int, float]]:
    """
    Pre-build word_id to logprob dictionaries for all output files
    Returns: {filename: {word_id: logprob}}
    """
    print("\nBuilding output feature indices...")
    indices = {}
    
    for file_name, df in tqdm(output_features_cache.items(), desc="Building output indices"):
        if logprob_column not in df.columns:
            raise ValueError(f"Column '{logprob_column}' not found in {file_name}. "
                           f"Available columns: {df.columns.tolist()}")
        
        # Build the dictionary once per file using vectorized operations
        indices[file_name] = dict(zip(df['word_id'].values, df[logprob_column].values))
    
    return indices


def get_logprobs_batch(
    word_ids: np.ndarray,
    output_file_name: str,
    output_indices: Dict[str, Dict[int, float]]
) -> List[float]:
    """
    Extract logprobs for given word_ids using pre-built index
    """
    if output_file_name not in output_indices:
        raise ValueError(f"File {output_file_name} not found in output_indices")
    
    word_id_to_logprob = output_indices[output_file_name]
    
    # Vectorized lookup
    logprobs = [word_id_to_logprob.get(wid) for wid in word_ids]
    
    # Filter out None values and warn if any are missing
    valid_logprobs = [lp for lp in logprobs if lp is not None]
    
    if len(valid_logprobs) != len(word_ids):
        missing_count = len(word_ids) - len(valid_logprobs)
        # Only print occasionally to avoid spam
        if missing_count > 0 and np.random.random() < 0.01:  # 1% sample
            print(f"Warning: {missing_count} word_ids not found in {output_file_name}")
    
    return valid_logprobs


def compute_metrics_from_logprobs(
    logprobs: List[float],
    metric: str = "seq_logprob"
) -> float:
    """
    Compute metric from pre-computed token logprobs
    """
    if len(logprobs) == 0:
        return float('inf')
    
    logprobs_array = np.array(logprobs, dtype=np.float32)
    
    if metric == "seq_logprob":
        return float(logprobs_array.sum())
    elif metric == "normalized_seq_logprob":
        return float(logprobs_array.mean())
    elif metric == "lowest_token_logprob":
        return float(logprobs_array.max())  # max NLL = lowest logprob
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def evaluate_blimp_from_files(
    data_path: str,
    input_file_to_doc_path: str,
    output_file_to_doc_path: str,
    input_features_dir: str,
    output_features_dir: str,
    metric: str = "seq_logprob",
    logprob_column: str = "logprobs"
) -> Dict[str, Union[int, float, List]]:
    """
    Evaluate BLiMP from pre-saved data and logprobs in parquet format (OPTIMIZED)
    """
    print("Loading data files...")
    
    # Load BLiMP data
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} BLiMP examples")
    
    # Load file to doc mappings
    input_file_to_doc_df = pd.read_csv(input_file_to_doc_path)
    output_file_to_doc_df = pd.read_csv(output_file_to_doc_path)
    print(f"Loaded mappings: {len(input_file_to_doc_df)} input, {len(output_file_to_doc_df)} output entries")
    
    # Load all necessary parquet files
    input_features_cache = load_parquet_files_by_mapping(
        input_file_to_doc_df, input_features_dir, "input"
    )
    
    output_features_cache = load_parquet_files_by_mapping(
        output_file_to_doc_df, output_features_dir, "output"
    )
    
    # PRE-BUILD INDICES (major optimization)
    input_indices = build_input_indices(input_features_cache)
    output_indices = build_output_indices(output_features_cache, logprob_column)
    
    # Create mappings from doc_id to file for quick lookup
    input_doc_to_file = dict(zip(input_file_to_doc_df['doc_id'], input_file_to_doc_df['file']))
    output_doc_to_file = dict(zip(output_file_to_doc_df['doc_id'], output_file_to_doc_df['file']))
    
    # Group by pair_id and split
    print("\nGrouping sentences by pairs...")
    pairs = {}
    for item in data:
        pair_id = item['pair_id']
        split = item['split']
        label = item['label']
        
        key = (split, pair_id)
        if key not in pairs:
            pairs[key] = {}
        
        pairs[key][label] = {
            'id': item['id'],
            'text': item['text']
        }
    
    print(f"Found {len(pairs)} pairs")
    
    # Compute metrics for each pair with progress bar
    results = {
        'split': [],
        'pair_id': [],
        'good_sentence': [],
        'bad_sentence': [],
        'good_doc_id': [],
        'bad_doc_id': [],
        'good_input_file': [],
        'bad_input_file': [],
        'good_output_file': [],
        'bad_output_file': [],
        f'good_{metric}': [],
        f'bad_{metric}': [],
        'good_num_tokens': [],
        'bad_num_tokens': []
    }
    
    missing_count = 0
    processed_count = 0
    
    # Process pairs with progress bar
    for (split, pair_id), pair_data in tqdm(pairs.items(), desc="Processing pairs", total=len(pairs)):
        if 'good' not in pair_data or 'bad' not in pair_data:
            missing_count += 1
            continue
        
        good_id = pair_data['good']['id']
        bad_id = pair_data['bad']['id']
        
        # Check if doc_ids exist in mappings
        if good_id not in input_doc_to_file or bad_id not in input_doc_to_file:
            missing_count += 1
            continue
        
        if good_id not in output_doc_to_file or bad_id not in output_doc_to_file:
            missing_count += 1
            continue
        
        good_input_file = input_doc_to_file[good_id]
        bad_input_file = input_doc_to_file[bad_id]
        good_output_file = output_doc_to_file[good_id]
        bad_output_file = output_doc_to_file[bad_id]
        
        try:
            # Get word_ids from pre-built indices (FAST!)
            if good_input_file not in input_indices or good_id not in input_indices[good_input_file]:
                missing_count += 1
                continue
            
            if bad_input_file not in input_indices or bad_id not in input_indices[bad_input_file]:
                missing_count += 1
                continue
            
            good_word_ids = input_indices[good_input_file][good_id]
            bad_word_ids = input_indices[bad_input_file][bad_id]
            
            # Get logprobs from pre-built indices (FAST!)
            good_logprobs = get_logprobs_batch(
                good_word_ids, good_output_file, output_indices
            )
            bad_logprobs = get_logprobs_batch(
                bad_word_ids, bad_output_file, output_indices
            )
            
            # Compute metrics
            good_metric_val = compute_metrics_from_logprobs(good_logprobs, metric)
            bad_metric_val = compute_metrics_from_logprobs(bad_logprobs, metric)
            
            results['split'].append(split)
            results['pair_id'].append(pair_id)
            results['good_sentence'].append(pair_data['good']['text'])
            results['bad_sentence'].append(pair_data['bad']['text'])
            results['good_doc_id'].append(good_id)
            results['bad_doc_id'].append(bad_id)
            results['good_input_file'].append(good_input_file)
            results['bad_input_file'].append(bad_input_file)
            results['good_output_file'].append(good_output_file)
            results['bad_output_file'].append(bad_output_file)
            results[f'good_{metric}'].append(good_metric_val)
            results[f'bad_{metric}'].append(bad_metric_val)
            results['good_num_tokens'].append(len(good_logprobs))
            results['bad_num_tokens'].append(len(bad_logprobs))
            
            processed_count += 1
                
        except Exception as e:
            if processed_count < 10:  # Only print first few errors
                print(f"Error processing {split}, pair {pair_id}: {e}")
            missing_count += 1
            continue
    
    print(f"\nSuccessfully processed: {processed_count}")
    print(f"Missing/Failed: {missing_count}")
    
    return results


def compute_accuracy(results: pd.DataFrame, metric: str) -> float:
    """
    Compute accuracy: percentage where good sentence has lower NLL (higher logprob)
    """
    correct = (results[f'good_{metric}'] > results[f'bad_{metric}']).sum()
    total = len(results)
    return 100 * correct / total


@click.command()
@click.option(
    "--data_path",
    help="Path to JSONL file with BLiMP subset",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--features_dir",
    help="Directory containing input and output features parquet files",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--model_name",
    help="Name of the model",
    type=str,
    required=True,
)
@click.option(
    "--save_dir",
    help="Directory to save results",
    type=click.Path(),
    required=True,
)
@click.option(
    "--metric",
    help="Metric to compute from logprobs",
    default="seq_logprob",
    type=click.Choice([
        "lowest_token_logprob",
        "seq_logprob",
        "normalized_seq_logprob",
    ]),
)
@click.option(
    "--logprob_column",
    help="Name of the logprob column in output features",
    default="logprobs",
    type=str,
)
def main(
    data_path: str,
    features_dir: str,
    model_name: str,
    save_dir: str,
    metric: str = "seq_logprob",
    logprob_column: str = "logprobs"
):
    dataset_name = data_path.split("/")[-1].split(".")[0]
    features_dir = Path(features_dir)
    input_features_dir = features_dir / "input_features"
    output_features_dir = features_dir / "output_features" / model_name
   
    input_file_to_doc_path = Path(input_features_dir) / "file_to_doc.csv"
    output_file_to_doc_path = Path(output_features_dir) / "file_to_doc.csv"

    save_dir = Path(save_dir) / dataset_name / model_name
    
    # Evaluate
    results = evaluate_blimp_from_files(
        data_path,
        input_file_to_doc_path,
        output_file_to_doc_path,
        input_features_dir,
        output_features_dir,
        metric,
        logprob_column
    )
    
    df = pd.DataFrame.from_dict(results)
    
    if len(df) == 0:
        print("\nNo results to save! Check your data files.")
        return
    
    # Compute overall accuracy
    accuracy = compute_accuracy(df, metric)
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Compute per-split accuracy
    print("\nPer-split Accuracy:")
    print(f"{'Split':<50} {'Accuracy':<10} {'N'}")
    print("-" * 70)
    split_accuracies = {}
    for split in sorted(df['split'].unique()):
        split_df = df[df['split'] == split]
        split_acc = compute_accuracy(split_df, metric)
        split_accuracies[split] = split_acc
        print(f"{split:<50} {split_acc:>6.2f}%    {len(split_df):>4}")
    
    # Save detailed results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(save_dir) / f"blimp_results_{metric}.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to {output_path}")
    
    # Save summary
    summary = {
        'overall_accuracy': accuracy,
        'per_split_accuracy': split_accuracies,
        'total_pairs': len(df),
        'metric': metric,
        'num_splits': len(split_accuracies)
    }
    summary_path = Path(save_dir) / f"blimp_summary_{metric}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
    
    # Print some example predictions
    print("\nExample predictions:")
    example_df = df[[
        'split', 'pair_id', 
        f'good_{metric}', f'bad_{metric}', 
        'good_num_tokens', 'bad_num_tokens'
    ]].head(10)
    print(example_df.to_string(index=False))


if __name__ == "__main__":
    main()

# python blimp_evaluate.py \
#     --data_path /home/nsrikant/BehaviorBoxNew/data/blimp_samples.jsonl \
#     --input_features_dir /home/nsrikant/bbox_outputs/output/blimp_samples/input_features \
#     --output_features_dir /home/nsrikant/bbox_outputs/output/blimp_samples/output_features/pythia-160m \
#     --save_dir results/blimp \
#     --metric seq_logprob


# python blimp_evaluate.py \
#     --data_path /home/nsrikant/BehaviorBoxNew/data/blimp_samples.jsonl \
#     --features_dir /home/nsrikant/bbox_outputs/output/blimp_samples \
#     --model_name pythia-160m-step1 \
#     --save_dir results \
#     --metric seq_logprob