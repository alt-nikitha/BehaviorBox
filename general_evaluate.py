#!/usr/bin/env python3
"""
general_evaluate.py

Evaluate BLiMP / multiple-choice style data from pre-computed logprobs stored in parquet/pickle format.

Supports:
 - One correct option and N wrong options.
 - Labels accepted: "good"/"bad" OR "correct"/"wrong" (both recognized).
 - Pre-builds indices for fast lookups.
 - Vectorized dictionary building for logprobs.
 - Progress bars for visibility.

Example:
 python general_evaluate.py \
    --data_path /path/to/blimp_samples.jsonl \
    --features_dir /path/to/features_dir \
    --model_name pythia-160m \
    --save_dir results/blimp \
    --metric seq_logprob
"""
import json
from pathlib import Path
from typing import Dict, List, Union, Tuple

import click
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
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

    # Vectorized-ish lookup via list comprehension (fast enough for typical sizes)
    try:
        logprobs = [word_id_to_logprob.get(int(wid)) for wid in word_ids]
    except:
        logprobs = [word_id_to_logprob.get(wid) for wid in word_ids]

    # Filter out None values and warn if any are missing (sampled warnings)
    valid_logprobs = [float(lp) for lp in logprobs if lp is not None]

    if len(valid_logprobs) != len(word_ids):
        missing_count = len(word_ids) - len(valid_logprobs)
        if missing_count > 0 and np.random.random() < 0.01:  # 1% sample to avoid spam
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
        # Return -inf for probability-like metrics (so missing tokens are very bad).
        # However earlier code used +inf; here we want a value that won't be counted as "best".
        # Use a very large negative number to indicate extremely low prob.
        return float("-1e9")

    logprobs_array = np.array(logprobs, dtype=np.float32)

    if metric == "seq_logprob":
        return float(logprobs_array.sum())
    elif metric == "normalized_seq_logprob":
        return float(logprobs_array.mean())
    elif metric == "lowest_token_logprob":
        # Interpret as the lowest single-token logprob (i.e., min)
        return float(logprobs_array.min())
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
    Supports one correct + N wrong options. Accepts labels:
      - correct labels: "good" or "correct"
      - wrong labels:   "bad"  or "wrong"
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

    # Group by pair_id and split; support both label schemes.
    print("\nGrouping sentences by pairs...")
    pairs = {}
    for item in data:
        
        doc_id = item.get('id')
        split = "_".join(doc_id.split("_")[:-3])
        label = doc_id.split("_")[-3]
        pair_id = doc_id.split("_")[-2]
        text = item.get('text', '')
        # print(label)
        
        if pair_id is None:
            continue

        key = (split, pair_id)
        if key not in pairs:
            pairs[key] = {"wrong": []}

        # Accept both "good"/"bad" and "correct"/"wrong"
        if "good" in label or "correct" in label:
            pairs[key]["correct"] = {'id': doc_id, 'text': text}
        elif label in ("bad", "wrong"):
            pairs[key]["wrong"].append({'id': doc_id, 'text': text})
        else:
            # If label is unknown, try to infer: if there is already a correct, treat others as wrong.
            # This is permissive; prefer explicit labels in data.
            if "correct" not in pairs[key]:
                pairs[key]["correct"] = {'id': doc_id, 'text': text}
            else:
                pairs[key]["wrong"].append({'id': doc_id, 'text': text})

    print(f"Found {len(pairs)} pairs")

    
    # Prepare results structure (generic names)
    results = {
        'split': [],
        'pair_id': [],
        'correct_sentence': [],
        'correct_doc_id': [],
        'correct_input_file': [],
        'correct_output_file': [],
        'correct_metric': [],
        'correct_num_tokens': [],
        'best_wrong_metric': [],
        'num_wrong_options': [],
        'wrong_metrics': [],  # list of per-option metrics
    }

    missing_count = 0
    processed_count = 0

    # Process pairs with progress bar
    for (split, pair_id), pair_data in tqdm(pairs.items(), desc="Processing pairs", total=len(pairs)):
        # Need exactly one correct and at least one wrong option
        
        if "correct" not in pair_data or len(pair_data.get("wrong", [])) == 0:
            
            missing_count += 1
            continue

        try:
            correct = pair_data["correct"]
            correct_id = correct['id']

            # Quick mapping existence checks
            if correct_id not in input_doc_to_file or correct_id not in output_doc_to_file:
                missing_count += 1
                continue
            # print(1)

            correct_input_file = input_doc_to_file[correct_id]
            correct_output_file = output_doc_to_file[correct_id]

            # Verify presence in prebuilt indices
            if correct_input_file not in input_indices or correct_id not in input_indices[correct_input_file]:
                missing_count += 1
                continue

            

            correct_word_ids = input_indices[correct_input_file][correct_id]
            
            correct_logprobs = get_logprobs_batch(correct_word_ids, correct_output_file, output_indices)
            
            correct_metric_val = compute_metrics_from_logprobs(correct_logprobs, metric)
            
            # Evaluate all wrong options
            wrong_metrics = []
            for w in pair_data["wrong"]:
                wid = w['id']
                # Ensure mapping exists
                if wid not in input_doc_to_file or wid not in output_doc_to_file:
                    continue

                w_input_file = input_doc_to_file[wid]
                w_output_file = output_doc_to_file[wid]

                # Ensure indices available
                if w_input_file not in input_indices or wid not in input_indices[w_input_file]:
                    continue

                w_word_ids = input_indices[w_input_file][wid]
                
                w_logprobs = get_logprobs_batch(w_word_ids, w_output_file, output_indices)
                w_metric = compute_metrics_from_logprobs(w_logprobs, metric)
                wrong_metrics.append(w_metric)

            if len(wrong_metrics) == 0:
                # No valid wrong option processed
                missing_count += 1
                continue
            
            # print(3)

            best_wrong_metric = float(np.max(np.array(wrong_metrics, dtype=np.float32)))

            # Store results
            results['split'].append(split)
            results['pair_id'].append(pair_id)
            results['correct_sentence'].append(correct.get('text', ''))
            results['correct_doc_id'].append(correct_id)
            results['correct_input_file'].append(correct_input_file)
            results['correct_output_file'].append(correct_output_file)
            results['correct_metric'].append(correct_metric_val)
            results['correct_num_tokens'].append(len(correct_logprobs))
            results['best_wrong_metric'].append(best_wrong_metric)
            results['num_wrong_options'].append(len(wrong_metrics))
            results['wrong_metrics'].append(wrong_metrics)

            processed_count += 1

        except Exception as e:
            # Only print the first few errors to avoid spamming
            if processed_count < 10:
                print(f"Error processing {split}, pair {pair_id}: {e}")
            missing_count += 1
            continue

    print(f"\nSuccessfully processed: {processed_count}")
    print(f"Missing/Failed: {missing_count}")

    return results


def compute_accuracy(results: pd.DataFrame) -> float:
    """
    Compute accuracy: percentage where correct option has strictly higher metric
    than the best wrong option.
    """
    if len(results) == 0:
        return 0.0
    correct_bool = (results['correct_metric'] > results['best_wrong_metric'])
    correct = correct_bool.sum()
    total = len(results)
    return 100.0 * correct / total


@click.command()
@click.option(
    "--data_path",
    help="Path to JSONL file with BLiMP subset",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--features_dir",
    help="Directory containing input_features and output_features subdirs",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--model_name",
    help="Name of the model (subdir under output_features)",
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
    dataset_name = Path(data_path).stem
    features_dir = Path(features_dir)
    input_features_dir = features_dir / "input_features"
    output_features_dir = features_dir / "output_features" / model_name

    input_file_to_doc_path = input_features_dir / "file_to_doc.csv"
    output_file_to_doc_path = output_features_dir / "file_to_doc.csv"

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
    accuracy = compute_accuracy(df)
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    # Compute per-split accuracy
    print("\nPer-split Accuracy:")
    print(f"{'Split':<40} {'Accuracy':<10} {'N'}")
    print("-" * 60)
    split_accuracies = {}
    for split in sorted(df['split'].unique()):
        split_df = df[df['split'] == split]
        split_acc = compute_accuracy(split_df)
        split_accuracies[split] = split_acc
        print(f"{split:<40} {split_acc:>6.2f}%    {len(split_df):>4}")

    # Save detailed results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(save_dir) / f"{dataset_name}_results_{metric}.csv"
    # Convert lists to JSON strings for CSV-friendly storage (wrong_metrics)
    df_to_save = df.copy()
    df_to_save['wrong_metrics'] = df_to_save['wrong_metrics'].apply(lambda x: json.dumps(x))
    df_to_save.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to {output_path}")

    # Save summary
    summary = {
        'overall_accuracy': accuracy,
        'per_split_accuracy': split_accuracies,
        'total_pairs': len(df),
        'metric': metric,
        'num_splits': len(split_accuracies)
    }
    summary_path = Path(save_dir) / f"{dataset_name}_summary_{metric}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")

    # Print some example predictions
    print("\nExample predictions:")
    example_df = df[[
        'split', 'pair_id',
        'correct_metric', 'best_wrong_metric',
        'correct_num_tokens', 'num_wrong_options'
    ]].head(10)
    print(example_df.to_string(index=False))


if __name__ == "__main__":
    main()

# python general_evaluate.py \
# --data_path "/data/user_data/nsrikant/bbox_data/data/mmlu_sample.jsonl" \
# --features_dir "/data/user_data/nsrikant/bbox_data/output/mmlu_sample" \
# --model_name pythia-160m \
# --save_dir results/mmlu \
# --metric seq_logprob
