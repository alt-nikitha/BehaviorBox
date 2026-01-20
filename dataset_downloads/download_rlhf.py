"""
Download and process Anthropic HH Golden dataset.

Downloads 500 random samples from the test split, extracts good/bad responses,
shuffles, and saves in JSONL format.

Usage:
    python anthropic_hh_pipeline.py

Output files:
    - anthropic_hh_samples.json       : Original format with pairs
    - anthropic_hh_samples.jsonl      : Shuffled responses with metadata
    - anthropic_hh_minimal.jsonl      : Shuffled responses with only id and text
"""

from datasets import load_dataset
import random
import json

# Set seed for reproducibility
random.seed(42)


def download_anthropic_hh_samples(n_samples=500):
    """Download random samples from Anthropic HH Golden test split."""
    
    print("=" * 70)
    print("STEP 1: Downloading Anthropic HH Golden samples")
    print("=" * 70)
    
    # Load the test split
    print("\nLoading dataset 'Unified-Language-Model-Alignment/Anthropic_HH_Golden'...")
    print("Split: test")
    
    try:
        dataset = load_dataset("Unified-Language-Model-Alignment/Anthropic_HH_Golden", split='test')
        total_rows = len(dataset)
        print(f"✓ Loaded dataset with {total_rows} rows")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return []
    
    # Sample random indices
    print(f"\nSampling {n_samples} random rows...")
    if total_rows >= n_samples:
        sampled_indices = random.sample(range(total_rows), n_samples)
    else:
        print(f"⚠ Only {total_rows} rows available (< {n_samples}), taking all")
        sampled_indices = list(range(total_rows))
    
    print(f"✓ Sampled {len(sampled_indices)} indices")
    
    # Extract samples
    print("\nExtracting chosen (good) and rejected (bad) responses...")
    all_samples = []
    
    for i, idx in enumerate(sampled_indices):
        sample = dataset[idx]
        
        # The dataset has 'chosen' (good) and 'rejected' (bad) responses
        all_samples.append({
            'index': idx,
            'prompt': sample.get('prompt', ''),
            'response_good': sample.get('chosen', ''),
            'response_bad': sample.get('rejected', ''),
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sampled_indices)} samples...")
    
    print(f"\n{'='*70}")
    print(f"Extracted: {len(all_samples)} samples")
    print(f"Total responses: {len(all_samples) * 2} (good + bad)")
    print(f"{'='*70}\n")
    
    # Save original format
    # print("Saving to 'anthropic_hh_samples.json'...")
    # with open('anthropic_hh_samples.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_samples, f, indent=2, ensure_ascii=False)
    # print("✓ Saved\n")
    
    # Show sample
    print("=" * 70)
    print("Sample entry")
    print("=" * 70)
    if all_samples:
        sample = all_samples[0]
        print(f"\nPrompt: {sample['prompt'][:200]}...")
        print(f"\nGood response: {sample['response_good'][:200]}...")
        print(f"\nBad response: {sample['response_bad'][:200]}...")
    
    return all_samples


def convert_and_shuffle(samples):
    """Convert pairs to individual responses, shuffle, and save as JSONL."""
    
    print("\n" + "=" * 70)
    print("STEP 2: Converting to JSONL format and shuffling")
    print("=" * 70)
    
    # Create full entries (with metadata)
    full_entries = []
    minimal_entries = []
    
    print(f"\nCreating individual response entries...")
    
    for idx, sample in enumerate(samples):
        # Good response
        full_entries.append({
            'id': f"hh_good_{idx}",
            'text': sample['response_good'],
            'label': 'good',
            'prompt': sample['prompt'],
            'pair_id': idx,
            'original_index': sample['index']
        })
        minimal_entries.append({
            'id': f"hh_good_{idx}",
            'text': sample['response_good']
        })
        
        # Bad response
        full_entries.append({
            'id': f"hh_bad_{idx}",
            'text': sample['response_bad'],
            'label': 'bad',
            'prompt': sample['prompt'],
            'pair_id': idx,
            'original_index': sample['index']
        })
        minimal_entries.append({
            'id': f"hh_bad_{idx}",
            'text': sample['response_bad']
        })
    
    print(f"✓ Created {len(full_entries)} response entries")
    
    # Shuffle both lists
    print(f"\nShuffling {len(full_entries)} entries...")
    random.shuffle(full_entries)
    random.shuffle(minimal_entries)
    print("✓ Shuffled")
    
    # Save full version
    print("\nSaving 'anthropic_hh_samples.jsonl' (with metadata)...")
    with open('anthropic_hh_samples.jsonl', 'w', encoding='utf-8') as f:
        for entry in full_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(full_entries)} entries")
    
    # Save minimal version
    # print("\nSaving 'anthropic_hh_minimal.jsonl' (id + text only)...")
    # with open('anthropic_hh_minimal.jsonl', 'w', encoding='utf-8') as f:
    #     for entry in minimal_entries:
    #         f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    # print(f"✓ Saved {len(minimal_entries)} entries")
    
    # Statistics
    good_count = sum(1 for e in full_entries if e['label'] == 'good')
    bad_count = sum(1 for e in full_entries if e['label'] == 'bad')
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total entries: {len(full_entries)}")
    print(f"  - Good responses: {good_count}")
    print(f"  - Bad responses:  {bad_count}")
    print(f"Number of conversation pairs: {len(samples)}")
    
    # Show examples from full version
    print("\n" + "=" * 70)
    print("Sample entries (after shuffling)")
    print("=" * 70)
    for i in range(min(3, len(full_entries))):
        e = full_entries[i]
        print(f"\n[{i+1}] {e['id']} ({e['label']})")
        print(f"    Prompt: {e['prompt'][:100]}...")
        print(f"    Response: {e['text'][:150]}...")
    
    # Show examples from minimal version
    print("\n" + "=" * 70)
    print("Sample minimal entries")
    print("=" * 70)
    for i in range(min(2, len(minimal_entries))):
        e = minimal_entries[i]
        text_preview = e['text'][:100] + '...' if len(e['text']) > 100 else e['text']
        print(f"\n[{i+1}] {{'id': '{e['id']}', 'text': '{text_preview}'}}")
    
    return full_entries, minimal_entries


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Anthropic HH Golden Dataset Pipeline")
    print("=" * 70)
    print("This script will:")
    print("  1. Download 500 random samples from the test split")
    print("  2. Extract good (chosen) and bad (rejected) responses")
    print("  3. Shuffle all responses")
    print("  4. Save in multiple formats")
    print("=" * 70 + "\n")
    
    # Step 1: Download
    samples = download_anthropic_hh_samples(n_samples=500)
    
    if not samples:
        print("\n✗ No samples downloaded. Exiting.")
        exit(1)
    
    # Step 2: Convert and shuffle
    full_entries, minimal_entries = convert_and_shuffle(samples)
    
    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE! Generated files:")
    print("=" * 70)
    # print("  1. anthropic_hh_samples.json       - Original format (pairs)")
    print("  2. anthropic_hh_samples.jsonl      - Shuffled with metadata")
    # print("  3. anthropic_hh_minimal.jsonl      - Shuffled (id + text only)")
    print("=" * 70)
    print("\nRecommendation: Use 'anthropic_hh_minimal.jsonl' for BehaviorBox")
    print("=" * 70 + "\n")