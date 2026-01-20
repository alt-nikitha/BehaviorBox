"""
All-in-one script: Download BLiMP samples, shuffle, and convert to JSONL format.

Usage:
    python blimp_pipeline.py

Output files:
    - blimp_samples.json       : Original format with pairs
    - blimp_samples.jsonl      : Shuffled sentences with metadata (id, text, label, split)
    - blimp_minimal.jsonl      : Shuffled sentences with only id and text
"""

from datasets import load_dataset
import random
import json

# Set seed for reproducibility
random.seed(42)

# All 67 BLiMP phenomena
BLIMP_CONFIGS = [
    'adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement',
    'animate_subject_passive', 'animate_subject_trans', 'causative',
    'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch',
    'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1',
    'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2',
    'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun',
    'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1',
    'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1',
    'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising',
    'inchoative', 'intransitive', 'irregular_past_participle_adjectives',
    'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1',
    'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question',
    'left_branch_island_simple_question', 'matrix_question_npi_licensor_present',
    'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope',
    'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1',
    'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2',
    'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present',
    'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1',
    'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2',
    'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap',
    'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap',
    'wh_vs_that_with_gap_long_distance'
]


def download_blimp_samples():
    """Download 10 random pairs from each of the 67 BLiMP configurations."""
    
    print("=" * 70)
    print("STEP 1: Downloading BLiMP samples")
    print("=" * 70)
    print(f"\nLoading {len(BLIMP_CONFIGS)} BLiMP configurations...\n")
    
    all_samples = []
    
    for i, config in enumerate(BLIMP_CONFIGS, 1):
        try:
            # Load configuration
            ds = load_dataset("blimp", config, split='train')
            n = len(ds)
            
            # Sample 10 indices
            # indices = random.sample(range(n), min(10, n))
            # Sample 100 indices
            indices = random.sample(range(n), min(100, n))
            # indices = list(range(n))
            
            # Extract samples
            for idx in indices:
                all_samples.append({
                    'split': config,
                    'sentence_good': ds[idx]['sentence_good'],
                    'sentence_bad': ds[idx]['sentence_bad'],
                })
            
            print(f"[{i:2d}/67] ✓ {config:<50} ({len(indices)} samples)")
            
        except Exception as e:
            print(f"[{i:2d}/67] ✗ {config:<50} Error: {str(e)[:40]}")
    
    print(f"\n{'='*70}")
    print(f"Downloaded: {len(all_samples)} pairs = {len(all_samples)*2} sentences")
    print(f"{'='*70}\n")
    
    # # Save original format
    # with open('blimp_samples.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    # print("✓ Saved to 'blimp_samples.json'\n")
    
    return all_samples


def convert_and_shuffle(samples):
    """Convert pairs to individual sentences, shuffle, and save as JSONL."""
    
    print("=" * 70)
    print("STEP 2: Converting to JSONL format and shuffling")
    print("=" * 70)
    
    # Create full entries (with metadata)
    full_entries = []
    minimal_entries = []
    
    for idx, sample in enumerate(samples):
        split = sample['split']
        
        # Good sentence
        full_entries.append({
            'id': f"{split}_good_{idx}",
            'text': sample['sentence_good'],
            'label': 'good',
            'split': split,
            'pair_id': idx
        })
        minimal_entries.append({
            'id': f"{split}_good_{idx}",
            'text': sample['sentence_good']
        })
        
        # Bad sentence
        full_entries.append({
            'id': f"{split}_bad_{idx}",
            'text': sample['sentence_bad'],
            'label': 'bad',
            'split': split,
            'pair_id': idx
        })
        minimal_entries.append({
            'id': f"{split}_bad_{idx}",
            'text': sample['sentence_bad']
        })
    
    # Shuffle both lists with same random state
    print(f"\nShuffling {len(full_entries)} entries...")
    random.shuffle(full_entries)
    random.shuffle(minimal_entries)
    
    # Save full version
    print("\nSaving 'blimp_samples_100.jsonl' (with metadata)...")
    with open('blimp_samples_100.jsonl', 'w', encoding='utf-8') as f:
        for entry in full_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(full_entries)} entries")
    
    # # Save minimal version
    # print("\nSaving 'blimp_minimal.jsonl' (id + text only)...")
    # with open('blimp_minimal.jsonl', 'w', encoding='utf-8') as f:
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
    print(f"  - Good sentences: {good_count}")
    print(f"  - Bad sentences:  {bad_count}")
    print(f"Number of splits: {len(set(e['split'] for e in full_entries))}")
    
    # Show examples from full version
    print("\n" + "=" * 70)
    print("Sample entries (after shuffling)")
    print("=" * 70)
    for i in range(min(5, len(full_entries))):
        e = full_entries[i]
        print(f"\n[{i+1}] {e['id']} ({e['label']})")
        print(f"    {e['text']}")
    
    # Show examples from minimal version
    print("\n" + "=" * 70)
    print("Sample minimal entries")
    print("=" * 70)
    for i in range(min(3, len(minimal_entries))):
        e = minimal_entries[i]
        print(f"\n[{i+1}] {json.dumps(e, ensure_ascii=False)}")
    
    return full_entries, minimal_entries


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BLiMP Dataset Pipeline")
    print("=" * 70)
    print("This script will:")
    print("  1. Download 10 random pairs from each of 67 BLiMP splits")
    print("  2. Shuffle all sentences")
    # print("  3. Save in multiple formats")
    print("=" * 70 + "\n")
    
    # Step 1: Download
    samples = download_blimp_samples()
    
    # Step 2: Convert and shuffle
    full_entries, minimal_entries = convert_and_shuffle(samples)
    
    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE! Generated files:")
    print("=" * 70)
    # print("  1. blimp_samples.json       - Original format (pairs)")
    print("  2. blimp_samples_100.jsonl      - Shuffled with metadata")
    # print("  3. blimp_minimal.jsonl      - Shuffled (id + text only)")
    print("=" * 70)
    print("\nRecommendation: Use 'blimp_minimal_full.jsonl' for BehaviorBox analysis")
    print("=" * 70 + "\n")
