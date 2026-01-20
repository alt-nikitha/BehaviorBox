import json
import os
from litellm import completion
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
# PYTHIA_MODEL = "160m"  # "160m" or "6.9b"


# Ensure the OpenAI API key is set from Colab secrets
API_KEY = "sk-XDzLzqJQ2IzJ4z6eZAtSHA"
BASE_URL = "https://cmu.litellm.ai"
MODEL = "litellm_proxy/neulab/claude-sonnet-4-20250514"
BATCH_SIZE = 100

np.random.seed(SEED)

import json
import os
from litellm import completion


# ============================================================================
# SETUP DIRECTORIES
# ============================================================================

topics_dir = f"/home/nsrikant/BehaviorBoxNew/analysis/topics"
os.makedirs(topics_dir, exist_ok=True)

print(f"Topics directory: {topics_dir}")

# ============================================================================
# LOAD DATA FOR BOTH MODELS
# ============================================================================

print("\nLoading data for both models...")

data_160m = None
data_6_9b = None

json_path_160m = f"/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_blimp_full/Pythia-piletrainedonpile.json"
json_path_6_9b = f"/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_blimp_full/Pythia6.9b-piletrainedonpile.json"

try:
    print(f"Loading 160m from {json_path_160m}...")
    with open(json_path_160m, 'r') as f:
        data_160m = json.load(f)
    print(f"✓ Loaded {len(data_160m['features'])} features for 160m")
except FileNotFoundError:
    print(f"✗ File not found: {json_path_160m}")

try:
    print(f"Loading 6.9b from {json_path_6_9b}...")
    with open(json_path_6_9b, 'r') as f:
        data_6_9b = json.load(f)
    print(f"✓ Loaded {len(data_6_9b['features'])} features for 6.9b")
except FileNotFoundError:
    print(f"✗ File not found: {json_path_6_9b}")

if data_160m is None or data_6_9b is None:
    print("ERROR: Could not load one or both model data files")
    exit(1)

# ============================================================================
# COMBINE FEATURES FROM BOTH MODELS
# ============================================================================

print("\nCombining features from both models...")

# Create feature list with model source info
combined_features = []

# Add 160m features
for feature in data_160m['features']:
    combined_features.append({
        'feature_id': feature.get('feature_id'),
        'description': feature.get('description', ''),
        'model': '160m',
        'pearson': feature.get('pearson_corr'),
    })

# Add 6.9b features (with unique ID to avoid conflicts)
for feature in data_6_9b['features']:
    combined_features.append({
        'feature_id': f"6.9b_{feature.get('feature_id')}",
        'original_id': feature.get('feature_id'),
        'description': feature.get('description', ''),
        'model': '6.9b',
        'pearson': feature.get('pearson_corr'),
    })

print(f"Combined total: {len(combined_features)} features")
print(f"  - 160m: {len(data_160m['features'])} features")
print(f"  - 6.9b: {len(data_6_9b['features'])} features")

# ============================================================================
# LOAD OR CREATE COMMON TOPICS
# ============================================================================

common_topics_file = f"{topics_dir}/common_topics.json"
feature_topics = {}

print(f"\nLooking for existing topics in {common_topics_file}...")
if os.path.exists(common_topics_file):
    print(f"Loading existing topics...")
    with open(common_topics_file, 'r') as f:
        feature_topics = json.load(f)
    print(f"✓ Loaded {len(feature_topics)} existing topic assignments")
else:
    print("No existing topics found. Starting fresh.")
    feature_topics = {}

# ============================================================================
# ITERATIVELY GENERATE COMMON TOPIC LABELS (LLM)
# ============================================================================

print(f"\n" + "="*80)
print("Generating common topic labels for both models...")
print("="*80)

# Get unprocessed features
unprocessed_features = [f for f in combined_features if f['feature_id'] not in feature_topics]
print(f"Features to process: {len(unprocessed_features)}")

if len(unprocessed_features) > 0:
    # Process in batches
    for batch_start in range(0, len(unprocessed_features), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(unprocessed_features))
        batch = unprocessed_features[batch_start:batch_end]
        
        # Create prompt for this batch
        feature_list = "\n".join([
            f"[{f['feature_id']}] ({f['model']}) {f['description']}"
            for f in batch
        ])
        
        # Build context from existing topics
        existing_topics_context = ""
        if feature_topics:
            existing_topics_context = "\n\nEXISTING COMMON TOPICS (for consistency):\n"
            existing_unique = list(set(feature_topics.values()))[:10]
            for topic_name in existing_unique:
                existing_topics_context += f"- {topic_name}\n"
        
        prompt = f"""Analyze these neural network feature descriptions from two different model sizes (160m and 6.9b) and assign them to semantic topics.

The goal is to create COMMON topics that apply across both models when features are semantically similar. Features from different models that describe the same semantic concept should get the same topic.

Each feature has:
- A feature ID (includes model name: 160m or 6.9b)
- A model source (160m or 6.9b)
- A description

For each feature, assign it to a semantic topic that:
- Captures the MEANING of the feature description
- Is SHARED across models when describing similar concepts
- Is specific and descriptive
- Remains CONSISTENT across different model versions when the features are similar{existing_topics_context}

FEATURES:
{feature_list}

For each feature, respond with ONLY this format (one per line):
[feature_id] TOPIC_NAME

Where TOPIC_NAME is a semantic category (create new topics as needed, but reuse topics for similar features across models).
Examples of good common topics: "Syntax and Punctuation", "Mathematical Operations", "Conversational Markers", etc.
Be specific and use natural language for topic names.
"""
        
        print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1} (features {batch_start+1}-{batch_end})...")
        
        try:
            response = completion(
                model=MODEL,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                api_base=BASE_URL,
                api_key=API_KEY
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            for line in response_text.split('\n'):
                line = line.strip()
                if '[' in line and ']' in line and ' ' in line:
                    try:
                        # Extract feature_id and topic
                        parts = line.split('] ', 1)
                        if len(parts) == 2:
                            feature_id = parts[0].replace('[', '')
                            topic = parts[1].strip()
                            feature_topics[feature_id] = topic
                            print(f"  [{feature_id}] -> {topic}")
                    except:
                        pass
        
        except Exception as e:
            print(f"Error processing batch: {e}")

# ============================================================================
# CONSOLIDATE SIMILAR TOPICS
# ============================================================================

print(f"\n" + "="*80)
print("Consolidating similar topics...")
print("="*80)

# Find similar topics using simple heuristics
from difflib import SequenceMatcher

def similarity_ratio(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Group topics by similarity
unique_topics = list(set(feature_topics.values()))
topic_mapping = {}  # Maps original topic to consolidated topic

for topic in unique_topics:
    if topic in topic_mapping:
        continue
    
    # Find similar topics (threshold: 0.7 similarity)
    similar_topics = [t for t in unique_topics if t != topic and similarity_ratio(topic, t) > 0.7]
    
    if similar_topics:
        # Use the longest/most descriptive topic name as the canonical one
        group = [topic] + similar_topics
        canonical = max(group, key=len)
        
        print(f"\nMerging similar topics:")
        for t in group:
            topic_mapping[t] = canonical
            if t != canonical:
                print(f"  '{t}' -> '{canonical}'")
    else:
        topic_mapping[topic] = topic

# Apply consolidation to all features
print(f"\nApplying consolidation...")
consolidated_topics = {}
for feature_id, topic in feature_topics.items():
    consolidated_topics[feature_id] = topic_mapping.get(topic, topic)

print(f"Consolidated from {len(set(feature_topics.values()))} to {len(set(consolidated_topics.values()))} unique topics")

feature_topics = consolidated_topics

print(f"\n\nSaving common topics to {topics_dir}...")

with open(common_topics_file, 'w') as f:
    json.dump(feature_topics, f, indent=2)
print(f"✓ Saved common topics to {common_topics_file}")

# ============================================================================
# ADD TOPICS TO ORIGINAL DATA FOR EACH MODEL
# ============================================================================

print(f"\nAdding topics to individual model data...")

# Process 160m
pythia_160m_dir = f"{topics_dir}/Pythia160m"
os.makedirs(pythia_160m_dir, exist_ok=True)

combined_160m = data_160m.copy()
for feature in combined_160m['features']:
    feature_id = feature['feature_id']
    # Try int first, then str
    fid = str(feature_id) if isinstance(feature_id, int) else feature_id
    topic = feature_topics.get(fid, 'Other')
    feature['topic'] = topic

combined_160m_file = f"{pythia_160m_dir}/data_with_topics.json"
with open(combined_160m_file, 'w') as f:
    json.dump(combined_160m, f, indent=2)
print(f"✓ Saved 160m with topics to {combined_160m_file}")

# Process 6.9b
pythia_6_9b_dir = f"{topics_dir}/Pythia6.9b"
os.makedirs(pythia_6_9b_dir, exist_ok=True)

combined_6_9b = data_6_9b.copy()
for feature in combined_6_9b['features']:
    feature_id = feature['feature_id']
    # Map back to common feature ID format (6.9b_<id>)
    fid_common = f"6.9b_{feature_id}"
    topic = feature_topics.get(fid_common, 'Other')
    feature['topic'] = topic

combined_6_9b_file = f"{pythia_6_9b_dir}/data_with_topics.json"
with open(combined_6_9b_file, 'w') as f:
    json.dump(combined_6_9b, f, indent=2)
print(f"✓ Saved 6.9b with topics to {combined_6_9b_file}")

# ============================================================================
# VERIFY TOPIC CONSISTENCY
# ============================================================================

print("\n" + "="*80)
print("VERIFYING TOPIC CONSISTENCY")
print("="*80)

# Load both saved files
with open(combined_160m_file, 'r') as f:
    verify_160m = json.load(f)
with open(combined_6_9b_file, 'r') as f:
    verify_6_9b = json.load(f)

topics_160m = set(f['topic'] for f in verify_160m['features'])
topics_6_9b = set(f['topic'] for f in verify_6_9b['features'])

print(f"Topics in 160m: {len(topics_160m)}")
print(f"Topics in 6.9b: {len(topics_6_9b)}")

common_topics_verified = topics_160m & topics_6_9b
print(f"Common topics across both: {len(common_topics_verified)}")

only_160m = topics_160m - topics_6_9b
only_6_9b = topics_6_9b - topics_160m

if only_160m:
    print(f"\n⚠ Topics only in 160m: {only_160m}")
if only_6_9b:
    print(f"⚠ Topics only in 6.9b: {only_6_9b}")
else:
    print(f"\n✓ All topics are consistent across both models!")

# ============================================================================
# SAVE MAPPING FOR REFERENCE
# ============================================================================

print(f"\nSaving topic mapping and statistics...")

# Create topic statistics
topic_stats = {}
for feature in combined_features:
    topic = feature_topics.get(feature['feature_id'], 'Other')
    if topic not in topic_stats:
        topic_stats[topic] = {'160m': 0, '6.9b': 0, 'total': 0}
    
    topic_stats[topic][feature['model']] += 1
    topic_stats[topic]['total'] += 1

stats_file = f"{topics_dir}/topic_statistics.json"
with open(stats_file, 'w') as f:
    json.dump(topic_stats, f, indent=2)
print(f"✓ Saved topic statistics to {stats_file}")

# Print summary
print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total topics created: {len(set(feature_topics.values()))}")
print(f"Total features assigned: {len(feature_topics)}")
print(f"\nTopic distribution:")
for topic in sorted(topic_stats.keys()):
    stats = topic_stats[topic]
    print(f"  {topic}: {stats['160m']} (160m) + {stats['6.9b']} (6.9b) = {stats['total']} total")

print(f"\n✓ Done! Files saved:")
print(f"  - {common_topics_file}")
print(f"  - {combined_160m_file}")
print(f"  - {combined_6_9b_file}")
print(f"  - {stats_file}")