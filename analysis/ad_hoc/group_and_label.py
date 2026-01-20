import pandas as pd

# Load the feature data from 'features.csv'
df_features = pd.read_csv('/home/nsrikant/BehaviorBoxNew/analysis/rq1_qualitative_6400/acquired_increasing_Pythia-160m.csv')

# Display the first few rows of the DataFrame
print("First 5 rows of the feature data:")
print(df_features.head())

checkpoints = [
    "step1",
    "step2",
    "step4",
    "step8",
    "step16",
    "step32",
    "step64",
    "step128",
    "step256",
    "step512",
    "step1000",
    "step10000",
    "step70000",
    "step100000",
    "final"
]

print("Missing values before handling:")
print(df_features.isnull().sum())



# Check the data type of 'acq_checkpoint'
print(f"\nData type of 'acq_checkpoint': {df_features['acq_checkpoint'].dtype}")

# Convert 'acq_checkpoint' to numeric if it's not already
# Use errors='coerce' to turn any non-convertible values into NaN
df_features['acq_checkpoint'] = pd.to_numeric(df_features['acq_checkpoint'], errors='coerce')

# Check for any NaNs introduced by the conversion and handle them if necessary
if df_features['acq_checkpoint'].isnull().any():
    print("\nNaNs introduced in 'acq_checkpoint' after conversion. Dropping rows with NaN in this column.")
    df_features.dropna(subset=['acq_checkpoint'], inplace=True)
    df_features['acq_checkpoint'] = df_features['acq_checkpoint'].astype(int) # Convert to integer after dropping NaNs

print(f"\nNew data type of 'acq_checkpoint': {df_features['acq_checkpoint'].dtype}")
print(f"Number of rows after handling missing values in 'acq_checkpoint': {len(df_features)}")


df_features_sorted = df_features.sort_values(by='acq_checkpoint', ascending=True).reset_index(drop=True)
print("First 5 rows of sorted DataFrame by 'acq_checkpoint':")
print(df_features_sorted.head())



import numpy as np

# Create a mapping from the 1-based index of the 'checkpoints' list to its numerical value
# This assumes df_features['acq_checkpoint'] values are 1-based indices into the 'checkpoints' list
checkpoint_index_to_numeric_value = {}
for i, cp_name in enumerate(checkpoints):
    if cp_name.startswith('step'):
        checkpoint_index_to_numeric_value[i + 1] = int(cp_name[4:])
    elif cp_name == 'final':
        # Assign a value for 'final' that is distinct and clearly larger than any other step value
        checkpoint_index_to_numeric_value[i + 1] = 200000 

# Apply this mapping to create a new column 'numeric_acq_checkpoint' in df_features
df_features['numeric_acq_checkpoint'] = df_features['acq_checkpoint'].map(checkpoint_index_to_numeric_value)

# Handle any NaN values that might occur if an acq_checkpoint doesn't map to a known index
if df_features['numeric_acq_checkpoint'].isnull().any():
    print("Warning: Some 'acq_checkpoint' values did not map to a numeric checkpoint. These will be dropped or handled as 'unassigned'.")
    # For this task, we will assign them 'unassigned' and then drop for analysis clarity
    df_features['numeric_acq_checkpoint'].fillna(-1, inplace=True) # Use -1 for unassigned before dropping

# Sort by the new numeric checkpoint
df_features_sorted = df_features.sort_values(by='numeric_acq_checkpoint', ascending=True).reset_index(drop=True)

# Now define the new buckets based on the user's explicit request and the 'numeric_acq_checkpoint'
# early: 1-10000
# mid: 70000
# late: 100000 and final (using 200000 for final)

def assign_new_bucket(numeric_acq_checkpoint_value):
    if numeric_acq_checkpoint_value <= 10000:
        return 'early'
    elif numeric_acq_checkpoint_value == 70000:
        return 'mid'
    elif numeric_acq_checkpoint_value >= 100000: 
        return 'late'
    else:
        return 'unassigned' # For any values that don't fit the specified criteria

df_features_sorted['acquisition_bucket'] = df_features_sorted['numeric_acq_checkpoint'].apply(assign_new_bucket)

# Filter out any 'unassigned' features for the main analysis if they exist
df_features_sorted = df_features_sorted[df_features_sorted['acquisition_bucket'] != 'unassigned']

# Map the 'acquisition_bucket' back to the original df_features DataFrame
# Drop existing 'acquisition_bucket' and 'numeric_acq_checkpoint' if they exist to avoid conflicts
df_features = df_features.drop(columns=['acquisition_bucket', 'numeric_acq_checkpoint'], errors='ignore')
df_features = df_features.merge(df_features_sorted[['feature_group_id', 'acquisition_bucket', 'numeric_acq_checkpoint']], on='feature_group_id', how='left')

print("Distribution of features across new buckets:")
print(df_features['acquisition_bucket'].value_counts())
print("\nFirst 5 rows of df_features with new 'acquisition_bucket' and 'numeric_acq_checkpoint':")
print(df_features[['feature_group_id', 'acquisition_bucket', 'numeric_acq_checkpoint', 'description']].head())

print("\nDistribution of numeric_acq_checkpoint values in each bucket (unique values):")
for bucket in ['early', 'mid', 'late']:
    bucket_values = df_features[df_features['acquisition_bucket'] == bucket]['numeric_acq_checkpoint'].unique()
    if len(bucket_values) > 0:
        print(f"{bucket.capitalize()} bucket numeric checkpoints: {np.sort(bucket_values)}")
    else:
        print(f"{bucket.capitalize()} bucket: No features assigned.")


import litellm
from litellm import completion
import os


# Ensure the OpenAI API key is set from Colab secrets
LITELLM_API_KEY = "sk-XDzLzqJQ2IzJ4z6eZAtSHA"
LITELLM_BASE_URL = "https://cmu.litellm.ai"
labeling_model = "neulab/claude-sonnet-4-20250514"

print(LITELLM_BASE_URL)
# Define a function to get semantic labels using an LLM
def get_semantic_labels_with_llm(descriptions, bucket_name, max_retries=3):
    # Join descriptions to send to LLM
    descriptions_str = '\n'.join([f"- {d}" for d in descriptions])
    
    # Craft the prompt for the LLM
    # We'll ask it to provide a concise label for each description within the context of the bucket.
    # We want it to return a JSON array where each element is the label for the corresponding description.
    # Example: ["label1", "label2", "label1"]
    prompt = f"""Given the following list of feature descriptions from the '{bucket_name}' acquisition bucket, provide a very concise semantic category label (max 3 words) for each description. The labels should reflect the core semantic theme of each description. If a description stands alone, provide a unique but relevant label for it. Ensure the order of labels corresponds exactly to the order of the input descriptions. Return your response as a JSON array of strings. Do NOT include any additional text or formatting outside the JSON array. \n\nDescriptions:\n{descriptions_str}\n\nExample Expected Output: ["code formatting", "punctuation", "programming keywords", ...]"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that categorizes feature descriptions into concise semantic labels."},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            # Make the LLM call using litellm
            response = completion(
                api_key=LITELLM_API_KEY,
                base_url=LITELLM_BASE_URL,
                model="litellm_proxy/"+labeling_model,
                messages=messages,
            )

            # Attempt to parse the JSON response
            # Note: The model might embed the JSON string within a larger string or markdown block.
            # We need to extract the raw JSON string first.
            response_content = response.choices[0].message.content.strip()
            
            # LLM response_format='json_object' will wrap the array in an object, so we adjust.
            # For example: {'labels': ['label1', 'label2']}
            # We need to guide the LLM to put the array directly, or handle the object wrapping.
            # Let's adjust the prompt to expect the array directly if possible, or parse generically.
            import json
            try:
                parsed_response = json.loads(response_content)
                # If the LLM wraps it in an object, try to extract the array
                if isinstance(parsed_response, dict) and 'labels' in parsed_response:
                    return parsed_response['labels']
                elif isinstance(parsed_response, list):
                    return parsed_response
                else:
                    print(f"Warning: Unexpected JSON format from LLM: {parsed_response}")
                    continue
            except json.JSONDecodeError:
                print(f"JSON Decode Error on attempt {attempt+1}: Could not parse LLM response as JSON. Response: {response_content[:200]}...")
                continue

        except Exception as e:
            print(f"LiteLLM API call failed on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("Max retries reached. Returning empty list.")
                return [f"{bucket_name}_unlabeled"] * len(descriptions)
                
    return [f"{bucket_name}_unlabeled"] * len(descriptions)

semantic_subgroup_data = []

for bucket_name in df_features['acquisition_bucket'].unique():
    print(f"\nProcessing bucket: {bucket_name}")

    # Filter features for the current bucket
    bucket_features = df_features[df_features['acquisition_bucket'] == bucket_name].copy()

    # Get the descriptions for the current bucket
    descriptions_for_llm = bucket_features['description'].tolist()

    if not descriptions_for_llm:
        print(f"No features in {bucket_name} bucket. Skipping semantic grouping.")
        continue

    print(f"Sending {len(descriptions_for_llm)} descriptions to LLM for '{bucket_name}' bucket...")
    
    # Get semantic labels using the LLM function
    llm_labels = get_semantic_labels_with_llm(descriptions_for_llm, bucket_name)

    # Ensure the number of labels matches the number of descriptions
    if len(llm_labels) != len(descriptions_for_llm):
        print(f"Warning: LLM returned {len(llm_labels)} labels for {len(descriptions_for_llm)} descriptions in '{bucket_name}' bucket. Assigning generic labels.")
        # Fallback if LLM output is malformed or mismatched
        llm_labels = [f"{bucket_name}_fallback_label"] * len(descriptions_for_llm)

    # Assign resulting LLM labels, prefixed with the bucket name for uniqueness
    # We'll use a simple sequential number for the 'subgroup_id' within each LLM-generated category
    bucket_features['semantic_subgroup_raw'] = llm_labels
    
    # Further process LLM labels to ensure distinct, usable subgroup names and numeric IDs
    # Group by the raw LLM label and assign sequential subgroup IDs
    unique_llm_labels = bucket_features['semantic_subgroup_raw'].unique()
    label_to_id_map = {label: i for i, label in enumerate(unique_llm_labels)}
    
    bucket_features['semantic_subgroup'] = bucket_features['semantic_subgroup_raw'].apply(
        lambda x: f"{bucket_name}_{label_to_id_map.get(x, 'unassigned')}"
    )

    # Store feature_group_id and semantic_subgroup for merging later
    semantic_subgroup_data.extend(bucket_features[['feature_group_id', 'semantic_subgroup']].to_dict(orient='records'))

# Convert the list of dictionaries to a DataFrame
df_semantic_subgroups = pd.DataFrame(semantic_subgroup_data)

# Before merging, drop the 'semantic_subgroup' column if it already exists in df_features
# This prevents MergeError on subsequent runs
df_features = df_features.drop(columns=['semantic_subgroup', 'semantic_subgroup_raw'], errors='ignore')

# Merge the semantic subgroup labels back into the original df_features DataFrame
df_features = df_features.merge(df_semantic_subgroups, on='feature_group_id', how='left')

print("\nSemantic grouping with LiteLLM complete. First 5 rows of df_features with 'semantic_subgroup':")
print(df_features[['feature_group_id', 'acquisition_bucket', 'description', 'semantic_subgroup']].head())
print("\nValue counts of semantic_subgroup:")
print(df_features['semantic_subgroup'].value_counts())

