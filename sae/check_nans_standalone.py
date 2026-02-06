
import dask.dataframe as dd
import dask.array as da
import numpy as np
import os
from dask.distributed import Client
from data_utils import load_dataframe, preprocess_data

def check_nans_standalone():
    # Hardcoded paths from config_n_moreearly_olmo3_7b.json
    data_dir = "/data/user_data/nsrikant/bbox_data/output/olmo_validation_texts"
    model_names = [
        "olmo-3-7b-stage1-step1000",
        "olmo-3-7b-stage1-step15000",
        "olmo-3-7b-stage1-step73000",
        "olmo-3-7b-stage1-step146000",
        "olmo-3-7b-stage1-step365000",
        "olmo-3-7b-stage1-step731000",
        "olmo-3-7b-stage1-step1096000",
        "olmo-3-7b-stage1-step1169000",
        "olmo-3-7b-stage1-step1315000",
        "olmo-3-7b-stage2-step18000",
        "olmo-3-7b-main"
    ]
    output_feature_weight = 0.7
    
    print("Initializing Dask Client...")
    # Use a local cluster
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='16GB')
    print(client)

    print(f"Checking data in {data_dir}...")
    
    input_feature_dir = f"{data_dir}/input_features"
    output_feature_dirs = [f"{data_dir}/output_features/{model_name}" for model_name in model_names]
    
    # 1. Load Dataframe pieces manually to check for source
    print("Checking individual model files for NaNs...")
    
    doc_ids = None # getting valid doc ids is complex, let's just peek at the raw files
    
    # helper to check a single directory
    def check_dir(dir_path, name):
        print(f"Checking {name} in {dir_path}...")
        try:
            ddf = dd.read_parquet(dir_path)
            # Check for NaNs in logprobs if present
            if "logprobs" in ddf.columns:
                 nan_count = ddf["logprobs"].isnull().sum().compute()
                 if nan_count > 0:
                     print(f"!!! NaNs found in raw parquet for {name}: {nan_count}")
                 else:
                     print(f"No NaNs in raw parquet for {name}")
            else:
                 print(f"Column 'logprobs' not found in {name}")
            
            return ddf
        except Exception as e:
            print(f"Error reading {name}: {e}")
            return None

    # Check a few models
    for model_name in model_names[:3]: # check first 3
         check_dir(f"{data_dir}/output_features/{model_name}", model_name)

    print("Checking if NaNs are introduced by alignement/concat...")
    # Load using the utils to mimic the real process which does filtering and indexing
    try:
        from data_utils import _get_valid_doc_ids
        doc_ids = _get_valid_doc_ids(input_feature_dir, output_feature_dirs)
        print(f"Found {len(doc_ids)} valid docs.")
        
        # Check index alignment for first two models
        m1_name = model_names[0]
        m2_name = model_names[1]
        
        m1_df = dd.read_parquet(f"{data_dir}/output_features/{m1_name}", filters=[("doc_id", 'in', doc_ids)])
        m1_df["word_id"] = m1_df["word_id"].astype(str)
        m1_df = m1_df.set_index("word_id")
        
        m2_df = dd.read_parquet(f"{data_dir}/output_features/{m2_name}", filters=[("doc_id", 'in', doc_ids)])
        m2_df["word_id"] = m2_df["word_id"].astype(str)
        m2_df = m2_df.set_index("word_id")
        
        print("Checking index lengths...")
        l1 = len(m1_df.index)
        l2 = len(m2_df.index)
        print(f"{m1_name} length: {l1}")
        print(f"{m2_name} length: {l2}")
        
        if l1 != l2:
            print("!!! Length mismatch! Concat will generate NaNs.")

        # Check lengths of ALL models
        print("Checking lengths of ALL models...")
        lengths = {}
        for name in model_names: # Check all
             print(f"Checking length of {name}...")
             m_df = dd.read_parquet(f"{data_dir}/output_features/{name}", filters=[("doc_id", 'in', doc_ids)])
             lengths[name] = len(m_df.index)
             print(f"{name}: {lengths[name]}")
        
        # Check if all lengths are equal
        first_len = list(lengths.values())[0]
        mismatch = False
        for name, l in lengths.items():
            if l != first_len:
                print(f"!!! MISMATCH: {name} has length {l}, expected {first_len}")
                mismatch = True
        
        if not mismatch:
            print("All model dataframe lengths match. Checking exact index matches...")
            # We must load the first index fully to compare
            base_idx_df = dd.read_parquet(f"{data_dir}/output_features/{model_names[0]}", filters=[("doc_id", 'in', doc_ids)])
            base_idx_df["word_id"] = base_idx_df["word_id"].astype(str)
            base_idx = base_idx_df.set_index("word_id").index.compute()
            base_set = set(base_idx)
            
            # Check for uniqueness in model index
            if len(base_idx) != len(base_set):
                 print(f"!!! DUPLICATE INDICES in {model_names[0]} !!! Length: {len(base_idx)}, Unique: {len(base_set)}")
            else:
                 print(f"Indices in {model_names[0]} are unique.")

            # Prepare checks for input_feature alignment
            print("Checking alignment between Input Features and Models...")
            input_df = dd.read_parquet(input_feature_dir, filters=[("doc_id", 'in', doc_ids)])
            input_df["word_id"] = input_df["word_id"].astype(str)
            input_idx = input_df.set_index("word_id").index.compute()
            input_set = set(input_idx)
            
            if len(input_idx) != len(input_set):
                 print(f"!!! DUPLICATE INDICES in Input Features !!! Length: {len(input_idx)}, Unique: {len(input_set)}")
            else:
                 print("Indices in Input Features are unique.")
            
            intersection = input_set.intersection(base_set)
            print(f"Input Features size: {len(input_set)}")
            print(f"Model Features size: {len(base_set)}")
            print(f"Intersection size: {len(intersection)}")
            
            if len(intersection) < len(input_set):
                 print(f"Indices in Input but not in Model: {len(input_set - base_set)}")
                 print("Sample:", list(input_set - base_set)[:5])
            if len(intersection) < len(base_set):
                 print(f"Indices in Model but not in Input: {len(base_set - input_set)}")
                 print("Sample:", list(base_set - input_set)[:5])

            # Skip checking model-to-model again as we did it in previous step and it takes time
            # We assume they match based on previous results.
            
        else:
             print("Length mismatch found. This causes NaNs during concat.")

    except Exception as e:
         print(f"Error in alignment check: {e}")

    client.close()
    print("Done.")

if __name__ == "__main__":
    check_nans_standalone()
