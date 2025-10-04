import dask.array as da
import dask.dataframe as dd
import numpy as np
import os
import pandas as pd
import shutil

from tqdm import tqdm


def get_file_list(
    directory: str,
) -> tuple[str]:
    print(directory)
    files = (os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f)))
    return files


def _get_valid_doc_ids(
    input_feature_dir: str,
    output_feature_dirs: list[str],
) -> list[str]:
    input_file_df = pd.read_csv(f"{input_feature_dir}/file_to_doc.csv")
    input_file_df.drop_duplicates(subset="doc_id", keep="last", inplace=True)
    input_file_df.set_index("doc_id", inplace=True)
    all_doc_ids = [set(input_file_df.index.tolist())]
    all_output_file_df = []
    for dir in output_feature_dirs:
        model_file_df = pd.read_csv(f"{dir}/file_to_doc.csv")
        model_file_df.drop_duplicates(subset="doc_id", keep="last", inplace=True)
        model_file_df.set_index("doc_id", inplace=True)
        all_doc_ids.append(set(model_file_df.index.tolist()))
        all_output_file_df.append(model_file_df)
    # get intersection of doc_ids across all features
    overlapping_doc_ids = list(set.intersection(*all_doc_ids))
    input_file_df = input_file_df.loc[overlapping_doc_ids]
    # filter the dataframes to only include these features
    # reset index so that the order matches
    # filter for rows where the number of words match
    for model_df in all_output_file_df:
        model_df = model_df.loc[overlapping_doc_ids]
        model_df = model_df.reindex(index=input_file_df.index)
        input_file_df = input_file_df[input_file_df["num_words"] == model_df["num_words"]]
    return input_file_df.index.to_list()


def load_dataframe(
    client,
    input_feature_dir: str,
    output_feature_dirs: list[str] = None,
    include_logprobs: bool = True,
) -> dd.DataFrame:
    """
    loads features from directories containing parquet files
    into a dask dataframe that we can manipulate
    """
    doc_ids = _get_valid_doc_ids(input_feature_dir, output_feature_dirs)
    print(f"reading data from {input_feature_dir}")
    df = dd.read_parquet(input_feature_dir, filters=[("doc_id", 'in', doc_ids)])
    df["word_id"] = df["word_id"].astype(str)
    df = df.set_index("word_id")
    df = client.persist(df)
    print("Initial df shape:", df.shape[0].compute())
    print("Initial df index sample:", df.index.compute()[:5])  # Check index format
    if include_logprobs:
        for dir in tqdm(output_feature_dirs):
            model = os.path.basename(os.path.normpath(dir))
            print(f"reading data for {model}", flush=True)
            model_df = dd.read_parquet(dir, filters=[("doc_id", 'in', doc_ids)])
            print("model_df shape:", model_df.shape[0].compute())
            model_df["word_id"] = model_df["word_id"].astype(str)
            model_df = model_df.set_index("word_id", inplace=True)
            print("model_df index sample:", model_df.index.compute()[:5])
            model_logprobs = model_df["logprobs"].values.compute()
            print("clipping data...", flush=True)
            min_neg_fp16 = -6.10352e-05
            model_logprobs_clipped = np.where((model_logprobs < 0) & (model_logprobs > min_neg_fp16), min_neg_fp16, model_logprobs)
            print("merging data...", flush=True)
            temp_df = pd.DataFrame({f"{model}_logprobs": model_logprobs_clipped}, index=model_df.index)
            temp_df.index = temp_df.index.astype(str)  # Ensure string type index
            print("temp_df index sample:", temp_df.index[:5])
            model_logprobs_df = dd.from_pandas(temp_df, npartitions=df.npartitions)
            model_logprobs_df.index = model_logprobs_df.index.astype(str)
            print("Pre-merge shapes:")
            print(" - df:", df.shape[0].compute())
            print(" - model_logprobs_df:", model_logprobs_df.shape[0].compute())
            df = dd.merge(df, model_logprobs_df, left_index=True, right_index=True, how='inner')
            df = client.persist(df)
            print("Post-merge shape:", df.shape[0].compute())
            print("Post-merge columns:", df.columns)
    df = df.repartition(partition_size="100MB")
    df = client.persist(df)
    print(df.shape[0].compute())
    return df


def preprocess_data(
    data_df: dd.DataFrame,
    output_feature_dim: int,
    input_feature_dim: int = 768,
    output_feature_weight: float = None,
    logprobs: bool = False,
) -> da.Array:
    def block_to_probs(block, input_feature_dim):
        block[:, input_feature_dim:] = np.exp(block[:, input_feature_dim:])
        return block
    
    def log_space_mean_norm(data):
        def log_norm_block(block):
            """Calculate norms in log space for a single block"""
            block = block.astype('float64')
            log_norms = np.zeros(block.shape[0])
            for i in range(block.shape[0]):
                row = block[i]
                # Handle zero rows
                if np.all(row == 0):
                    log_norms[i] = -np.inf  # log(0)
                    continue
                # Find absolute values
                abs_vals = np.abs(row)
                # Find max for scaling
                max_val = np.max(abs_vals)
                # Scale values (avoid overflow when squaring)
                scaled = abs_vals / max_val
                # Calculate log of squared norm: log(sum(x²)) = log(max²) + log(sum((x/max)²))
                log_sum_squared = np.log(np.sum(scaled**2))
                log_norm_squared = log_sum_squared + 2 * np.log(max_val)
                # Convert to log(norm): log(sqrt(x)) = 0.5 * log(x)
                log_norms[i] = 0.5 * log_norm_squared
            return log_norms
        
        # Apply to blocks
        log_norms = data.map_blocks(log_norm_block, drop_axis=1)
        # Convert back from log space for final mean
        norms = da.exp(log_norms)
        # Compute mean
        return norms.mean().compute()

    def scale_block(
            block,
            input_feature_dim,
            output_feature_weight,
            mean_total_norm,
            mean_embedding_norm,
            mean_prob_norm
        ):
        if output_feature_weight < 1 and output_feature_weight > 0:
            embedding_multiplier = (1-output_feature_weight) * mean_total_norm / mean_embedding_norm
            prob_multiplier = output_feature_weight * mean_total_norm / mean_prob_norm
        elif output_feature_weight == 1:
            embedding_multiplier = 0
            prob_multiplier = 1
        elif output_feature_weight == 0:
            embedding_multiplier = 1
            prob_multiplier = 0
        block[:, :input_feature_dim] = block[:, :input_feature_dim] * embedding_multiplier
        block[:, input_feature_dim:] = block[:, input_feature_dim:] * prob_multiplier
        return block
    
    if "domain" in data_df.columns:
        data_features = data_df.drop(columns=["doc_id", "domain", "word"])
    else:
        data_features = data_df.drop(columns=["doc_id", "word"])
    
    print("Converting to dask array", flush=True)
    data = data_features.to_dask_array(lengths=True)
    
    if not logprobs:
        print("Converting to probabilities", flush=True)
        data = data.map_blocks(block_to_probs, input_feature_dim, dtype=np.float16)
    
    mean_total_norm = None
    mean_embedding_norm = None
    mean_prob_norm = None
    # scale each sample (row) of the data according to the output weight specified
    if output_feature_weight is not None and output_feature_weight > 0 and output_feature_weight < 1:
        assert output_feature_dim == data.shape[1] - input_feature_dim, "Output feature dim must match data shape"
        print("Computing norms", flush=True)
        mean_total_norm = log_space_mean_norm(data)
        mean_embedding_norm = log_space_mean_norm(data[:, :input_feature_dim])
        mean_prob_norm = log_space_mean_norm(data[:, input_feature_dim:])
        print(mean_total_norm, mean_embedding_norm, mean_prob_norm, flush=True)
    if output_feature_weight is not None:
        print("Scaling features", flush=True)
        data = data.map_blocks(
            scale_block,
            input_feature_dim,
            output_feature_weight,
            mean_total_norm,
            mean_embedding_norm,
            mean_prob_norm,
            dtype=np.float16
        )
    return data


def dask_to_cached_memmap(
        dask_array: da.Array, cache_dir: str, cache_key: str, temp_dir: str = None, logger = None,
    ) -> tuple[np.memmap, str]:
    """
    Convert a Dask array to a memory-mapped NumPy array with caching.

    Parameters:
        dask_array (da.Array): The Dask array to process.
        cache_dir (str): The directory to store the cache file. If None, only save to a temporary file.
        cache_key (str): A unique identifier for the cached file.
        temp_dir (str): The directory to store the temporary file.

    Returns:
        tuple[np.memmap, str, str]: The memory-mapped array, the cache file path, and temporary file path.
    """
    # Create a temporary file
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = os.path.join(cache_dir, f"{cache_key}.dat")
    else:
        cache_filename = os.path.join(temp_dir, f"{cache_key}.dat")
    
    try:
        # Create a memmap array
        shape = dask_array.shape
        dtype = np.float16
        memmap_array = np.memmap(cache_filename, dtype=dtype, mode='w+', shape=shape)

        # Calculate the slices and store each chunk
        for i, chunk in enumerate(dask_array.to_delayed().flatten()):
            # Compute the chunk
            chunk_result = chunk.compute()

            # Determine the slices for this chunk
            slices = []
            for dim, block_size in zip(dask_array.chunks, np.unravel_index(i, [len(c) for c in dask_array.chunks])):
                start = sum(dim[:block_size])
                stop = start + dim[block_size]
                slices.append(slice(start, stop))
        
            # Convert list of slices to a tuple and write to memmap
            memmap_array[tuple(slices)] = chunk_result

        if cache_dir is not None:
            print(f"Created cache file: {cache_filename}")
            if logger is not None:
                logger.info(f"Created cache file: {cache_filename}")
            # Copy it to a temp file for local access
            if temp_dir is not None:
                tempfile_path = os.path.join(temp_dir, f"{cache_key}.dat")
                copy_temp_memmap(cache_filename, tempfile_path)
                print(f"Created temporary file: {tempfile_path}")
                memmap_array = np.memmap(tempfile_path, dtype=dtype, mode='r', shape=shape)
            else:
                tempfile_path = cache_filename
        else:
            print(f"Created temporary file: {cache_filename}")
            tempfile_path = cache_filename
        return memmap_array, cache_filename, tempfile_path
    except Exception as e:
        # If an exception occurs, make sure to delete the file
        os.unlink(cache_filename)
        raise e


def copy_temp_memmap(orig_memmap_filepath, temp_filepath) -> str:
    # Create a temporary file
    shutil.copy2(orig_memmap_filepath, temp_filepath)


def cleanup_temp_memmap(memmap_array, temp_filename):
    del memmap_array  # Close the memmap
    os.unlink(temp_filename)  # Delete the temporary file


def get_model_logprobs(
    model_names: list[str],
    data: dd.DataFrame,
) -> tuple[dict, dict]:
    """
    Function that takes as model names and the raw dataframe 
    and returns dicts of logprobs and z-scores for each model
    """
    # we z-score the logprobs to assist with comparison across models
    model_logprobs = {}
    model_zscores = {}
    for model in model_names:
        model_logprobs[model] = data[f"{model}_logprobs"].values.compute()
        model_zscores[model] = \
            ((data[f"{model}_logprobs"] - data[f"{model}_logprobs"].mean()) /  data[f"{model}_logprobs"].std()).values.compute()
    return model_logprobs, model_zscores


def get_words_in_context(
    input_feature_dir: str,
    word_ids: list[str],
    N: int = 10,
) -> dict[str, dict]:
    def clean_string(s):
        s = s.replace("Ġ", " ")
        s = s.replace("Ċ", "\t")
        return s
    
    file_df = pd.read_csv(f"{input_feature_dir}/file_to_doc.csv")
    file_df.drop(columns=["num_words"], inplace=True)
    doc_pos = {}
    for id in word_ids:
        doc_id = "_".join(id.split("_")[:-1])
        word_pos = int(id.split("_")[-1])
        if doc_id in doc_pos:
            doc_pos[doc_id].append(word_pos)
        else:
            doc_pos[doc_id] = [word_pos]
    doc_ids = list(doc_pos.keys())
    file_df = file_df[file_df["doc_id"].isin(doc_ids)]
    words_in_context = {}
    for file in tqdm(file_df["file"].to_list()):
        docs = file_df[file_df["file"] == file]["doc_id"].to_list()
        file = os.path.join(input_feature_dir, file)
        docs_df = pd.read_parquet(file, columns=["doc_id", "word"], filters=[("doc_id", 'in', docs)])
        for doc in tqdm(docs):
            doc_df = docs_df[docs_df["doc_id"] == doc]
            doc_words = doc_df["word"].tolist()
            for pos in doc_pos[doc]:
                word_id = f"{doc}_{pos}"
                start_id = max(pos - N, 0)
                end_id = min(pos + N + 1, len(doc_words))
                before = clean_string("".join(doc_words[start_id:pos]))
                word = clean_string(doc_words[pos])
                after = clean_string("".join(doc_words[pos+1:end_id]))
                words_in_context[word_id] = {
                    "before": before,
                    "word": word,
                    "after": after,
                }
    return words_in_context
