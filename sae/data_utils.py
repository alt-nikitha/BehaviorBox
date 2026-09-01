import dask.array as da
import dask.dataframe as dd
import gc
import glob as globmod
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
    # allow model num_words <= input num_words (output models may truncate long texts)
    for model_df in all_output_file_df:
        model_df = model_df.loc[overlapping_doc_ids]
        model_df = model_df.reindex(index=input_file_df.index)
        input_file_df = input_file_df[input_file_df["num_words"] >= model_df["num_words"]]
    return input_file_df.index.to_list()


# def load_dataframe(
#     client,
#     input_feature_dir: str,
#     output_feature_dirs: list[str] = None,
#     include_logprobs: bool = True,
# ) -> dd.DataFrame:
#     """
#     loads features from directories containing parquet files
#     into a dask dataframe that we can manipulate
#     """
#     doc_ids = _get_valid_doc_ids(input_feature_dir, output_feature_dirs)
#     print(f"reading data from {input_feature_dir}")
#     df = dd.read_parquet(input_feature_dir, filters=[("doc_id", 'in', doc_ids)])
#     df["word_id"] = df["word_id"].astype(str)
#     df = df.set_index("word_id")
#     df = client.persist(df)
#     print("Initial df shape:", df.shape[0].compute())
#     print("Initial df index sample:", df.index.compute()[:5])  # Check index format
#     if include_logprobs:
#         for dir in tqdm(output_feature_dirs):
#             model = os.path.basename(os.path.normpath(dir))
#             print(f"reading data for {model}", flush=True)
#             model_df = dd.read_parquet(dir, filters=[("doc_id", 'in', doc_ids)])
#             print("model_df shape:", model_df.shape[0].compute())
#             model_df["word_id"] = model_df["word_id"].astype(str)
#             model_df = model_df.set_index("word_id")
#             print("model_df index sample:", model_df.index.compute()[:5])
#             model_logprobs = model_df["logprobs"].values.compute()
#             print("clipping data...", flush=True)
#             min_neg_fp16 = -6.10352e-05
#             model_logprobs_clipped = np.where((model_logprobs < 0) & (model_logprobs > min_neg_fp16), min_neg_fp16, model_logprobs)
#             print("merging data...", flush=True)
#             temp_df = pd.DataFrame({f"{model}_logprobs": model_logprobs_clipped}, index=model_df.index)
#             temp_df.index = temp_df.index.astype(str)  # Ensure string type index
#             print("temp_df index sample:", temp_df.index[:5])
#             model_logprobs_df = dd.from_pandas(temp_df, npartitions=df.npartitions)
#             model_logprobs_df.index = model_logprobs_df.index.astype(str)
#             print("Pre-merge shapes:")
#             print(" - df:", df.shape[0].compute())
#             print(" - model_logprobs_df:", model_logprobs_df.shape[0].compute())
#             df = dd.merge(df, model_logprobs_df, left_index=True, right_index=True, how='inner')
#             # df = client.persist(df)
#             print("Post-merge shape:", df.shape[0].compute())
#             print("Post-merge columns:", df.columns)
#     df = df.repartition(partition_size="100MB")
#     df = client.persist(df)
#     print(df.shape[0].compute())
#     return df


def load_dataframe(
    client,
    input_feature_dir: str,
    output_feature_dirs: list[str] = None,
    include_logprobs: bool = True,
) -> dd.DataFrame:
    """
    loads features from directories containing parquet files.

    Everything is loaded into Pandas (the node has plenty of RAM),
    joined in-memory, then converted to a Dask DataFrame at the end.
    """
    doc_ids = [str(x) for x in _get_valid_doc_ids(input_feature_dir, output_feature_dirs)]

    df = None
    if input_feature_dir is not None:
        print(f"reading data from {input_feature_dir}")
        parquet_files = sorted(globmod.glob(os.path.join(input_feature_dir, "*.parquet")))
        df = pd.concat([pd.read_parquet(f, filters=[("doc_id", 'in', doc_ids)]) for f in parquet_files])
        df["word_id"] = df["word_id"].astype(str)
        df = df.set_index("word_id")
        print(f"Input features loaded: {df.shape}")

    if include_logprobs and output_feature_dirs:
        print("Loading model logprobs...", flush=True)
        all_model_dfs = []

        for dir in tqdm(output_feature_dirs):
            model = os.path.basename(os.path.normpath(dir))
            model_files = sorted(globmod.glob(os.path.join(dir, "*.parquet")))
            model_pd = pd.concat([
                pd.read_parquet(f, columns=["doc_id", "word_id", "logprobs"],
                                filters=[("doc_id", 'in', doc_ids)])
                for f in model_files
            ])[["word_id", "logprobs"]]
            model_pd["word_id"] = model_pd["word_id"].astype(str)
            model_pd = model_pd.rename(columns={"logprobs": f"{model}_logprobs"})
            model_pd = model_pd.set_index("word_id")

            min_neg_fp16 = -6.10352e-05
            col_name = f"{model}_logprobs"
            vals = model_pd[col_name].values
            vals = np.where((vals < 0) & (vals > min_neg_fp16), min_neg_fp16, vals)
            model_pd[col_name] = vals

            all_model_dfs.append(model_pd)

        print("Concatenating model logprobs...", flush=True)
        models_pd = pd.concat(all_model_dfs, axis=1, join='inner')
        print(f"Model data shape: {models_pd.shape}")

        if df is not None:
            print("Joining input features with model logprobs...", flush=True)
            df = df.join(models_pd, how='inner')
            print(f"Joined shape: {df.shape}")
        else:
            df = models_pd

    # Convert back to Dask for downstream compatibility
    print(f"Final df shape: {len(df)}")
    result = dd.from_pandas(df, npartitions=200)
    del df
    gc.collect()
    result = client.persist(result)
    return result





def preprocess_data(
    data_df: dd.DataFrame,
    output_feature_dim: int,
    input_feature_dim: int = 768,
    output_feature_weight: float = None,
    logprobs: bool = False,
    only_probs: bool = False,
    use_delta_prob: bool = False,
    use_delta_logprob: bool = False,
    normalize_per_part: bool = False,
    znorm_prob_per_sample: bool = False,
    znorm_prob_eps: float = 1e-2,
    pairwise_sign: bool = False,
    norm_stats_out: dict = None,
) -> da.Array:
    if use_delta_prob and use_delta_logprob:
        raise ValueError("use_delta_prob and use_delta_logprob are mutually exclusive")
    if pairwise_sign and znorm_prob_per_sample:
        raise ValueError(
            "pairwise_sign and znorm_prob_per_sample are mutually exclusive: the sign "
            "encoding is already scale-free, and z-norming it would destroy the +/-1 "
            "structure that makes L2 distance equal to Kendall tau"
        )
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
    
    if not logprobs and not use_delta_logprob:
        print("Converting to probabilities", flush=True)
        data = data.map_blocks(block_to_probs, input_feature_dim, dtype=np.float16)

    if use_delta_prob or use_delta_logprob:
        space = "logprobs" if use_delta_logprob else "probabilities"
        print(f"Converting {space} to consecutive deltas", flush=True)
        embed = data[:, :input_feature_dim]
        vals = data[:, input_feature_dim:]
        deltas = vals[:, :-1] - vals[:, 1:]
        data = da.concatenate([embed, deltas], axis=1)
        data = data.rechunk({0: data.chunks[0], 1: -1})

    if pairwise_sign:
        # Replace the C-checkpoint trajectory with the signs of all C*(C-1)/2 pairwise
        # differences: s_ij = sign(p_j - p_i) for i<j, in {-1, 0, +1} (0 on ties).
        #
        # For two tokens a, b this makes the dot product exactly the Kendall numerator:
        #   a . b = #concordant - #discordant = n_pairs * tau_ab
        #   ||a - b||^2 = 2 * n_pairs * (1 - tau_ab)
        # so Euclidean distance is strictly monotone in tau, and every L2-based part of
        # the pipeline (SAE reconstruction MSE, KMeans stratification) groups tokens by
        # rank agreement instead of by curve values. Unlike Pearson r on the raw curves
        # -- which saturates near 0.95 for almost any pair of 11-point trajectories --
        # tau has real dynamic range and an exact permutation null.
        #
        # Note this widens the output block from C to C*(C-1)/2 (11 -> 55), which the
        # caller must reflect in output_feature_dim. The ofw scaling downstream measures
        # the block norm empirically, so it rebalances on its own.
        print("Encoding trajectory as pairwise checkpoint signs", flush=True)
        data = data.rechunk({0: data.chunks[0], 1: -1})
        n_ckpt = data.shape[1] - input_feature_dim
        iu, ju = np.triu_indices(n_ckpt, k=1)

        def pairwise_sign_block(block, input_feature_dim, iu, ju):
            block = block.astype(np.float32)
            prob = block[:, input_feature_dim:]
            signs = np.sign(prob[:, ju] - prob[:, iu])
            return np.concatenate(
                [block[:, :input_feature_dim], signs], axis=1
            ).astype(np.float16)

        data = data.map_blocks(
            pairwise_sign_block, input_feature_dim, iu, ju,
            dtype=np.float16,
            chunks=(data.chunks[0], (input_feature_dim + len(iu),)),
        )

    if znorm_prob_per_sample:
        # Per-sample z-norm of the prob block ACROSS CHECKPOINTS: each token's prob
        # trajectory is mapped to zero mean / unit-ish std over its OWN checkpoints,
        # so the SAE clusters on trajectory *shape* (matching the analysis z_full
        # metric) instead of level/amplitude. The embedding block is left untouched.
        # Std is floored with sqrt(var + eps**2) so near-flat curves collapse toward
        # 0 rather than being amplified into noise. Block-level magnitude balance is
        # then handled by the ofw scaling below (so no output_dim_loss_weight needed).
        if normalize_per_part:
            raise ValueError(
                "znorm_prob_per_sample and normalize_per_part are mutually exclusive"
            )
        print(
            f"Per-sample z-norming prob block across checkpoints (eps={znorm_prob_eps})",
            flush=True,
        )

        def znorm_prob_per_sample_block(block, input_feature_dim, eps):
            block = block.astype(np.float32)
            prob = block[:, input_feature_dim:]
            mean = prob.mean(axis=1, keepdims=True)
            std = np.sqrt(prob.var(axis=1, keepdims=True) + eps ** 2)
            block[:, input_feature_dim:] = (prob - mean) / std
            return block.astype(np.float16)

        data = data.map_blocks(
            znorm_prob_per_sample_block, input_feature_dim, znorm_prob_eps,
            dtype=np.float16,
        )

    if normalize_per_part:
        # Per-column z-score, computed independently on the embedding block and the
        # output (prob/logprob/delta) block so each block has unit variance per dim.
        # `output_feature_weight` is ignored here — block-level magnitude balance is
        # handled by the loss-side per-dim weighting (output_dim_loss_weight).
        print("Computing per-column means/stds for z-score normalization", flush=True)
        data_f32 = data.astype(np.float32)
        col_mean = data_f32.mean(axis=0).compute()
        col_std = data_f32.std(axis=0).compute()
        eps = 1e-6
        col_std_safe = np.where(col_std < eps, 1.0, col_std).astype(np.float32)
        col_mean_f32 = col_mean.astype(np.float32)

        if norm_stats_out is not None:
            norm_stats_out["mean"] = col_mean_f32
            norm_stats_out["std"] = col_std_safe
            norm_stats_out["embedding_dim"] = input_feature_dim
            norm_stats_out["output_feature_dim"] = output_feature_dim

        def znorm_block(block, mean, std):
            block = block.astype(np.float32)
            block = (block - mean) / std
            return block.astype(np.float16)

        data = data.map_blocks(znorm_block, col_mean_f32, col_std_safe, dtype=np.float16)
        return data

    mean_total_norm = None
    mean_embedding_norm = None
    mean_prob_norm = None
    # scale each sample (row) of the data according to the output weight specified
    if not only_probs and (output_feature_weight is not None and output_feature_weight > 0 and output_feature_weight < 1):
        assert output_feature_dim == data.shape[1] - input_feature_dim, "Output feature dim must match data shape"
        print("Computing norms", flush=True)
        mean_total_norm = log_space_mean_norm(data)
        mean_embedding_norm = log_space_mean_norm(data[:, :input_feature_dim])
        mean_prob_norm = log_space_mean_norm(data[:, input_feature_dim:])
        print(mean_total_norm, mean_embedding_norm, mean_prob_norm, flush=True)
    if output_feature_weight is not None or not only_probs:
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
    # Use copyfile instead of copy2 to avoid permission issues with metadata
    shutil.copyfile(orig_memmap_filepath, temp_filepath)


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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def clean_string(s):
        s = s.replace("Ġ", " ")
        s = s.replace("Ċ", "\t")
        return s

    file_df = pd.read_csv(f"{input_feature_dir}/file_to_doc.csv")
    file_df.drop(columns=["num_words"], inplace=True)
    file_df["doc_id"] = file_df["doc_id"].astype(str)
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
    # Group doc_ids by file upfront to avoid O(n) scan per file
    file_to_docs = file_df.groupby("file")["doc_id"].apply(list).to_dict()

    def _process_file(file, docs):
        filepath = os.path.join(input_feature_dir, file)
        docs_df = pd.read_parquet(filepath, columns=["doc_id", "word"], filters=[("doc_id", 'in', docs)])
        results = {}
        for doc_id, group in docs_df.groupby("doc_id"):
            doc_id = str(doc_id)
            if doc_id not in doc_pos:
                continue
            doc_words = group["word"].tolist()
            for pos in doc_pos[doc_id]:
                word_id = f"{doc_id}_{pos}"
                start_id = max(pos - N, 0)
                end_id = min(pos + N + 1, len(doc_words))
                before = clean_string("".join(doc_words[start_id:pos]))
                word = clean_string(doc_words[pos])
                after = clean_string("".join(doc_words[pos+1:end_id]))
                results[word_id] = {
                    "before": before,
                    "word": word,
                    "after": after,
                }
        return results

    words_in_context = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_process_file, file, docs): file
            for file, docs in file_to_docs.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            words_in_context.update(future.result())
    return words_in_context
