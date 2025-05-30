import click
import dask.dataframe as dd
import json
import os
import pickle

from dask import config as dask_cfg
from dask.distributed import Client

from data_utils import (
    preprocess_data,
    dask_to_cached_memmap,
    load_dataframe,
    get_model_logprobs
)

def cache_data_per_dir(
    client: Client,
    cache_dir: str,
    data_dir: str,
    model_names: list[str] = None,
    output_feature_weight: float = None,
) -> str:
    model_string = "_".join(model_names)
    cache_dir = os.path.join(cache_dir, f"{model_string}/{os.path.basename(data_dir)}")
    cache_data_dir = os.path.join(cache_dir, f"ofw={output_feature_weight}")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)
    
    dataframes = []
    input_feature_dir = f"{data_dir}/input_features"
    if model_names is not None:
        output_feature_dirs = [f"{data_dir}/output_features/{model_name}" for model_name in model_names]
    else:
        output_feature_dirs = []
    dataframes.append(load_dataframe(
        client,
        input_feature_dir,
        output_feature_dirs,
    ))

    dataframe = dd.concat(dataframes)
    word_ids = dataframe.index.values.compute()
    with open(os.path.join(cache_dir, "word_ids.pkl"), "wb") as f:
        pickle.dump(word_ids, f)
    print(f"Word IDs cached to {os.path.join(cache_dir, 'word_ids.pkl')}")
    if "domain" in dataframe.columns:
        domains = dataframe["domain"].values.compute()
        with open(os.path.join(cache_dir, "domains.pkl"), "wb") as f:
            pickle.dump(domains, f)
        print(f"Domain data cached to {os.path.join(cache_dir, 'domains.pkl')}")
    models_to_cache = []
    cache_subdir_names = [os.path.basename(dir) for dir in os.listdir(cache_dir)]
    for model in model_names:
        if model not in cache_subdir_names:
            models_to_cache.append(model)
    logprobs, zscores = get_model_logprobs(models_to_cache, dataframe)
    for model in models_to_cache:
        model_data_dir = os.path.join(cache_dir, model)
        if not os.path.exists(model_data_dir):
            os.makedirs(model_data_dir)
            model_logprobs = logprobs[model]
            model_zscores = zscores[model]
            with open(os.path.join(model_data_dir, "logprobs.pkl"), "wb") as f:
                pickle.dump(model_logprobs, f)
            print(f"Logprobs cached to {os.path.join(model_data_dir, 'logprobs.pkl')}")
            with open(os.path.join(model_data_dir, "zscores.pkl"), "wb") as f:
                pickle.dump(model_zscores, f)
            print(f"Z-scores cached to {os.path.join(model_data_dir, 'zscores.pkl')}")
    
    print("preprocessing data...", flush=True)
    data_array = preprocess_data(
        data_df=dataframe,
        output_feature_dim=len(model_names),
        output_feature_weight=output_feature_weight,
    )
    
    print("caching data...", flush=True)
    _, cache_filename, _ = dask_to_cached_memmap(
        data_array, cache_dir=cache_data_dir, cache_key="preprocessed_data", temp_dir=None)
    print(f"Preprocessed data cached to {cache_filename}", flush=True)
    
    cached_data_info = {
        "shape": data_array.shape,
        "dtype": "np16",
        "filename": cache_filename,
        "data_dir": data_dir,
        "model_names": model_names,
    }
    print(cached_data_info, flush=True)
    with open(f"{cache_data_dir}/cached_data_info.json", "w") as f:
        json.dump(cached_data_info, f, indent=4)
    print(f"Saved cached data info to {cache_data_dir}/cached_data_info.json", flush=True)
    
    client.cancel(data_array)
    return cache_dir


@click.command()
@click.option(
    "--cache_dir",
    help="Cache directory to save data to",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--data_dirs",
    help="Directories containing input and output features",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
)
@click.option(
    "--model_names",
    help="Names of language models to evaluate, should match output feature directory names",
    type=str,
    multiple=True,
)
@click.option(
    "--output_feature_weight",
    help="Relative weight of model logprobs",
    default=None,
)
@click.option(
    "--spill_dir",
    help="Directory to save spilled data",
    type=click.Path(exists=True),
)
@click.option(
    "--workers",
    type=int,
    default=16,
)
def main(
    cache_dir: str,
    data_dirs: list[str],
    model_names: list[str],
    spill_dir: str,
    output_feature_weight = None,
    workers: int = 16,
):
    if output_feature_weight == "None":
        output_feature_weight = None
    else:
        output_feature_weight = float(output_feature_weight)

    print(output_feature_weight)

    dask_cfg.set({'distributed.scheduler.worker-ttl': None})
    client = Client(
        n_workers=workers, memory_limit='20GB', processes=True, timeout='30s', local_directory=spill_dir
    )
    print(client)

    for data_dir in data_dirs:
        _ = cache_data_per_dir(
            client,
            cache_dir,
            data_dir,
            model_names,
            output_feature_weight,
        )
    client.close()
    
if __name__ == "__main__":
    main()
