"""
Script to get input features (Longformer embeddings) per word
Generates the following:
    1. A csv file that maps documents to the batch file containing 
        their features and number of words per doc
    2. (Per batch) a parquet file containing Longformer embeddings per word
"""
import click
import gc
import logging
import math
import os
import pandas as pd
import time
import torch

from transformers import LongformerModel, AutoTokenizer
from tqdm import tqdm

from utils.text_samples import filter_and_sort_data, get_batch_data
from utils.input_features import get_longformer_word_features


logging_dir = "../logs/input_features"
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
timestr = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    filename=f"{logging_dir}/{timestr}.log",
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Using GPU")
    else:
        device = "cpu"
        logger.info("No GPU available")
    return device

def get_model_and_tokenizer(device: str):
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", torch_dtype=torch.float16)
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096", torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    return tokenizer, model

def process_jsonl_file(
    filepath: str,
    is_sorted: bool,
    output_dir: str,
    start_batch: int = 0,
    batch_size: int = 10,
) -> torch.Tensor:
    text, sample_ids, domains = filter_and_sort_data(filepath, is_sorted)
    
    logger.info(f"processing {len(text)} samples")
    num_batches = math.ceil(len(text) / batch_size)
    device = get_device()
    tokenizer, model = get_model_and_tokenizer(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"processing data from {filepath} with batch size {batch_size}")
    logger.info(f"saving to {output_dir}")
    batch_to_doc_file = f"{output_dir}/file_to_doc.csv"
    logger.info(f"recording location of docs in {batch_to_doc_file}")

    # check if we're continuing processing the data
    continuing_processing = False
    if os.path.exists(batch_to_doc_file):
        batch_to_doc_df = pd.read_csv(batch_to_doc_file)
        continuing_processing = True

    for batch_idx in range(start_batch, num_batches):
        batch_text, batch_sample_ids, batch_domains, = get_batch_data(
            text, sample_ids, domains, batch_size, batch_idx)
        sample_domains = {sample_id: domain for sample_id, domain in zip(batch_sample_ids, batch_domains)}
        # first check that we haven't already processed these documents
        # we do this by checking that the last batch_sample_id is not in the batch_to_doc_file
        if continuing_processing:
            if str(batch_sample_ids[-1]) in batch_to_doc_df["doc_id"].astype(str).values:
                logger.info(f"samples up to {((batch_idx+1)*batch_size)-1} already processed")
                continue
        logger.debug(f"longest sample length in batch: {len(batch_text[-1])}")
        try:
            input_features, words, num_words = get_longformer_word_features(
                device,
                tokenizer,
                model,
                input_text=batch_text,
                sample_ids=batch_sample_ids,
                use_sliding_window=True,
            )
            output_file = f"longformer-{batch_idx*batch_size}_{((batch_idx+1)*batch_size)-1}.parquet"
            # update the batch file to doc ID csv
            batch_to_doc_df = pd.DataFrame(
                {
                    "file": [output_file] * len(input_features),
                    "doc_id": input_features.keys(),
                    "num_words": num_words,
                }
            )
            if not os.path.exists(batch_to_doc_file):
                batch_to_doc_df.to_csv(batch_to_doc_file, index=False)
            else:
                batch_to_doc_df.to_csv(batch_to_doc_file, mode="a", header=False, index=False)
            # save words and Longformer features
            output_filepath = os.path.join(output_dir, output_file)
            logger.info(f"saving input features for batch {batch_idx}: {batch_idx*batch_size} to {((batch_idx+1)*batch_size)-1}")
            doc_dfs = []
            for i, doc_id in enumerate(input_features.keys()):
                word_id = []
                doc_words = words[i]
                doc_embeddings = input_features[doc_id]
                word_dim_embeddings = {}
                for dim in range(768):
                    word_dim_embeddings[f"embedding_{dim}"] = []
                for word_idx in range(num_words[i]):
                    word_id.append(f"{doc_id}_{word_idx}")
                    word_embedding = doc_embeddings[word_idx]
                    for dim in range(len(word_embedding)):
                        word_dim_embeddings[f"embedding_{dim}"].append(word_embedding[dim])
                embedding_df = pd.DataFrame(word_dim_embeddings)
                df = pd.DataFrame(
                    {
                        "word_id": word_id,     # unique ID for the word (doc_pos)
                        "doc_id": doc_id,
                        "domain": [sample_domains[doc_id]] * len(doc_words),
                        "word": doc_words,
                    }
                )
                df = pd.concat([df, embedding_df], axis=1)
                doc_dfs.append(df)
            features_df = pd.concat(doc_dfs, axis=0)
            features_df.to_parquet(output_filepath, engine="pyarrow")
            del input_features
            del features_df
            gc.collect()
            torch.cuda.empty_cache()
        except AssertionError:
            logger.info(f"Unable to process samples from batch {batch_idx}: {batch_idx*batch_size} to {((batch_idx+1)*batch_size)-1}")
    logger.info(f"Job complete, processed {num_batches} batches")


@click.command()
@click.option(
    "--data",
    help="Either a jsonl file or a directory of jsonl files containing text data",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--is_sorted",
    help="Whether to keep current sorting of data",
    type=bool,
    default=False,
)
@click.option(
    "--output_dir",
    help="Directory to save feature tensor(s)",
    type=click.Path(),
    required=True,
)
@click.option(
    "--start_batch",
    help="Batch index to start at",
    type=int,
    default=0,
)
@click.option(
    "--batch_size",
    help="Size of batch to process",
    type=int,
    default=10,
)
def main(
    data: str,
    output_dir: str,
    is_sorted: bool = False, 
    start_batch: int = 0,
    batch_size: int = 10,
):
    files = []
    if data.endswith("jsonl"):
        files.append(data)
    else:
        assert os.path.isdir(data), "data must be a jsonl or directory"
        for file in os.listdir(data):
            if file.endswith("jsonl"):
                files.append(os.path.join(data, file))
        files = sorted(files)
    for file in tqdm(files):
        process_jsonl_file(
            filepath=file,
            is_sorted=is_sorted,
            output_dir=output_dir,
            start_batch=start_batch,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    main()
