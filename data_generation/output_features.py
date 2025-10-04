"""
Script to get output features (logprobs under evaluated LM) per word
Generates the following:
    1. A csv file that maps documents to the batch file containing 
        their features and number of words per doc
    2. (Per batch) a parquet file containing the logprob of each word
"""
import click
import gc
import logging
import math
import numpy as np
import os
import pandas as pd
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.text_samples import filter_and_sort_data, get_batch_data
from utils.output_features import get_output_overlapping_strings, get_model_logprobs

logging_dir = "../logs/output_features"
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
timestr = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    filename=f"{logging_dir}/{timestr}.log",
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
# Only log errors from openai and httpx
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def process_jsonl_file(
    filepath: str,
    is_sorted: bool,
    model_addr: str,
    model_id: str,
    model_name: str,
    output_dir: str,
    start_batch: int = 0,
    end_batch: int = None,
    batch_size: int = 100,
    max_length: int = 1800,  # was 4000, reduced for Pythia-160m context window (2048)
    stride: int = 900,       # was 2000, reduced proportionally
    async_limiter: int = 100,
) -> torch.Tensor:
    text, sample_ids, domains = filter_and_sort_data(filepath, is_sorted)
    
    logger.info(f"processing {len(text)} samples")
    num_batches = math.ceil(len(text) / batch_size)
    if end_batch is not None:
        num_batches = end_batch

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
    
    for batch_idx in tqdm(range(start_batch, num_batches)):
        batch_text, batch_sample_ids, _ = get_batch_data(
            text, sample_ids, domains, batch_size, batch_idx
        )
        # first check that we haven't already processed these documents
        # we do this by checking that the last batch_sample_id is not in the batch_to_doc_file
        if continuing_processing:
            if str(batch_sample_ids[-1]) in batch_to_doc_df["doc_id"].astype(str).values:
                logger.info(f"samples up to {((batch_idx+1)*batch_size)-1} already processed")
                continue
        try:
            print("getting overlapping strings\n")
            # this may need to be changed if we switch out the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            embedding_model_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
            overlapping_strings, window_start_idx, window_to_sample_mapping, window_word_ids = get_output_overlapping_strings(
                text=batch_text,
                sample_ids=batch_sample_ids,
                tokenizer=tokenizer,
                embedding_model_tokenizer=embedding_model_tokenizer,
                max_length=max_length,
                stride=stride,
            )
            print("getting model logprobs\n")
            output_features = get_model_logprobs(
                model_addr=model_addr,
                model_id=model_id,
                input_text=overlapping_strings,
                sample_ids=batch_sample_ids,
                window_word_ids=window_word_ids,
                window_start_idx=window_start_idx,
                window_to_sample_mapping=window_to_sample_mapping,
                async_limiter=async_limiter,
            )
            logprobs = list(output_features.values())
            num_words = [len(sample_logprobs) for sample_logprobs in logprobs]
            doc_ids = list(output_features.keys())
            output_file = f"{model_name}-{batch_idx*batch_size}_{((batch_idx+1)*batch_size)-1}.parquet"
            # update the batch file to doc ID csv
            batch_to_doc_df = pd.DataFrame(
                {
                    "file": [output_file] * len(output_features),
                    "doc_id": doc_ids,
                    "num_words": num_words,
                }
            )
            if not os.path.exists(batch_to_doc_file):
                batch_to_doc_df.to_csv(batch_to_doc_file, index=False)
            else:
                batch_to_doc_df.to_csv(batch_to_doc_file, mode="a", header=False, index=False)
            
            output_filepath = os.path.join(output_dir, output_file)
            flattened_logprobs = []
            flattened_doc_ids = []
            word_id = []
            for i, sample in enumerate(logprobs):
                flattened_logprobs += [logprob for logprob in sample]
                flattened_doc_ids += [doc_ids[i]] * len(sample)
                for word_idx in range(len(sample)):
                    word_id.append(f"{doc_ids[i]}_{word_idx}")
            logprobs_df = pd.DataFrame({
                "word_id": word_id,
                "doc_id": flattened_doc_ids,
                "logprobs": flattened_logprobs,
            })
            logger.info(f"saving output features for batch {batch_idx}")
            logprobs_df.to_parquet(output_filepath, engine="pyarrow")
            del output_features
            gc.collect()
            torch.cuda.empty_cache()
        except AssertionError as msg:
            logger.info(f"Unable to process samples from batch {batch_idx}: {batch_idx*batch_size} to {((batch_idx+1)*batch_size)-1}")
            logger.info(msg)


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
    help="Directory to save logprob tensor(s)",
    type=click.Path(),
    required=True,
)
@click.option(
    "--model_addr",
    help="Address of hosted model",
    type=str,
    required=True,
)
@click.option(
    "--model_id",
    help="Huggingface ID of hosted model",
    type=str,
    required=True,
)
@click.option(
    "--model_name",
    help="Name of model",
    type=str,
    required=True,
)
@click.option(
    "--start_batch",
    help="Batch index to start at (inclusive)",
    type=int,
    default=0,
)
@click.option(
    "--end_batch",
    help="Batch index to end at (exclusive)",
    type=int,
    default=None,
)
@click.option(
    "--batch_size",
    help="Size of batch to process",
    type=int,
    default=100,
)
@click.option(
    "--async_limiter",
    help="Number of samples that can be processed asynchronously",
    type=int,
    default=100,
)
def main(
    data: str,
    model_addr: str,
    model_id: str,
    model_name: str,
    output_dir: str,
    is_sorted: bool = False,
    start_batch: int = 0,
    end_batch: int = None,
    batch_size: int = 100,
    async_limiter: int = 100,
):
    logger.info(f"collecting logprobs from {model_name}")
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
            model_addr=model_addr,
            model_id=model_id,
            model_name=model_name,
            output_dir=output_dir,
            start_batch=start_batch,
            end_batch=end_batch,
            batch_size=batch_size,
            async_limiter=async_limiter,
        )


if __name__ == "__main__":
    main()
