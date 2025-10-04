import gc
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

def filter_and_sort_data(
    filepath: str,
    is_sorted: bool,
) -> tuple[list[str], list[str]]:
    """
    Args:
        filepath: path to jsonl file
        is_sorted: whether the data is sorted by length

    Returns:
        text: list of texts
        sample_ids: list of sample IDs
    """
    def remove_empty_samples(text, sample_ids, domains):
        non_empty_text = []
        non_empty_sample_ids = []
        non_empty_domains = []
        text_len = []
        for i, t in enumerate(text):
            if len(t) > 0:
                non_empty_text.append(t)
                non_empty_sample_ids.append(sample_ids[i])
                non_empty_domains.append(domains[i])
                text_len.append(len(t))
        return non_empty_text, non_empty_sample_ids, non_empty_domains, text_len

    def sort_by_length(text, sample_ids, domains, text_len):
        text_len_indices = np.argsort(text_len, kind="stable").tolist()
        sorted_text = [text[i] for i in text_len_indices]
        sorted_sample_ids = [sample_ids[i] for i in text_len_indices]
        sorted_domains = [domains[i] for i in text_len_indices]
        return sorted_text, sorted_sample_ids, sorted_domains
    
    raw_data = pl.read_ndjson(filepath)
    raw_data = raw_data.with_columns(pl.col("id").cast(pl.String))
    # make sure to dedeuplicate the data
    raw_data_deduped = raw_data.unique(keep="first")
    del raw_data
    print(raw_data_deduped.head())
    text = raw_data_deduped["text"].to_list()
    sample_ids = raw_data_deduped["id"].to_list()
    if "domain" in raw_data_deduped.columns:
        domains = raw_data_deduped["domain"].to_list()
    else:
        domains = ["0"] * len(text)
    del raw_data_deduped
    text, sample_ids, domains, text_len = remove_empty_samples(text, sample_ids, domains)
    if not is_sorted:
        text, sample_ids, domains = sort_by_length(text, sample_ids, domains, text_len)
    else:
        logger.info("data is already sorted, skip sort by length")
    del text_len
    gc.collect()
    assert len(text) == len(sample_ids)
    return text, sample_ids, domains


def get_batch_data(text, sample_ids, domains, batch_size, batch_idx):
    num_samples = len(text)
    start_idx = batch_idx * batch_size
    batch_text = text[start_idx:min(start_idx + batch_size, num_samples)]
    batch_sample_ids = sample_ids[start_idx:min(start_idx + batch_size, num_samples)]
    batch_domains = domains[start_idx:min(start_idx + batch_size, num_samples)]
    return (batch_text, batch_sample_ids, batch_domains)