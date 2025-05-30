import asyncio
import logging
import nest_asyncio
import numpy as np
import openai
import sys
import torch

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

LLAMA_MODEL_IDS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
]
OLMO_MODEL_IDS = [
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMo-2-1124-7B-SFT",
    "allenai/OLMo-2-1124-7B-DPO",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-1124-13B-SFT",
    "allenai/OLMo-2-1124-13B-DPO",
]
OLMO_WHITESPACE_MAPPING = {
    "Ċ": "\n",
    "ċ": "\t",
    "Ġ": " "
}

def get_output_overlapping_strings(
    text: list[str],
    sample_ids: list[str],
    tokenizer,
    embedding_model_tokenizer,
    max_length: int = 4000,
    stride: int = 2000,
) -> tuple[list[str], list[int], list[int], list[list[int]]]:
    """
    Args:
        text: list of raw text
        sample_ids: list of sample IDs
        tokenizer: tokenizer of model being evaluated
        max_length: max length of string
            (Llama2 and Olmo2 context window is 4096, but to be safe we set it lower)
        stride: stride of window (defaults to 2000)

    Returns:
        overlapping_strings: overlapping strings to calculate logprobs of (length = num_windows >= num_samples)
        window_start_idx: list of token index of last completed word in a window (length = num_windows >= num_samples)
        window_to_sample_mapping: updated mapping between window and sample (length = num_windows >= num_samples)
        window_word_ids: list of word IDs per window (length = num_windows >= num_samples)
    """
    encodings = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    overflow_to_sample_mapping = encodings["overflow_to_sample_mapping"]
    num_samples = overflow_to_sample_mapping[-1].item() + 1
    assert num_samples == len(text)
    overlapping_ids = []
    window_start_idx = []
    window_to_sample_mapping = []
    window_word_ids = []
    for i in range(num_samples):
        end_index = max_length
        sample_windows = (overflow_to_sample_mapping == i).nonzero().flatten()
        sample_input_ids = torch.index_select(input_ids, 0, sample_windows).flatten()
        num_windows = sample_windows.shape[0]
        sample_text = text[i]
        sample_id = sample_ids[i]
        sample_word_indices = get_word_indices(
            text=sample_text,
            sample_id=sample_id,
            input_model_tokenizer=embedding_model_tokenizer,
            output_model_tokenizer=tokenizer,
        )
        if sample_word_indices is None:
            continue
        # create overlapping strings if a sample is split between multiple windows
        # make sure not to split within a word
        if num_windows > 1:
            # sample_word_indices do not include pad tokens
            # we exclude these from the input_ids as well
            sample_input_ids = sample_input_ids[:len(sample_word_indices)]
            # first check that splitting on end_index will not split a word
            # otherwise, adjust until it no longer does
            while sample_word_indices[end_index-1] == sample_word_indices[end_index]:
                end_index -= 1
            # first window starts at index 0
            # since there is nothing overlapping to throw out
            overlapping_ids.append(sample_input_ids[:end_index])
            window_start_idx.append(0)
            window_to_sample_mapping.append(i)
            window_word_ids.append(sample_word_indices[:end_index])
            # also need to check that the first split we make will not split a word
            first_split_index = stride
            # slicing at the beginning is inclusive (hence +1)
            while sample_word_indices[first_split_index] == sample_word_indices[first_split_index+1]:
                first_split_index -= 1
            # shorten the remaining input and word IDs
            # so that we can start at index 0 again
            sample_input_ids = sample_input_ids[first_split_index:]
            sample_word_indices = sample_word_indices[first_split_index:]
            # adjust previous end index accordingly
            prev_end_index = end_index - first_split_index
            while True:
                if len(sample_input_ids) == 0:
                    break
                end_index = prev_end_index + stride
                # if the length of the remaining sequence is shorter than or equal to
                # the max window length, append as is
                if end_index >= len(sample_input_ids):
                    overlapping_ids.append(sample_input_ids)
                    window_start_idx.append(prev_end_index)
                    window_to_sample_mapping.append(i)
                    window_word_ids.append(sample_word_indices)
                    break
                # adjust the end_index before appending a new window
                while sample_word_indices[end_index-1] == sample_word_indices[end_index]:
                    end_index -= 1
                overlapping_ids.append(sample_input_ids[:end_index])
                window_start_idx.append(prev_end_index)
                window_to_sample_mapping.append(i)
                window_word_ids.append(sample_word_indices[:end_index])
                # shorten the remaining input and word IDs
                sample_input_ids = sample_input_ids[prev_end_index:]
                sample_word_indices = sample_word_indices[prev_end_index:]
                prev_end_index = end_index - prev_end_index
        # otherwise keep as is
        else:
            overlapping_ids.append(sample_input_ids)
            window_start_idx.append(0)
            window_to_sample_mapping.append(i)
            window_word_ids.append(sample_word_indices)

    assert len(overlapping_ids) == len(window_start_idx), f"{len(overlapping_ids)}, {len(window_start_idx)}"
    assert len(overlapping_ids) == len(window_to_sample_mapping), f"{len(overlapping_ids)}, {len(window_to_sample_mapping)}"
    overlapping_strings = tokenizer.batch_decode(
        overlapping_ids,
        skip_special_tokens=True,
    )
    return overlapping_strings, window_start_idx, window_to_sample_mapping, window_word_ids


def get_model_logprobs(
    model_addr: str,
    model_id: str,
    input_text: list[str],
    sample_ids: list[str],
    window_word_ids: list[list[int]],
    window_start_idx: list[int],
    window_to_sample_mapping: list[int],
    async_limiter: int,
) -> dict[str, np.ndarray]:
    """
    Args:
        model_addr: address of hosted model
        model_id: name of the model
        input_text: list of texts, documents are split across
            multiple strings if they exceed max model length
        sample_ids: list of sample IDs to identify each text (document IDs)
        window_word_ids: list of word IDs for each window to align token->word logprobs
        window_start_idx: list of token indices to start each window from for each text
            when going from token->word logprobs
        window_to_sample_mapping: list of sample IDs for each window
        async_limiter: number of maximum concurrent requests
    Returns:
        all_word_logprobs: dict of tensors (sample IDs as keys) tensor of logprobs for each word
    """
    nest_asyncio.apply()
    limiter = AsyncLimiter(async_limiter)
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://{model_addr}/v1"
    )

    async def generate_with_limiter(text, add_bos_token=False):
        if model_id in OLMO_MODEL_IDS:
            text = "<|endoftext|>" + text
        async with limiter:
            for _ in range(3):
                try:
                    return await client.completions.create(
                        model=model_id,
                        prompt=text,
                        max_tokens=0,
                        echo=True,
                        logprobs=0
                    )
                # If we encounter a timeout error, it's likely that we've encountered an OOM error
                # This can only be resolved by reducing the async limiter or restarting the model :(
                # So we'll exit with an error message
                except (
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.InternalServerError,
                ) as e:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                    raise SystemExit("Model is either down or overloaded. Please reduce the async limiter or restart the model.")
                except openai.RateLimitError as e:
                    logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
                    await asyncio.sleep(60)
                except Exception as e:
                    logging.warning(e)
                await asyncio.sleep(30)

    async def batch_generate(input_text):
        try:
            logprobs = await tqdm_asyncio.gather(
            *[generate_with_limiter(text) for text in input_text]
        )
        except SystemExit as e:
            sys.exit(e)
        return logprobs

    generations = asyncio.run(batch_generate(input_text))
    all_word_logprobs = {}
    prev_sample_index = -1
    i = 0
    while i < len(generations):
        generation = generations[i]
        sample_index = window_to_sample_mapping[i]
        sample_id = sample_ids[sample_index]
        logprobs = generation.choices[0].logprobs.token_logprobs[1:]
        word_logprobs = token_to_word_logprobs(window_word_ids[i], logprobs, window_start_idx[i])
        if word_logprobs is None:
            logger.info(f"Skipping sample {sample_id}")
            while i < len(generations) and window_to_sample_mapping[i] == sample_index:
                i += 1
            continue
        # if the sample index is the same as the previous one,
        # we append the logprobs to the running list of logprobs
        if sample_index == prev_sample_index:
            # get the previous window logprobs
            sample_prev_logprobs = all_word_logprobs[sample_id]
            # update dictionary after appending current window logprobs
            all_word_logprobs[sample_id] = np.concatenate([sample_prev_logprobs, word_logprobs], axis=0)
        # otherwise, we want to create a new running list of logprobs
        else:
            all_word_logprobs[sample_id] = word_logprobs
        # update the previous sample index
        prev_sample_index = sample_index
        i += 1
    return all_word_logprobs


def get_word_indices(
    text: str,
    sample_id: str,
    input_model_tokenizer,
    output_model_tokenizer,
) -> list[int]:
    """
    Args:
        text: text to align between input and output models
        sample_id: document ID associated with the text
        input_model: model ID for input pre-tokenizer, defaults to longformer
        output_model: path to weights/model ID for output model, defaults to Llama2-70b

    Returns:
        word_ids: list of word ID per token
    """
    word_tuples = input_model_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    words = [word_tuple[0] for word_tuple in word_tuples]
    num_words = len(words)
    word_ids = []
    word_idx = 0
    tokens = output_model_tokenizer.tokenize(text)
    # llama2 adds an additional space at the beginning of the text
    # we need to remove this for this to work
    # sometimes, the added space is its own token
    # if that's the case, we don't include it in the list of tokens to align
    if output_model_tokenizer.name_or_path in LLAMA_MODEL_IDS:
        if tokens[0] == "▁":
            aligned_tokens = tokens[1:]
            # but for aligning the tokens to their indices,
            # we still need to add it as part of the first word
            word_ids.append(word_idx)
        else:
            # otherwise, just remove the space at the beginning
            tokens[0] = tokens[0][1:]
            aligned_tokens = tokens
    else:
        aligned_tokens = tokens
    built_word = ""
    word_tokenized_ids = input_model_tokenizer(words[word_idx], add_special_tokens=False)['input_ids']
    word_from_tokens = input_model_tokenizer.decode(word_tokenized_ids)
    word = word_from_tokens
    for token in aligned_tokens:
        output_model_token_id = output_model_tokenizer(token, add_special_tokens=False)['input_ids']
        token = output_model_tokenizer.decode(output_model_token_id)
        # change whitespace hexadecimal encodings to char literals
        # if we cannot decode, throw out the sample
        # this is for the llama models
        if output_model_tokenizer.name_or_path in LLAMA_MODEL_IDS:
            if token.startswith("<0x") and token.endswith(">"):
                try:
                    token = bytearray.fromhex(token[3:-1]).decode()
                except UnicodeDecodeError:
                    logger.info(f"Unable to decode {token}...throwing out sample {sample_id}")
                    return None
            input_model_tokens = input_model_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(token)
            token = "".join([input_model_token[0] for input_model_token in input_model_tokens])
        built_word += token
        # if the formed word is longer than the original input word
        # we have an alignment issue between tokenizers
        # and need to append the next word to the current word
        while len(built_word) > len(word):
            word_idx += 1
            word_tokenized_ids = input_model_tokenizer(words[word_idx], add_special_tokens=False)['input_ids']
            word_from_tokens = input_model_tokenizer.decode(word_tokenized_ids)
            word += word_from_tokens
        # add the word ID for the current token
        word_ids.append(word_idx)
        # if the words match
        # we can move on to the next input word
        if built_word == word:
            word_idx += 1
            built_word = ""
            if word_idx == num_words:
                break
            word_tokenized_ids = input_model_tokenizer(words[word_idx], add_special_tokens=False)['input_ids']
            word_from_tokens = input_model_tokenizer.decode(word_tokenized_ids)
            word = word_from_tokens
        
    # check that the number of words matches the number we've counted
    if num_words != word_ids[-1] + 1:
        logger.info(f"Number of words do not match for {sample_id}: {num_words}, {word_ids[-1] + 1}")
        return None
    if len(word_ids) != len(tokens):
        logger.info(f"Dimension of word_ids and tokens do not match for {sample_id}: {len(word_ids)}, {len(tokens)}")
        return None
    return word_ids


def token_to_word_logprobs(
    word_ids: list[int],
    text_token_logprobs: list[float],
    start_idx: int,
) -> np.ndarray:
    """
    For a window, adds the log probabilities of tokens in the same word
    and splits the log probs between words if they are part of the same token
    """
    # first check that the windows are aligned
    # if the dimensions do not match,
    # it's almost certain that the offending tokens are at the beginning
    # (they are being split differently when at the beginning of a text vs in the middle)
    if len(text_token_logprobs) != len(word_ids):
        # it's also almost certain that it is an off by one error
        # if not, we'll log and throw out that sample
        len_diff = len(text_token_logprobs) - len(word_ids)
        if abs(len_diff) > 2:
            logger.info(f"Invalid length difference: {len_diff}")
            return None
        # if the number of tokens is greater than the number of word IDs,
        #   we aggregate the logprobs of the extra tokens
        if len_diff > 0:
            first_token_logprob = sum(text_token_logprobs[0:len_diff+1])
            text_token_logprobs = [first_token_logprob] + text_token_logprobs[len_diff+1:]
        # if the number of word IDs is greater than the number of tokens,
        #   we add zeros to the beginning of the token logprobs
        else:
            text_token_logprobs = [0]*abs(len_diff) + text_token_logprobs
    assert len(text_token_logprobs) == len(word_ids), f"{len(text_token_logprobs)}, {len(word_ids)}"

    text_token_logprobs = text_token_logprobs[start_idx:]
    prev_window_last_word_id = -1
    if start_idx != 0:
        prev_window_last_word_id = word_ids[start_idx-1]
    # if the start index is nonzero
    truncated_word_ids = word_ids[start_idx:]
    # we need to account for the case where the first token of a window comprises multiple words
    # this is done by subtracting the first word ID from the previous window's last word ID
    # which is just -1 if the start index is 0 (first window of the sample)
    word_idx_diff = truncated_word_ids[0] - prev_window_last_word_id
    # reset the word ID to start at zero to match with word_logprobs indices
    truncated_word_ids = [word_id - truncated_word_ids[0] for word_id in truncated_word_ids]
    window_num_words = truncated_word_ids[-1] + word_idx_diff
    word_logprobs = []
    prev_word_idx = word_idx_diff * -1
    for i, word_idx in enumerate(truncated_word_ids):
        num_words = word_idx - prev_word_idx
        if num_words == 0:
            # aggregate logprobs of tokens in the same word
            word_logprobs[-1] += text_token_logprobs[i]
        elif num_words == 1:
            word_logprobs.append(text_token_logprobs[i])
        else:
            # split logprobs between words if they are part of the same token
            split_logprob = text_token_logprobs[i] / num_words
            word_logprobs += [split_logprob] * num_words
        prev_word_idx = word_idx
    assert len(word_logprobs) == window_num_words, f"{len(word_logprobs)}, {window_num_words}"
    return np.array(word_logprobs)        # [window_num_words]