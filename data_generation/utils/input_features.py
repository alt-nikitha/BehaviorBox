import gc
import numpy as np
import torch

def get_input_overlapping_windows(
    input_ids: torch.Tensor,                    # [num_encodings, seq_len]
    attention_masks: torch.Tensor,              # [num_encodings, seq_len]
    overflow_to_sample_mapping: torch.Tensor,   # [num_samples]
    output_word_ids: list[list[int]],           # [num_encodings, seq_len]
    window_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    """
    Function that creates overlapping windows of input encodings, attention masks, and word IDs
    Also throws out a buffer of tokens (half the stride) so that we don't consider the same tokens twice
    Args:
        input_ids: tensor of token IDs [num_encodings, seq_len]
        attention_masks: tensor attention masks [num_encodings, seq_len]
        overflow_to_sample_mapping: list of sample IDs to map tokens to samples
        output_word_ids: list of associated word IDs
            (each list has length equal to number of tokens in that sample)
        window_size: size of window to split encodings into
        stride: stride of window

    Returns:
        overlapped_ids: overlapping windows of token input ids
            (windows contain overlap of size (window_size - stride))
        overlapped_masks: overlapping windows of attention masks
        overlapped_word_ids: overlapping windows of word IDs
        window_to_sample_mapping: updated mapping between window and sample
    """
    num_encodings = input_ids.shape[0]  # number of original encodings of length 4096 tokens each
    num_samples = overflow_to_sample_mapping[-1].item() + 1
    overlapped_ids = []
    overlapped_masks = []
    overlapped_word_ids = []
    encoding_idx = 0
    window_to_sample_mapping = []
    shifts = window_size // stride
    for i in range(num_samples):
        window_to_sample_mapping.append(i)
        overlapped_ids.append(input_ids[encoding_idx])
        overlapped_masks.append(attention_masks[encoding_idx])
        overlapped_word_ids.append(output_word_ids[encoding_idx])
        start_token_idx = stride
        end_token_idx = stride
        encoding_idx += 1
        if encoding_idx == num_encodings:
            break
        # perform sliding window operation on text within a sample
        window = 1
        while overflow_to_sample_mapping[encoding_idx].item() == i:
            window_to_sample_mapping.append(i)
            # if window coincides with original window,
            #   append that window
            if (start_token_idx == 0) and (end_token_idx == 0):
                overlapped_ids.append(input_ids[encoding_idx])
                overlapped_masks.append(attention_masks[encoding_idx])
                overlapped_word_ids.append(output_word_ids[encoding_idx])
            else:
                prev_context = input_ids[encoding_idx-1][start_token_idx:]
                cur_context = input_ids[encoding_idx][:end_token_idx]
                overlapped_id = torch.cat((prev_context, cur_context), 0)
                overlapped_ids.append(overlapped_id)
                prev_mask = attention_masks[encoding_idx-1][start_token_idx:]
                cur_mask = attention_masks[encoding_idx][:end_token_idx]
                overlapped_mask = torch.cat((prev_mask, cur_mask), 0)
                overlapped_masks.append(overlapped_mask)
                prev_word_ids = output_word_ids[encoding_idx-1][start_token_idx:]
                cur_word_ids = output_word_ids[encoding_idx][:end_token_idx]
                word_ids = prev_word_ids + cur_word_ids
                overlapped_word_ids.append(word_ids)
            # update indices
            start_token_idx = (start_token_idx + stride) % (window_size)
            end_token_idx = start_token_idx
            window += 1
            if window != shifts:
                encoding_idx += 1
            else:
                window = 0
            if encoding_idx == num_encodings:
                break
    
    overlapped_ids = torch.vstack(overlapped_ids).int()
    overlapped_masks = torch.vstack(overlapped_masks).int()
    window_to_sample_mapping = torch.Tensor(window_to_sample_mapping).int()
    return overlapped_ids, overlapped_masks, overlapped_word_ids, window_to_sample_mapping


def get_longformer_word_features(
    device,
    tokenizer,
    model,
    input_text: list[str],
    sample_ids: list[str],
    use_sliding_window: bool = True,
) -> tuple[dict[str, list[np.ndarray]], list[str], list[int]]:
    """
    Args:
        device: device to put tensors/model/tokenizer on
        tokenizer: Longformer tokenizer
        model: Longformer model
        input_text: list of texts to get embeddings for
        sample_ids: IDs to identify the sample in source text (document IDs)
        use_sliding_window: whether to use sliding window

    Returns:
        all_word_features: dict with doc IDs as keys and list of longformer embedding tensors as values
        all_words: list of words per document in batch
        all_num_words: list of number of words per document
    """
    # gets the longformer embeddings for each token in texts
    # automatically chunks text into windows of length 4096
    encodings = tokenizer(
        input_text,
        add_special_tokens=False,
        max_length=4096,
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    ).to(device)
    
    output_word_ids = [] # list of length # of windows, each element is a list of word IDs per sample
    all_words = []  # list of length # of samples, each element is a list of words per sample
    all_num_words = []  # list of length # of samples, each element is the number of words per sample
    
    # iterate through each text to create a list of words in each text
    # and the number of words in each text
    for sample_idx in range(len(sample_ids)):
        sample_word_ids = []
        sample_tokens = []
        # iterate through each window
        for encoding_idx in range(len(encodings.input_ids)):
            if encodings.overflow_to_sample_mapping[encoding_idx].item() == sample_idx:
                word_ids = encodings[encoding_idx].word_ids
                # -1 for padding tokens
                mod_word_ids = [-1 if word_id is None else word_id for word_id in word_ids]
                output_word_ids.append(mod_word_ids)
                sample_word_ids += mod_word_ids
                sample_tokens += encodings[encoding_idx].tokens
        num_words = max(sample_word_ids) + 1
        words = [''] * num_words
        for token_id, word_id in enumerate(sample_word_ids):
            if word_id != -1:
                words[word_id] = words[word_id] + sample_tokens[token_id]
        all_words.append(words)
        all_num_words.append(num_words)

    # Longformer max length is 4096
    window_size = 4096
    if use_sliding_window:
        # can change this as necessary
        stride = window_size // 2
        input_ids, attn_masks, output_word_ids, window_to_sample_mapping = get_input_overlapping_windows(
            encodings.input_ids,
            encodings.attention_mask,
            encodings.overflow_to_sample_mapping,
            output_word_ids,
            window_size,
            stride
        )
        del encodings
        gc.collect()
    else:
        input_ids = encodings.input_ids
        attn_masks = encodings.attention_mask
    all_last_token_id = torch.sub(torch.sum(attn_masks, dim=1), 1)

    print("getting embeddings...\n")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attn_masks)
        all_embeddings = outputs.last_hidden_state
    print("getting token->word features...\n")
    if use_sliding_window:
        buffer_size = stride // 2
    all_word_features = token_to_word_input_features(
        device,
        output_word_ids,
        all_last_token_id,
        all_embeddings,
        window_to_sample_mapping,
        sample_ids,
        buffer_size=buffer_size,
    )
    return all_word_features, all_words, all_num_words


def token_to_word_input_features(
    device: str,
    all_word_ids: list[list[int]],                  # [num_windows, seq_len]
    all_last_token_ids: torch.Tensor,               # [num_windows]
    all_text_token_embeddings: torch.Tensor,        # [num_windows, seq_len, hidden_size]
    window_to_sample_mapping: torch.Tensor,         # [num_windows]
    sample_ids: list[str],
    buffer_size: None,
) -> dict[str, list[np.ndarray]]:
    """
    Args:
        device: device to put tensors/model/tokenizer on
        all_word_ids: list of mappings between tokens in a window and corresponding words
        all_last_token_ids: index of last non-pad token in each window
        all_text_token_embeddings: list of model (longformer) embeddings for each window
        window_to_sample_mapping: mapping between windows (index) and samples (value)
        sample_ids: IDs to identify the sample in source text
        buffer_size: size of buffer to throw out at the beginning and end of windows
    Returns:
        input_features: dict with doc IDs as keys and list of word embeddings as values
    """
    hidden_size = all_text_token_embeddings.shape[2]
    all_word_ids = torch.Tensor(all_word_ids).int().unsqueeze(2).repeat(1, 1, hidden_size).to(device)
    # iterate over samples
    num_samples = window_to_sample_mapping[-1].item() + 1
    input_features = {}
    for i in range(num_samples):
        window_indices = (window_to_sample_mapping == i).nonzero().flatten().to(device)
        # number of words in the sample
        last_window_id = window_indices[-1].item()
        last_window_word_ids = all_word_ids[last_window_id]
        num_words = last_window_word_ids[all_last_token_ids[last_window_id]][-1] + 1
        # get embeddings for each word in the sample
        sample_embeddings = torch.index_select(all_text_token_embeddings, 0, window_indices)
        # create index tensor to group tokens in a word together
        sample_word_indices = torch.index_select(all_word_ids, 0, window_indices)
        # if buffer size is not None and there is more than one window,
        # throw out the buffer of tokens and the corresponding word IDs
        trimmed_sample_embeddings = []
        trimmed_sample_word_indices = []
        if buffer_size is not None and sample_embeddings.shape[0] > 1:
            # throw out the last buffer_size tokens and word IDs if they are not part of the last window
            # throw out the first buffer_size tokens and word IDs if they are not part of the first window
            for window_idx in range(sample_embeddings.shape[0]):
                if window_idx == 0:
                    trimmed_sample_embeddings.append(
                        sample_embeddings[window_idx, :-buffer_size, :].squeeze(dim=0)
                    )
                    trimmed_sample_word_indices.append(
                        sample_word_indices[window_idx, :-buffer_size, :].squeeze(dim=0)
                    )
                elif window_idx < sample_embeddings.shape[0] - 1:
                    trimmed_sample_embeddings.append(
                        sample_embeddings[window_idx, buffer_size:-buffer_size, :].squeeze(dim=0)
                    )
                    trimmed_sample_word_indices.append(
                        sample_word_indices[window_idx, buffer_size:-buffer_size, :].squeeze(dim=0)
                    )
                else:
                    trimmed_sample_embeddings.append(
                        sample_embeddings[window_idx, buffer_size:, :].squeeze(dim=0)
                    )
                    trimmed_sample_word_indices.append(
                        sample_word_indices[window_idx, buffer_size:, :].squeeze(dim=0)
                    )
            sample_embeddings = torch.cat(trimmed_sample_embeddings, dim=0).to(device)
            sample_word_indices = torch.cat(trimmed_sample_word_indices, dim=0).to(device)
        else:
            sample_embeddings = sample_embeddings.view(
                sample_embeddings.shape[0] * sample_embeddings.shape[1], hidden_size
            ).to(device)
            sample_word_indices = sample_word_indices.view(
                sample_word_indices.shape[0] * sample_word_indices.shape[1], hidden_size
            ).to(device)
        # get padding index if pad token is present
        min_word_id, min_word_index = torch.min(sample_word_indices, 0)
        pad_token_index = sample_word_indices.shape[0]
        # original pad ID is -1
        if min_word_id[0].item() == -1:
            pad_token_index = min_word_index[0].item()
        # then slice to exclude them
        sample_embeddings = sample_embeddings[:pad_token_index, :]
        sample_word_indices = sample_word_indices[:pad_token_index, :].long()
        sample_word_features = torch.zeros(sample_embeddings.shape, dtype=torch.float16).to(device)
        sample_word_features = sample_word_features.scatter_reduce_(
            0, sample_word_indices, sample_embeddings, "mean", include_self=False
        )
        # [num words, hidden_size]
        sample_word_features = sample_word_features[:num_words, :].numpy(force=True)
        # add to input_features dict
        # where the key is the sample ID, and the value is the input feature tensor
        input_features[sample_ids[i]] = []
        for word_idx in range(num_words):
            input_features[sample_ids[i]].append(sample_word_features[word_idx])
    return input_features