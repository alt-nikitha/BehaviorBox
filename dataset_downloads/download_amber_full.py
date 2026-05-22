import os
from transformers import AutoTokenizer
from datasets import load_dataset

CHECKPOINT_NUM = 310
CHECKPOINT_NUM_TO_SAVE = 300
NUM_WORKERS = 16 # Adjust based on your CPU
BATCH_SIZE = 1000

if __name__ == "__main__":
    # 1. Load dataset with streaming or mmap (default)
    dataset = load_dataset(
        "LLM360/AmberDatasets",
        data_files=f"train/train_{CHECKPOINT_NUM:03}.jsonl",
        split="train",
    )

    # 2. Setup output path
    OUTPUT_FILE = f"/data/user_data/nsrikant/bbox_data/data/amber_{CHECKPOINT_NUM_TO_SAVE}_unseen_full.jsonl"

    # 3. Load Fast Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "LLM360/Amber", 
        revision=f"ckpt_{CHECKPOINT_NUM}",
        use_fast=True 
    )

    # 4. Use the library's internal multiprocessing
    # This avoids the overhead of manual Pool/Queue management
    decoded_ds = dataset.map(
        lambda x: {"text": tokenizer.batch_decode(x["token_ids"], skip_special_tokens=False)},
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_WORKERS,
        remove_columns=dataset.column_names, # Drop token_ids to save RAM
        desc="Decoding with Fast Tokenizer"
    )

    # 5. Optimized multi-threaded save
    print(f"Saving to {OUTPUT_FILE}...")
    decoded_ds.to_json(OUTPUT_FILE, lines=True)