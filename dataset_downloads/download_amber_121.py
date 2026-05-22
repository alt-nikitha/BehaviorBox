import json
import random
from transformers import AutoTokenizer
from datasets import load_dataset

CHECKPOINT_NUM = 121  # Pretraining dataset shard (unseen by ckpt_120)
NUM_SAMPLES = 20000  # Number of random samples to decode
OUTPUT_FILE = f"/data/user_data/nsrikant/bbox_data/data/amber_120_unseen.jsonl"

dataset = load_dataset(
    "LLM360/AmberDatasets",
    data_files=f"train/train_{CHECKPOINT_NUM:03}.jsonl",
    split=None,
)

tokenizer = AutoTokenizer.from_pretrained("LLM360/Amber", revision=f"ckpt_{CHECKPOINT_NUM}")
samples = random.sample(range(len(dataset["train"])), k=NUM_SAMPLES)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, i in enumerate(samples):
        tokens = dataset["train"][i]["token_ids"]
        text = tokenizer.decode(tokens)
        f.write(json.dumps({"id": idx, "text": text}) + "\n")

print(f"Wrote {NUM_SAMPLES} samples to {OUTPUT_FILE}")
