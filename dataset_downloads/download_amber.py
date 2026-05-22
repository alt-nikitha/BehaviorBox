import json
import random
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

CHECKPOINT_NUM = 310  # Pretraining dataset for checkpoint
NUM_SAMPLES = 20000  # Number of random samples to decode
CHECKPOINT_NUM_TO_SAVE = 300
sample = True



dataset = load_dataset(
    "LLM360/AmberDatasets",
    data_files=f"train/train_{CHECKPOINT_NUM:03}.jsonl",
    split=None,
)

tokenizer = AutoTokenizer.from_pretrained("LLM360/Amber", revision=f"ckpt_{CHECKPOINT_NUM}")
if sample:
    OUTPUT_FILE = f"/data/user_data/nsrikant/bbox_data/data/amber_{CHECKPOINT_NUM_TO_SAVE}_unseen_with_domain.jsonl"

    samples = random.sample(range(len(dataset["train"])), k=NUM_SAMPLES)
else:
    OUTPUT_FILE = f"/data/user_data/nsrikant/bbox_data/data/amber_{CHECKPOINT_NUM_TO_SAVE}_unseen_full.jsonl"
    samples = range(len(dataset["train"]))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, i in tqdm(enumerate(samples), total=len(dataset["train"])):
        tokens = dataset["train"][i]["token_ids"]
        domain = dataset["train"][i]["source"]
        text = tokenizer.decode(tokens)
        f.write(json.dumps({"id": idx, "text": text, "domain": domain}) + "\n")

print(f"Wrote {NUM_SAMPLES} samples to {OUTPUT_FILE}")
