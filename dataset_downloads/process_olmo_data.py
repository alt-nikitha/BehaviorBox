import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import json

DATA_DIR = Path("/data/user_data/nsrikant/bbox_data/olmo3_eval")
OUTPUT_FILE = "/data/user_data/nsrikant/bbox_data/data/olmo_validation_texts.jsonl"

tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")


        
#         for i, tokens in enumerate(data):
#             text = tokenizer.decode(tokens, skip_special_tokens=True)
#             record = {
#                 "dataset": dataset_name,
#                 "id": i,
#                 "text": text
#             }
#             out.write(json.dumps(record) + "\n")

# print(f"Saved to {OUTPUT_FILE}")



tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

# Sequence length used in OLMo (typically 4096 or 2048)
SEQ_LENGTH = 4096

with open(OUTPUT_FILE, 'w') as out:
    for npy_file in sorted(DATA_DIR.rglob("*.npy")):
        dataset_name = npy_file.parent.parent.name
        print(f"Processing: {dataset_name} - {npy_file.name}")
        
        # Load as raw memory-mapped uint32 array
        data = np.memmap(npy_file, dtype=np.uint32, mode='r')
        
        # Reshape into sequences
        num_sequences = len(data) // SEQ_LENGTH
        data = data[:num_sequences * SEQ_LENGTH].reshape(num_sequences, SEQ_LENGTH)
        
        print(f"  Found {num_sequences} sequences")
        
        for i in range(num_sequences):
            tokens = data[i]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            record = {
                "dataset": dataset_name,
                "id": f"{dataset_name}_{npy_file.stem}_{i}",
                "text": text
            }
            out.write(json.dumps(record) + "\n")

print(f"Saved to {OUTPUT_FILE}")