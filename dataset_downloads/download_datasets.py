import os
from datasets import load_dataset

os.environ["DATA_DIR"] = "/data/user_data/nsrikant"


# Load Dolma v1.7.1 as a Parquet dataset
# test_data = load_dataset(
#     "allenai/dolma",
#     split="test",
#     trust_remote_code=True,
#     streaming=True
# )

from datasets import load_dataset
dataset = load_dataset(
  "json",
  data_files="path/to/dolma-v1_5/*.json.gz",
  streaming=True,
  split="test"
)





# Print some examples
for item in dataset:
    print(item)

