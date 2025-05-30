# BehaviorBox
<img src="media/logo.webp" style="float: left; margin-right: 20px; width: 170px;" alt="BehaviorBox logo">

`BehaviorBox` is a bottom-up automatic behavior discovery method to compare language models on next word prediction. Given a corpus of text, `BehaviorBox` extracts fine-grained, salient features—along with natural language descriptions for each—where one model outperforms another. These features are based on both semantic and structural qualities of the word (via a contextual embedding) as well as how easily a LM can predict it (the probability of that word under the LM). Some example feature labels found by `BehaviorBox`, which describe groups of words that one LM is better at predicting than another, are:
<br clear="all">

> Negative contractions in English informal or vernacular dialogue ("didn't", "wouldn't", "carn't")

> Commas before forms of address in dialogue

> Conditional "were" after "if you"

## Setup
To set up the environment, from the repository root run:
```
pip install -r requirements.txt
```
You'll need to create a `.env` file in `scripts/configs` (see `scripts/env_configs/.env.template` for an example).

## Overview
<img src="media/bbox.png" style="display: block; margin-left: auto; margin-right: auto;" width=700 alt="BehaviorBox pipeline">
<br clear="all">

This repository contains code for the each component of the pipeline:
1. Creating performance-aware embeddings [`data_generation/`]
2. Given a dataset of words and their features, extract human-interpretable features to uncover low-level behavior [`sae/`]
3. Labeling and analysis of the features discovered by the SAE [`analysis/`]

Scripts to run each step are in `scripts/`. 

## Data Generation
The first step is to create a dataset of *performance-aware embeddings*, which are representations that are a concatenation of contextual embeddings and LM probabilities. 

The text used to generate the contextual embeddings and LM probabilities should be in JSON lines format with fields for:
- `id`: A unique ID for the document
- `text`: The text of the document
- (optional) `domain`: The domain that the document belongs to, if available

### Input Features (Word embeddings)
To get input features, from the repository root run:
```bash
# For bash
bash scripts/data_generation/get_input_features.sh \
    --data=/path/to/data.jsonl \
    --output_dir=/path/to/output \
    --batch_size=5

# For Slurm
sbatch scripts/data_generation/get_input_features.sh \
    --data=/path/to/data.jsonl \
    --output_dir=/path/to/output \
    --batch_size=5
```

You will need to adjust batch size depending on the length of the documents being processed as well as your memory constraints (a good starting batch size is 5 or 10). Depending on the length of the text and the amount of available GPU memory, some text may not be able to be processed even with batch size 1.

The expected output of the script for a given `output_dir` is:
```
- <output_dir>
    - input_features
        - file_to_doc.csv (maps document IDs to the .parquet file)
        - longformer_*_*.parquet (* indicate index of document, in order of processing)
```
Depending on the batch size and number of documents processed, there will be a variable number of .parquet files.

#### Obtain LM (log-)probs
To obtain the output features (log probabilities), run:
```bash
bash scripts/data_generation/get_output_features.sh \
    --model_name=olmo2-13b \
    --model_id=allenai/OLMo-2-1124-13B \
    --data=/path/to/data.jsonl \
    --output_dir=/path/to/output \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm     #flag to indicate to run scripts using slurm (recommended if available)
```
This will actually run two separate scripts: one to host a vLLM server and another to query the server to obtain log probabilities. You may need to adjust batch size and async_limiter (number of asynchronous calls to the hosted LM) depending on resources.
The expected output is:
```
- <output_dir>
    - output_features
        - file_to_doc.csv (maps document IDs to the .parquet file)
        - <model_name>-*_*.parquet (* indicate index of document, in order of processing)
```

Note that the parquet files actually save *log-probabilities*; we transform these to probabilities when doing feature extraction.

## Extracting Features
Using the data we generated in the previous step, we can now train a SAE, which decomposes the performance-aware contextual embeddings into a sparse representation. Each dimension of this sparse representation is considered a feature.

### Training and Evaluation
For training and evaluation, you'll need to prepare:

1. An experiment configs JSON file (see `scripts/sae/experiment_configs/example_config.json`) containing:
   - `cache_dir`: Path to cache data
   - `data_dirs`: List of data directory paths (e.g. the `output_dir` specified for data generation)
   - `model_names`: List of model names
   - `output_feature_weight`: Weight for probabilities in training
   - `save_dir`: Where to save the SAE

2. A config JSON file (e.g., `sae/hyperparam_configs/N=3000_k=50.json`) with SAE hyperparameters

Then run:
```bash
# For bash
bash scripts/sae/sae_pipeline.sh \
    --args=/path/to/args.json \
    --config_path=/path/to/config.json

# For Slurm
sbatch scripts/sae/sae_pipeline.sh \
    --args=/path/to/args.json \
    --config_path=/path/to/config.json
```

The expected outputs are:
```
- save_dir/<model_names>_seed=<seed>_ofw=<output_feature_weight>_N=*_k=*lp=*/
    - config.json (SAE config)
    - feature_metrics-<model_names>.csv (metrics per feature)
    - sae.pt (SAE model)
    - top-50_activations.csv
    - top-50_words_in_context.json
```

### Feature Labeling
To generate then validate the labels:
```bash
# For bash
bash scripts/analysis/label_features.sh --sae_dir=/path/to/sae

# For Slurm
sbatch scripts/analysis/label_features.sh --sae_dir=/path/to/sae
```

The `sae_dir` is the directory where the model was saved during training.
