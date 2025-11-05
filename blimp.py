"""
code to evaluate blimp, adapted from camelids/evaluate/blimp.py
"""

import click
import math
import os
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    GPT2LMHeadModel, PreTrainedTokenizer, OPTForCausalLM, LlamaForCausalLM
from typing import Union

from metric_utils import get_threshold_param, get_metric_label
from prompt_utils import add_prefix

if torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"

# All the different templates in blimp
blimp_templates = ["adjunct_island", "anaphor_gender_agreement",
                   "anaphor_number_agreement", "animate_subject_passive",
                   "animate_subject_trans", "causative", "complex_NP_island",
                   "coordinate_structure_constraint_complex_left_branch",
                   "coordinate_structure_constraint_object_extraction",
                   "determiner_noun_agreement_1",
                   "determiner_noun_agreement_2",
                   "determiner_noun_agreement_irregular_1",
                   "determiner_noun_agreement_irregular_2",
                   "determiner_noun_agreement_with_adj_2",
                   "determiner_noun_agreement_with_adj_irregular_1",
                   "determiner_noun_agreement_with_adj_irregular_2",
                   "determiner_noun_agreement_with_adjective_1",
                   "distractor_agreement_relational_noun",
                   "distractor_agreement_relative_clause", "drop_argument",
                   "ellipsis_n_bar_1", "ellipsis_n_bar_2",
                   "existential_there_object_raising",
                   "existential_there_quantifiers_1",
                   "existential_there_quantifiers_2",
                   "existential_there_subject_raising",
                   "expletive_it_object_raising", "inchoative", "intransitive",
                   "irregular_past_participle_adjectives",
                   "irregular_past_participle_verbs",
                   "irregular_plural_subject_verb_agreement_1",
                   "irregular_plural_subject_verb_agreement_2",
                   "left_branch_island_echo_question",
                   "left_branch_island_simple_question",
                   "matrix_question_npi_licensor_present", "npi_present_1",
                   "npi_present_2", "only_npi_licensor_present",
                   "only_npi_scope", "passive_1", "passive_2",
                   "principle_A_c_command", "principle_A_case_1",
                   "principle_A_case_2", "principle_A_domain_1",
                   "principle_A_domain_2", "principle_A_domain_3",
                   "principle_A_reconstruction",
                   "regular_plural_subject_verb_agreement_1",
                   "regular_plural_subject_verb_agreement_2",
                   "sentential_negation_npi_licensor_present",
                   "sentential_negation_npi_scope",
                   "sentential_subject_island",
                   "superlative_quantifiers_1", "superlative_quantifiers_2",
                   "tough_vs_raising_1", "tough_vs_raising_2", "transitive",
                   "wh_island", "wh_questions_object_gap",
                   "wh_questions_subject_gap",
                   "wh_questions_subject_gap_long_distance",
                   "wh_vs_that_no_gap",
                   "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
                   "wh_vs_that_with_gap_long_distance"]


def evaluate_blimp(model: Union[AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM, LlamaForCausalLM],
                   tokenizer: PreTrainedTokenizer,
                   template: str,
                   batch_size: int = 10,
                   metric: str = "seq_logprob",
                   prefix_type: str = None,
                   ) -> dict[str, Union[int, float]]:
    """
    Evaluates a model on BLiMP.

    :param model: A Huggingface model to be evaluated
    :param tokenizer: The tokenizer for the model
    :param templates: Which templates to evaluate on (see above). If
        blank, then the model will be evaluated on all templates
    :param metric: Evaluating metric. By default, compare sequence logprobs.
    :param batch_size: The batch size for evaluation
    """
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    loss_function = nn.CrossEntropyLoss(reduction="none",
                                        ignore_index=tokenizer.pad_token_id)

    strategy_based_metrics = ["top_k", "top_p"]

    results = {}
    # Start test
    good_metric = []
    bad_metric = []
    data = load_dataset("nyu-mll/blimp", template)["train"]
    if prefix_type is not None:
        # Change batch size if we are adding the shuffled prefix
        # because I am too tired to figure out if there's a way to do this batched
        batch_size = 1
        data = data.add_column("add_prefix_good", data["sentence_good"])
        data = data.add_column("add_prefix_bad", data["sentence_bad"])
        data = add_prefix(data, tokenizer, prefix_type)
    # do the same thing for normalizing by sequence length
    if metric == "normalized_seq_logprob":
        batch_size = 1

    with torch.no_grad():
        n_total = len(data)
        for batch in tqdm(data.iter(batch_size=batch_size),
                            total=math.ceil(n_total / batch_size)):
            # Calculate metric for good sentences
            if prefix_type is not None:
                inputs = tokenizer(
                    batch["add_prefix_good"], return_tensors="pt", padding=True
                ).to(device)
                inputs_orig = tokenizer(
                    batch["sentence_good"], return_tensors="pt", padding=True
                ).to(device)
                labels = inputs_orig["input_ids"][:, 1:]
                logits = model(**inputs).logits[:, :-1]     # [batch_size, seq_len, vocab_size]
                # adjust logits so that they do not include the prefix
                label_len = labels.shape[1]
                logits = logits[:, -label_len:]
            else:
                inputs = tokenizer(
                    batch["sentence_good"], return_tensors="pt", padding=True
                ).to(device)
                labels = inputs["input_ids"][:, 1:]
                logits = model(**inputs).logits[:, :-1]     # [batch_size, seq_len, vocab_size]
            if metric in strategy_based_metrics:
                strategy_params = get_threshold_param(
                    metric, logits, labels, tokenizer.pad_token_id
                )
                good_metric += strategy_params.tolist()
            else:
                logits = logits.transpose(1, 2)     # [batch_size, vocab_size, seq_len]
                token_nll = loss_function(logits, labels)
                if metric == "seq_logprob":
                    seq_nll = token_nll.sum(-1)
                    good_metric += seq_nll.tolist()
                if metric == "normalized_seq_logprob":
                    norm_seq_nll = token_nll.sum(-1) / logits.shape[-1]
                    good_metric += norm_seq_nll.tolist()
                elif metric == "lowest_token_logprob":
                    max_nll = torch.max(token_nll, -1)[0]
                    good_metric += max_nll.tolist()

            # Calculate metric for bad sentences
            if prefix_type is not None:
                inputs = tokenizer(
                    batch["add_prefix_bad"], return_tensors="pt", padding=True
                ).to(device)
                inputs_orig = tokenizer(
                    batch["sentence_bad"], return_tensors="pt", padding=True
                ).to(device)
                labels = inputs_orig["input_ids"][:, 1:]
                logits = model(**inputs).logits[:, :-1]     # [batch_size, seq_len, vocab_size]
                # adjust logits so that they do not include the prefix
                label_len = labels.shape[1]
                logits = logits[:, -label_len:]
            else:
                inputs = tokenizer(
                    batch["sentence_bad"], return_tensors="pt", padding=True
                ).to(device)
                labels = inputs["input_ids"][:, 1:]
                logits = model(**inputs).logits[:, :-1]
            if metric in strategy_based_metrics:
                strategy_params = get_threshold_param(
                    metric, logits, labels, tokenizer.pad_token_id
                )
                bad_metric += strategy_params.tolist()
            else:
                logits = logits.transpose(1, 2)     # [batch_size, vocab_size, seq_len]
                token_nll = loss_function(logits, labels)
                if metric == "seq_logprob":
                    seq_nll = token_nll.sum(-1)
                    bad_metric += seq_nll.tolist()
                if metric == "normalized_seq_logprob":
                    norm_seq_nll = token_nll.sum(-1) / logits.shape[-1]
                    bad_metric += norm_seq_nll.tolist()
                elif metric == "lowest_token_logprob":
                    max_nll = torch.max(token_nll, -1)[0]
                    bad_metric += max_nll.tolist()

    results["ID"] = [i for i in range(n_total)]
    metric_label = get_metric_label(metric)
    results[f"good_{metric_label}"] = good_metric
    results[f"bad_{metric_label}"] = bad_metric
    return results


@click.command()
@click.option(
    "--model_name",
    help="Model name",
    type=str,
    required=True,
)
@click.option(
    "--save_dir",
    help="Directory to save results",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--template",
    help="Template in BLiMP to test",
    type=str,
)
@click.option(
    "--metric",
    help="Metric to record. If empty, compare sequence probs",
    default="seq_logprob",
    type=click.Choice([
        "top_k",
        "top_p",
        "lowest_token_logprob",
        "seq_logprob",
        "normalized_seq_logprob",
    ]),
)
@click.option(
    "--prefix_type",
    help="Whether to add prefix",
    default=None,
    type=click.Choice([
        "same_sentence",
        "start_with_period",
        "meta_prefix",
        "linguistic_example_prefix",
        "linguistic_example_prefix_instruct",
        "forced_choice",
        "forced_choice_instruct",
        None
    ]),
)
def main(
    model_name: str,
    save_dir: str = "results/blimp/",
    template: str = None,
    metric: str = "seq_logprob",
    prefix_type: str = None,
):
    if template is None:
        templates = blimp_templates
    else:
        templates = [template]

    model_id = {
        "opt-125m": "facebook/opt-125m",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    }
    assert model_name in model_id.keys()
    tokenizer = AutoTokenizer.from_pretrained(model_id[model_name])
    model = AutoModelForCausalLM.from_pretrained(model_id[model_name]).to(_device)
    if model_name in ["mistral-7b", "mistral-7b-instruct"]:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Add pad token to vocab")
        model.resize_token_embeddings(len(tokenizer))

    if prefix_type is None:
        results_dir = f"{save_dir}/{model_name}/no_prefix/{metric}"
    else:
        results_dir = f"{save_dir}/{model_name}/{prefix_type}/{metric}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for template in templates:
        results = evaluate_blimp(
            model,
            tokenizer,
            template=template,
            metric=metric,
            prefix_type=prefix_type,
        )
        df = pd.DataFrame.from_dict(results)
        print(f"saving results to {results_dir}/{template}-{metric}.csv ...")
        df.to_csv(f"{results_dir}/{template}-{metric}.csv", index=False)
        print(df.head())


if __name__ == "__main__":
    main()
