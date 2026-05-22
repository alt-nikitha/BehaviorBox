import asyncclick as click
import asyncio
import json
import litellm
import os
import pandas as pd
import numpy as np

from labeling_utils import \
    get_relevant_features, get_activations_and_wic, format_context_string
from tqdm.asyncio import tqdm

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL")
# MAX_REQUESTS = 10
MAX_REQUESTS = 100

system_prompt = """Your job is to determine if a group of words (surrounded by asterisks, e.g. *word*) in specific contexts form a coherent group that is accurately described by a given label. \
I will provide you with a list of words surrounded by asterisks and the context in which they appear, usually within a sentence or a block of text. \
Each word and how it appears in context will be its own item in a list.\n \
\n\
Coherence can come from EITHER:\n \
\t(a) the ACTIVATING WORDS sharing a theme — e.g., they belong to the same semantic field, grammatical category, morphological pattern, or stylistic role; OR\n \
\t(b) the SURROUNDING CONTEXTS sharing a theme — e.g., they all come from the same topic, domain, register, or genre, even when the activating words themselves are generic (stop-words, punctuation, common verbs).\n \
A label is accurate if it describes such a theme (word-based OR context-based). Do NOT score a label as inaccurate just because the activating words are generic — first check whether the contexts share a consistent topic, domain, or register.\n \
\n \
Your job is to score the label by providing a numerical score (0 to 3, or -1). \
Scores are defined as follows:\n \
\t- 0: The label is not accurate and the items (words + their contexts) do not form any coherent group.\n \
\t- 1: The label is not accurate, but the items DO form a coherent group (via either activating-word theme OR context theme).\n \
\t- 2: The label is accurate, but fails to capture a more specific trend.\n \
\t- 3: The label is accurate and captures a specific trend.\n \
\t- -1: There are two or more distinct coherent sub-groups (multi-themed feature).\n\n \
If you give a score of 1 or 2, provide an alternative label that you believe would be more accurate. When the coherence is context-based, the label should describe the context's theme (topic/domain/register), not the activating word. \
If you give a score of -1, provide a label for each sub-group. Each label should be separated with <SEP>. \
Labels should be precise, concise, and accurate, ideally a single sentence each. \
Otherwise, leave the alternative label field blank.\n\n\
Provide your answer in the following format, be sure to include both "Score" and "Label" fields:\n\n\
<BEGIN ANSWER>\n\
Score: <a number between 0-3 or -1>\n\
Label: <label(s) if score is 1, 2, or -1, empty otherwise>\n\
<END ANSWER>\n\n\
Do not provide any additional text after <END ANSWER>. \
Only respond with a number between 0 and 3 or -1 in the Score field. \
The label should NOT refer to the asterisks, those are only there to help you identify the words. \
If there are double asterisks in the text, assume the word of interest is the whitespace between them. \n\n\
Please score the following list of words and their label, and provide a new label if necessary:\n\n\
"""

async def get_response(user_prompt: tuple[int, str], labeling_model: str):
    feature = user_prompt[0]
    user_prompt_text = user_prompt[1]
    orig_label = user_prompt[2]
    try:
        response = await litellm.acompletion(
            api_key=LITELLM_API_KEY,
            base_url=LITELLM_BASE_URL,
            model="litellm_proxy/"+labeling_model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful data annotation assistant.",
                        },
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                },
                {"role": "user", "content": user_prompt_text},
            ]
        )
        return feature, response, orig_label
    except Exception as e:
        print(f"Skipping feature {feature}: {e}")
        return feature, None, orig_label

def get_relevant_labels(
    model_names: list[str],
    feature_metrics: pd.DataFrame,
    feature_labels: dict,
) -> pd.DataFrame:
    relevant_features = get_relevant_features(feature_metrics)
    feature_metrics = feature_metrics[feature_metrics["feature"].isin(relevant_features)]
    coherent_features = [x for x in feature_labels.keys() if feature_labels[x]["Coherent"] in ("YES", "MULTI")]
    feature_metrics = feature_metrics[feature_metrics["feature"].isin(coherent_features)]
    feature_metrics['prob_avg_ranks'] = feature_metrics['prob_avg_ranks'].apply(lambda x: eval(x))
    # model_1_win_features = feature_metrics[feature_metrics["prob_median_diff"] > 0]
    # model_1_win_features.sort_values(by="prob_median_diff", ascending=False, inplace=True)
    # model_1_win_features["label"] = model_1_win_features["feature"].apply(lambda x: feature_labels[x]["Description"])
    # model_1_win_features["model"] = model_names[0]
    
    # model_2_win_features = feature_metrics[feature_metrics["prob_median_diff"] < 0]
    # model_2_win_features.sort_values(by="prob_median_diff", ascending=True, inplace=True)
    # model_2_win_features["label"] = model_2_win_features["feature"].apply(lambda x: feature_labels[x]["Description"])
    # model_2_win_features["model"] = model_names[1]
    
    feature_metrics["winning_model"] = feature_metrics["prob_avg_ranks"].apply(lambda x: np.argmin(x))
    feature_metrics["winning_rank"] = feature_metrics["prob_avg_ranks"].apply(lambda x: np.min(x))
    all_model_win_features = []
    for i,model_name in enumerate(model_names):
        model_win_features = feature_metrics[feature_metrics['winning_model']==i]
        model_win_features.sort_values(by="winning_rank", ascending=True, inplace=True)
        model_win_features["label"] = model_win_features["feature"].apply(lambda x: feature_labels[x]["Description"])
        model_win_features["model"] = model_names[i]
        all_model_win_features.append(model_win_features)
        


    
    # label_df = pd.concat([model_1_win_features, model_2_win_features])
    label_df = pd.concat(all_model_win_features)

    return label_df


def extract_response(content: str) -> dict:
    try:
        response = {}
        answer = content.split("<BEGIN ANSWER>")[1].split("<END ANSWER>")[0]
        response["Score"] = answer.split("Score:")[1].split("\n")[0].strip()
        try:
            new_label = answer.split("Label:")[1].split("\n")[0].strip()
            response["Description"] = new_label
            return response
        except TypeError:
            new_label = None
            response["Description"] = new_label
            return response
    except Exception as e:
        print(e)
        print(content)
        response = None
    return response


@click.command()
@click.option("--sae_dir", type=click.Path(exists=True))
@click.option("--labeling_model", type=str)
@click.option("--k", type=int, default=50)
@click.option("--min_acts", type=int, default=10, help="Skip features with fewer than this many top activations")
async def main(
    sae_dir: str,
    labeling_model: str = "claude-3-5-sonnet-20241022",
    k: int = 50,
    min_acts: int = 50,
):
    if not os.path.exists(f"{sae_dir}/feature_labels_validated"):
        os.makedirs(f"{sae_dir}/feature_labels_validated")

    labeling_model_for_file = labeling_model.replace("/", "-")
    top_acts = pd.read_csv(f"{sae_dir}/top-{k}_activations.csv")
    with open(f"{sae_dir}/top-{k}_words_in_context.json", "r") as f:
        top_words_in_context = json.load(f)
    with open(f"{sae_dir}/feature_labels/{labeling_model_for_file}.json", "r") as f:
        raw_labels = json.load(f)
        feature_labels = pd.DataFrame({int(k): v for k, v in raw_labels.items()})
    sae_cfg = json.load(open(f"{sae_dir}/config.json", "r"))
    model_names = sae_cfg["model_names"]
    # model_string = "_".join(model_names)
    # model_string = "n_moreearly_models"

    feature_metrics = pd.read_csv(f"{sae_dir}/feature_metrics.csv")
    
    output_file = os.path.join(f"{sae_dir}/feature_labels_validated", f"{labeling_model_for_file}.json")
    if not os.path.exists(output_file):
        feature_responses = {}
    else:
        try:
            feature_responses = json.load(open(output_file, "r"))
            print(feature_responses)
        except Exception as e:
            feature_responses = {}
    
    label_df = get_relevant_labels(model_names, feature_metrics, feature_labels)
    features = label_df["feature"].values
    all_user_prompts = []
    for feature in features:
        feature_acts = get_activations_and_wic(top_acts, top_words_in_context, feature)
        if feature_acts is None or len(feature_acts.index) < min_acts:
            continue
        contexts = format_context_string(feature_acts, num_words=20)
        orig_label = label_df[label_df["feature"] == feature]["label"].values[0]
        orig_label_str = f"ORIGINAL LABEL: {orig_label}\n"
        user_prompt = f"{orig_label_str}{contexts}"
        all_user_prompts.append((feature, user_prompt, orig_label))
    
    # labeling_model = f"openai/neulab/{labeling_model}"
    with tqdm(total=len(all_user_prompts)) as pbar:
        for i in range(0, len(all_user_prompts), MAX_REQUESTS):
            last_prompt = min(i+MAX_REQUESTS, len(all_user_prompts)-1)
            if str(all_user_prompts[last_prompt][0]) in feature_responses.keys():
                pbar.update(MAX_REQUESTS)
                continue
            features_and_responses = await tqdm.gather(*[get_response(user_prompt, labeling_model) for user_prompt in all_user_prompts[i:i+MAX_REQUESTS]])
            for feature, response, orig_label in features_and_responses:
                feature = int(feature)
                if response is None:
                    continue
                try:
                    content = response.choices[0].message.content
                    response = extract_response(content)
                    if response is None:
                        continue
                    if response["Score"] == "3":
                        response["Description"] = orig_label

                    response["Winning Rank"] = int(label_df[label_df["feature"] == feature]["winning_rank"].values[0])

                    mean_ranks = label_df[label_df["feature"] == feature]["prob_avg_ranks"].values[0]
                    median_ranks = label_df[label_df["feature"] == feature]["prob_median_ranks"].values[0]
                    avg_probs = label_df[label_df["feature"] == feature]["prob_means"].values[0]
                    median_probs = label_df[label_df["feature"] == feature]["prob_medians"].values[0]
                    avg_logprobs = label_df[label_df["feature"] == feature]["logprob_means"].values[0]
                    median_logprobs = label_df[label_df["feature"] == feature]["logprob_medians"].values[0]

                    if isinstance(median_ranks, np.ndarray):
                        mean_ranks = mean_ranks.tolist()
                    if isinstance(median_ranks, np.ndarray):
                        median_ranks = median_ranks.tolist()
                    if isinstance(avg_probs, np.ndarray):
                        avg_probs = avg_probs.tolist()
                    if isinstance(median_probs, np.ndarray):
                        median_probs = median_probs.tolist()
                    if isinstance(avg_logprobs, np.ndarray):
                        avg_logprobs = avg_logprobs.tolist()
                    if isinstance(median_logprobs, np.ndarray):
                        median_logprobs = median_logprobs.tolist()

                    response["Mean Ranks"] = json.dumps(mean_ranks)
                    response["Median Ranks"] = json.dumps(median_ranks)
                    response["Avg Probs"] = json.dumps(avg_probs)
                    response["Median Probs"] = json.dumps(median_probs)
                    response["Avg LogProbs"] = json.dumps(avg_logprobs)
                    response["Median LogProbs"] = json.dumps(median_logprobs)

                    response["Model"] = label_df[label_df["feature"] == feature]["model"].values[0]
                    feature_responses[feature] = response
                except Exception as e:
                    print(f"Failed to process feature {feature}: {e}")
            pbar.update(MAX_REQUESTS)

    with open(output_file, "w") as f:
        json.dump(feature_responses, f, indent=4)
    return

if __name__ == "__main__":
    asyncio.run(main())
