import asyncclick as click
import asyncio
import json
import litellm
import os
import pandas as pd

from labeling_utils import \
    get_relevant_features, get_activations_and_wic, format_context_string
from tqdm.asyncio import tqdm

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL")
MAX_REQUESTS = 10

system_prompt = """Your job is to determine if a group of words (surrounded by asterisks, e.g. *word*) in specific contexts form a coherent group that is accurately described by a given label. \
I will provide you with a list of words surrounded by asterisks and the context in which they appear, usually within a sentence or a block of text. \
Each word and how it appears in context will be its own item in a list.\n \
Your job is to determine if the words form a group that is accurately described by the label by providing a numerical score (0 to 3, and -1). \
Scores are defined as follows:\n \
\t- 0: The label is not accurate and the words do not form any coherent groups.\n \
\t- 1: The label is not accurate, but the words form a coherent group.\n \
\t- 2: The label is accurate, but fails to capture a more specific trend.\n \
\t- 3: The label is accurate and captures a specific trend.\n \
\t- -1: There are two coherent groups.\n\n \
Additionally, if you give a score of 1 or 2, provide an alternative label that you believe would be more accurate. \
If you give a label of -1, provide a label for each group. Each label should be separated with <SEP>. \
This label should be precise, concise, and accurate, ideally a single sentence, \
Otherwise, leave the alternative label field blank.\n\n\
Provide your answer in the following format, be sure to include both "Score" and "Label" fields:\n\n\
<BEGIN ANSWER>\n\
Score: <a number between 1-3 or -1>\n\
Label: <label(s) if original score is 1, 2, or -1, empty otherwise>\n\
<END ANSWER>\n\n\
Do not provide any additional text after <END ANSWER>. \
Only respond a number between 0 and 3 or -1 in the Score field. \
The description should NOT refer to the asterisks, those are only there to help you identify the words. \
If there are double asterisks in the text, assume the word of interest is the whitespace between them. \n\n\
Please score the following list of words and their label, and provide a new label if necessary:\n\n\
"""

async def get_response(user_prompt: tuple[int, str], labeling_model: str):
    feature = user_prompt[0]
    user_prompt_text = user_prompt[1]
    orig_label = user_prompt[2]
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

def get_relevant_labels(
    model_names: list[str],
    feature_metrics: pd.DataFrame,
    feature_labels: dict,
) -> pd.DataFrame:
    relevant_features = get_relevant_features(feature_metrics)
    feature_metrics = feature_metrics[feature_metrics["feature"].isin(relevant_features)]
    coherent_features = [x for x in feature_labels.keys() if feature_labels[x]["Coherent"] == "YES"]
    feature_metrics = feature_metrics[feature_metrics["feature"].isin(coherent_features)]
    
    model_1_win_features = feature_metrics[feature_metrics["prob_median_diff"] > 0]
    model_1_win_features.sort_values(by="prob_median_diff", ascending=False, inplace=True)
    model_1_win_features["label"] = model_1_win_features["feature"].apply(lambda x: feature_labels[x]["Description"])
    model_1_win_features["model"] = model_names[0]
    
    model_2_win_features = feature_metrics[feature_metrics["prob_median_diff"] < 0]
    model_2_win_features.sort_values(by="prob_median_diff", ascending=True, inplace=True)
    model_2_win_features["label"] = model_2_win_features["feature"].apply(lambda x: feature_labels[x]["Description"])
    model_2_win_features["model"] = model_names[1]
    
    label_df = pd.concat([model_1_win_features, model_2_win_features])
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
async def main(
    sae_dir: str,
    labeling_model: str = "claude-3-5-sonnet-20241022",
    k: int = 50,
):
    if not os.path.exists(f"{sae_dir}/feature_labels_validated"):
        os.makedirs(f"{sae_dir}/feature_labels_validated")

    labeling_model_for_file = labeling_model.replace("/", "-")
    top_acts = pd.read_csv(f"{sae_dir}/top-{k}_activations.csv")
    top_words_in_context = pd.read_json(f"{sae_dir}/top-{k}_words_in_context.json")
    feature_labels = pd.read_json(f"{sae_dir}/feature_labels/{labeling_model_for_file}.json")
    sae_cfg = json.load(open(f"{sae_dir}/config.json", "r"))
    model_names = sae_cfg["model_names"]
    model_string = "_".join(model_names)
    feature_metrics = pd.read_csv(f"{sae_dir}/feature_metrics-{model_string}.csv")
    
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
        contexts = format_context_string(feature_acts, num_words=20)
        orig_label = label_df[label_df["feature"] == feature]["label"].values[0]
        orig_label_str = f"ORIGINAL LABEL: {orig_label}\n"
        user_prompt = f"{orig_label_str}{contexts}"
        all_user_prompts.append((feature, user_prompt, orig_label))
    
    # labeling_model = f"openai/neulab/{labeling_model}"
    feature_responses = {}
    with tqdm(total=len(all_user_prompts)) as pbar:
        for i in range(0, len(all_user_prompts), MAX_REQUESTS):
            last_prompt = min(i+MAX_REQUESTS, len(all_user_prompts)-1)
            if str(all_user_prompts[last_prompt][0]) in feature_responses.keys():
                continue
            try:
                features_and_responses = await tqdm.gather(*[get_response(user_prompt, labeling_model) for user_prompt in all_user_prompts[i:i+MAX_REQUESTS]])
                for feature, response, orig_label in features_and_responses:
                    feature = int(feature)
                    content = (response.choices[0].message.content)
                    response = extract_response(content)
                    if response["Score"] == "3":
                        response["Description"] = orig_label
                    response["Mean Prob Diff"] = abs(label_df[label_df["feature"] == feature]["prob_avg_diff"].values[0])
                    response["Median Prob Diff"] = abs(label_df[label_df["feature"] == feature]["prob_median_diff"].values[0])
                    response["Mean Logprob Diff"] = abs(label_df[label_df["feature"] == feature]["logprob_avg_diff"].values[0])
                    response["Median Logprob Diff"] = abs(label_df[label_df["feature"] == feature]["logprob_median_diff"].values[0])
                    response["Diff Consistency"] = label_df[label_df["feature"] == feature]["prob_diff_consistency"].values[0]
                    response["Model"] = label_df[label_df["feature"] == feature]["model"].values[0]
                    feature_responses[feature] = response
                pbar.update(MAX_REQUESTS)
            except Exception as e:
                print(e)
                break

    with open(output_file, "w") as f:
        response_json = json.dumps(feature_responses, indent=4)
        f.write(response_json)
    return

if __name__ == "__main__":
    asyncio.run(main())
