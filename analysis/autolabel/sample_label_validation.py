import asyncclick as click
import asyncio
import json
import litellm
import os
import pandas as pd

from tqdm.asyncio import tqdm

from labeling_utils import \
    get_activations_and_wic, sample_context_string

LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL")
MAX_REQUESTS = 10

system_prompt = """I will provide you with the label, the word, and the context in which the word appears. \
The word in context will be surrounded by askterisks (e.g. *word*). \
Keep in mind the word may be whitespace, in which case the asterisks will be surrounding the whitespace. \
Your job is to determine if a word in context is accurately described by a given label.\n \
\nHere is an example:\n\
[Label] "Nouns describing environmental conservation"\n\
[Word] conservation: Efforts in *conservation* are essential for protecting endangered species.\n\
\nYour answer should be in the following format:\n\
Answer: <YES or NO>\n\
Do not provide any additional text after Answer. Only respond with YES or NO in the "Answer" field. \
For the above example, the answer would be YES.\n\
I will now provide you with the label, word, and context. \
Answer with YES or NO in the above format if the word is accurately described by the label:\n\n\
"""

async def get_response(user_prompt: tuple[int, str], labeling_model: str):
    feature = user_prompt[0]
    word_id = user_prompt[1]
    user_prompt_text = user_prompt[2]
    response = await litellm.acompletion(
        api_key=LITELLM_API_KEY,
        base_url=LITELLM_BASE_URL,
        model=labeling_model,
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
    return feature, word_id, response


def extract_answer(content: str) -> dict:
    try:
        answer = content.split("Answer: ")[1].strip()
    except:
        answer = None
    return answer


def format_label_and_wic(
    wic_row: pd.Series,
    label: str,
) -> str:
    label_line = f"[Label] {label}\n"
    context_string = wic_row["context_string"]
    context_string = context_string.replace("* *", "*<SPACE>*")
    context_string = context_string.replace("*  *", "*<SPACE><SPACE>*")
    context_string = context_string.replace("\t", "<TAB>")
    context_string = context_string.replace("\n", "<NEWLINE>")
    word_line = f"[Word] {wic_row['word']}: {wic_row['context_string']}"
    return label_line + word_line


@click.command()
@click.option("--sae_dir", type=click.Path(exists=True))
@click.option("--label_csv", type=str, default=None, help="Path to CSV file with feature labels")
@click.option("--labeling_model", type=str, default="claude-3-5-sonnet-20241022")
@click.option("--k", type=int, default=50)
async def main(
    sae_dir: str,
    label_csv: str = None,
    labeling_model: str = "claude-3-5-sonnet-20241022",
    k: int = 50,
):
    if label_csv == "None":
        label_csv = None

    output_file = os.path.join(f"{sae_dir}/feature_labels_validated", f"sample_annotation.csv")
    top_acts = pd.read_csv(f"{sae_dir}/top-{k}_activations.csv")
    
    wic = []
    if not label_csv:
        validated_feature_labels = pd.read_json(f"{sae_dir}/feature_labels_validated/{labeling_model}.json")
        top_words_in_context = pd.read_json(f"{sae_dir}/top-{k}_words_in_context.json")
        validated_feature_ids = validated_feature_labels.keys()
        validated_feature_ids = [int(x) for x in validated_feature_ids]
        user_prompts = []
        for feature in validated_feature_ids:
            label = validated_feature_labels[feature]["Description"]
            activations_and_wic = get_activations_and_wic(top_acts, top_words_in_context, feature)
            activations_and_wic["context_string"] = activations_and_wic.apply(sample_context_string, axis=1)
            wic.extend(activations_and_wic["context_string"].tolist())
            feature_user_prompts = activations_and_wic.apply(
                lambda x: (feature, x["word_id"], format_label_and_wic(x, label)),
                axis=1
            )
            user_prompts.extend(feature_user_prompts)
    else:
        validated_feature_labels = pd.read_csv(label_csv)
        top_words_in_context = pd.read_json(f"{sae_dir}/top-{k}_words_in_context.json")
        validated_feature_ids = validated_feature_labels["Feature"].values
        user_prompts = []
        for feature in validated_feature_ids:
            label = validated_feature_labels[validated_feature_labels["Feature"] == feature]["Label"].values[0]
            activations_and_wic = get_activations_and_wic(top_acts, top_words_in_context, feature)
            activations_and_wic["context_string"] = activations_and_wic.apply(sample_context_string, axis=1)
            wic.extend(activations_and_wic["context_string"].tolist())
            feature_user_prompts = activations_and_wic.apply(
                lambda x: (feature, x["word_id"], format_label_and_wic(x, label)),
                axis=1
            )
            user_prompts.extend(feature_user_prompts)
    
    labeling_model = f"openai/neulab/{labeling_model}"
    annotations = {
        "feature": [],
        "word_id": [],
        "valid": []
    }
    with tqdm(total=len(user_prompts)) as pbar:
        for i in range(0, len(user_prompts), MAX_REQUESTS):
            try:
                features_ids_responses = await tqdm.gather(*[get_response(user_prompt, labeling_model) for user_prompt in user_prompts[i:i + MAX_REQUESTS]])
                for feature, word_id, response in features_ids_responses:
                    content = (response.choices[0].message.content)
                    answer = extract_answer(content)
                    annotations["feature"].append(feature)
                    annotations["word_id"].append(word_id)
                    annotations["valid"].append(answer)
                pbar.update(MAX_REQUESTS)
            except Exception as e:
                print(e)
                break
    
    annotations_df = pd.DataFrame(annotations)
    annotations_df["context_string"] = wic
    annotations_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    return
            
            
if __name__ == "__main__":
    asyncio.run(main())
