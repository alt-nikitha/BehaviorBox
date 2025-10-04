import asyncclick as click
import asyncio
import json
import litellm
import os
import pandas as pd

from tqdm.asyncio import tqdm

from labeling_utils import \
    get_relevant_features, get_activations_and_wic, format_context_string

# litellm._turn_on_debug()
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL")
MAX_REQUESTS = 10

system_prompt = """Your job is to determine if a group of words (surrounded by asterisks, e.g. *word*) in specific contexts form a coherent group that can be described concisely. \
I will provide you with a list of words surrounded by asterisks and the context in which they appear, usually within a sentence or a block of text. \
Each word and how it appears in context will be its own item in a list.\n \
\nHere are some examples:\n \
- conservation: Efforts in *conservation* are essential for protecting endangered species.\n \
- habitat: The loss of *habitat* is a significant threat to biodiversity.\n \
- ecosystem: An *ecosystem* needs a balance of various species to thrive.\n\n\
Your job is to determine if the words form a coherent group that can be described concisely. \
If the words do form a coherent group, please provide a concise description of the group. \
Provide your answer in the following format:\n\n\
<BEGIN ANSWER>\n\
Coherent: <YES or NO>\n\
Description: <if YES above, your description here; otherwise NONE>\n\
<END ANSWER>\n\n\
Do not provide any additional text after <END ANSWER>. \
Only respond with YES or NO for the "Coherent" field. \
If you respond with YES, you must provide a description in the "Description" field. \
Descriptions should be concise, ideally a single sentence. \
For the above example, descriptions may be something like "Nouns describing environmental conservation" or "Words related to biodiversity". \
Note that groups and descriptions may also pertain to formatting, such as "Punctuation before whitespaces in documents discussing logic" or "Series of whitespaces in documents discussing visual art".\
The description should NOT refer to the asterisks, those are only there to help you identify the words. \n\n\
Please categorize the following list of words and their contexts as coherent or not coherent, and provide a description if needed:\n\n\
"""

async def get_response(user_prompt: tuple[int, str], labeling_model: str):
    feature = user_prompt[0]
    user_prompt_text = user_prompt[1]
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
    # print(f"RESPONSE:   \n{response}")
    return feature, response

def extract_response(content: str) -> dict:
    try:
        response = {}
        answer = content.split("<BEGIN ANSWER>")[1].split("<END ANSWER>")[0]
        response["Coherent"] = answer.split("Coherent: ")[1].split("\n")[0]
        response["Description"] = answer.split("Description: ")[1].split("\n")[0]
    except:
        response = None
    return response


@click.command()
@click.option("--sae_dir", type=click.Path(exists=True))
@click.option("--labeling_model", type=str)
@click.option("--k", type=int, default=50)
@click.option("--replace", type=bool, default=False, help="Replace existing labels")
async def main(
    sae_dir: str,
    labeling_model: str,
    k: int = 50,
    replace: bool = False,
):
    top_acts = pd.read_csv(f"{sae_dir}/top-{k}_activations.csv")
    top_words_in_context = pd.read_json(f"{sae_dir}/top-{k}_words_in_context.json")
    
    cfg = json.load(open(f"{sae_dir}/config.json", "r"))
    model_names = cfg["model_names"]
    model_name_string = "_".join(model_names)
    feature_metrics = pd.read_csv(f"{sae_dir}/feature_metrics-{model_name_string}.csv")
    
    features = get_relevant_features(feature_metrics)
    print(len(features), flush=True)
    
    labeling_model_name = labeling_model.replace("/", "-")
    output_file = os.path.join(f"{sae_dir}/feature_labels", f"{labeling_model_name}.json")
    if not os.path.exists(output_file) or replace:
        feature_responses = {}
        start = 0
    else:
        try:
            feature_responses = json.load(open(output_file, "r"))
            start = int(list(feature_responses.keys())[-1])
            print("continuing from feature", start)
        except Exception as e:
            start = 0
    
    all_user_prompts = []
    for feature in features:
        feature_acts = get_activations_and_wic(top_acts, top_words_in_context, feature)
        
        if feature_acts is None:
            continue
        if len(feature_acts.index) < 10:
            continue
        contexts = format_context_string(feature_acts)
        all_user_prompts.append((feature, contexts))
    
    feature_responses = {}
    with tqdm(total=len(all_user_prompts)) as pbar:
        for i in range(0, len(all_user_prompts), MAX_REQUESTS):
            if start in range(i, i+MAX_REQUESTS) and start > 0:
                continue
            try:
                features_and_responses = await tqdm.gather(*[get_response(user_prompt, labeling_model) for user_prompt in all_user_prompts[i:i+MAX_REQUESTS]])
                print(f"RESPONSE: {features_and_responses}")
                for feature, response in features_and_responses:
                    feature = int(feature)
                    content = (response.choices[0].message.content)
                    response = extract_response(content)
                    if response is not None:
                        feature_responses[feature] = response
                pbar.update(MAX_REQUESTS)
            except Exception as e:
                print(f"Error at feature {i}: {e}")
                break

    with open(output_file, "w") as f:
        response_json = json.dumps(feature_responses, indent=4)
        f.write(response_json)
    return

if __name__ == "__main__":
    asyncio.run(main())
