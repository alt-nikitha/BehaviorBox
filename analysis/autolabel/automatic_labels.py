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
# MAX_REQUESTS = 10
MAX_REQUESTS = 100

system_prompt = """Your job is to determine if a group of words (surrounded by asterisks, e.g. *word*) in specific contexts form a coherent group that can be described concisely.
I will provide you with a list of words surrounded by asterisks and the context in which they appear, usually within a sentence or a block of text.
Each list item will contain exactly one word marked with asterisks and the surrounding text showing how it is used.

Here are some examples:
- conservation: Efforts in *conservation* are essential for protecting endangered species.
- habitat: The loss of *habitat* is a significant threat to biodiversity.
- ecosystem: An * ecosystem* needs a balance of various species to thrive.

Your task is to decide whether the items form a coherent group. Coherence can come from EITHER:
  (a) the ACTIVATING WORDS sharing a theme — e.g., they belong to the same semantic field, grammatical category, morphological pattern, or stylistic role; OR
  (b) the SURROUNDING CONTEXTS sharing a theme — e.g., they all come from the same topic, domain, register, or genre, even when the activating words themselves are generic (stop-words, punctuation, common verbs).

A feature is coherent if EITHER (a) or (b) — or both — holds. When only (b) holds, describe the contextual theme. Do NOT reject a feature just because the activating words look like punctuation, stop-words, or generic vocabulary — first check whether the contexts share a consistent topic, domain, or register.

A feature can be SINGLE-themed (one clear theme covering all items) or MULTI-themed (two or more clearly distinct themes that together cover the items — e.g., half the items are math notation, the other half are code syntax). Multi-themed features are still coherent and should be labeled, not rejected. When a feature has multiple themes, list each theme separately in the Description field, separated by " | ".

A feature is INCOHERENT only when neither the activating words nor the surrounding contexts share any clear theme — i.e., the contexts span unrelated topics AND the activating words have no semantic, grammatical, or stylistic pattern, even when split into subgroups.

If incoherent, indicate "NO" and set the description to "NONE."

Provide your answer in the following format:

<BEGIN ANSWER>
Coherent: <YES, MULTI, or NO>
Description: <if YES or MULTI above, your description here; for MULTI list themes separated by " | "; otherwise NONE>
<END ANSWER>

Do not provide any additional text after <END ANSWER>.
- Respond with YES if the items share a single coherent theme.
- Respond with MULTI if the items split into two or more distinct but each-coherent sub-themes.
- Respond with NO if no theme can be identified.
If you respond with YES or MULTI, you must provide a description in the "Description" field.
Descriptions should be concise — one sentence per theme. When the coherence is context-based, the description should name the context's theme rather than the activating word.
For the conservation/habitat/ecosystem example, the description could be "Nouns describing environmental conservation" (YES, word-theme), "Tokens appearing in passages about biodiversity and environmental protection" (YES, context-theme), or "Nouns about environmental conservation | Tokens within passages about endangered species" (MULTI).
The description should NOT refer to the asterisks, those are only there to help you identify the words.

Please categorize the following list of words and their contexts as coherent or not coherent, and provide a description if needed:

"""

async def get_response(user_prompt: tuple[int, str], labeling_model: str):
    feature = user_prompt[0]
    user_prompt_text = user_prompt[1]
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
        return feature, response
    except Exception as e:
        print(f"Skipping feature {feature}: {e}")
        return feature, None

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
@click.option("--min_acts", type=int, default=10, help="Skip features with fewer than this many top activations")
async def main(
    sae_dir: str,
    labeling_model: str,
    k: int = 50,
    replace: bool = False,
    min_acts: int = 50,
):
    top_acts = pd.read_csv(f"{sae_dir}/top-{k}_activations.csv")
    with open(f"{sae_dir}/top-{k}_words_in_context.json", "r") as f:
        top_words_in_context = json.load(f)
    
    cfg = json.load(open(f"{sae_dir}/config.json", "r"))
    model_names = cfg["model_names"]
    # model_name_string = "_".join(model_names)
    
    
    
    feature_metrics = pd.read_csv(f"{sae_dir}/feature_metrics.csv")
    
    features = get_relevant_features(feature_metrics)
    print(len(features), flush=True)
    
    labeling_model_name = labeling_model.replace("/", "-")
    output_file = os.path.join(f"{sae_dir}/feature_labels", f"{labeling_model_name}.json")
    if not os.path.exists(output_file) or replace:
        feature_responses = {}
    else:
        try:
            feature_responses = json.load(open(output_file, "r"))
            print(f"resuming with {len(feature_responses)} features already labeled")
        except Exception as e:
            feature_responses = {}

    already_labeled = {int(k) for k in feature_responses.keys()}

    all_user_prompts = []
    for feature in features:
        if int(feature) in already_labeled:
            continue
        feature_acts = get_activations_and_wic(top_acts, top_words_in_context, feature)

        if feature_acts is None:
            continue
        if len(feature_acts.index) < min_acts:
            continue
        contexts = format_context_string(feature_acts)
        all_user_prompts.append((feature, contexts))

    with tqdm(total=len(all_user_prompts)) as pbar:
        for i in range(0, len(all_user_prompts), MAX_REQUESTS):
            features_and_responses = await tqdm.gather(*[get_response(user_prompt, labeling_model) for user_prompt in all_user_prompts[i:i+MAX_REQUESTS]])
            for feature, response in features_and_responses:
                feature = int(feature)
                if response is None:
                    continue
                content = response.choices[0].message.content
                parsed = extract_response(content)
                if parsed is not None:
                    feature_responses[feature] = parsed
            pbar.update(MAX_REQUESTS)

    with open(output_file, "w") as f:
        json.dump(feature_responses, f, indent=4)
    return

if __name__ == "__main__":
    asyncio.run(main())
