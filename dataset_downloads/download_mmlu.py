import json
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
def build_text(question, options, choice_letter):
    text = f"Question: {question}\n"
    letters = ["A", "B", "C", "D"]
    for l, opt in zip(letters, options):
        text += f"{l}. {opt}\n"
    text += f"\nAnswer: {choice_letter}"
    return text


def process_cais_mmlu_all(output_file="cais_mmlu_processed.jsonl", small_sample=False):
    # Get all available configs
    configs = get_dataset_config_names("cais/mmlu")
    print("Found configs:", configs)

    letters = ["A", "B", "C", "D"]
    count = 0
    with open(output_file, "w") as f:
        for config in tqdm(configs):
            print(f"\nLoading config: {config}")
            if config == "all":
                continue
            dataset = load_dataset("cais/mmlu", config)
            

            for split_name, split_data in dataset.items():

                
                for idx, sample in enumerate(split_data):
                    if "question" not in sample:
                        break
                    question = sample["question"]
                    options = sample["choices"]
                    correct_answer = sample["answer"]  # "A", "B", "C", "D"
                    
                    # each option becomes one training example
                    for opt_letter in letters:
                        label = "correct" if opt_letter == letters[correct_answer] else "wrong"
                        text = build_text(question, options, opt_letter)

                        record = {
                            "id": f"{config}_{split_name}_{label}_{idx}_{opt_letter}",
                            "text": text,
                            "label": label,
                            "split": config,   # you wanted 'split' to be the subject
                            "pair_id": idx
                        }
                        count += 1
                        

                        f.write(json.dumps(record) + "\n")
                    if small_sample:
                        break
                if small_sample:
                    break

    print(f"\nSaved all {count} processed samples → {output_file}")


if __name__ == "__main__":
    # process_cais_mmlu_all("/data/user_data/nsrikant/bbox_data/data/mmlu_sample.jsonl", small_sample=True)
    process_cais_mmlu_all("/data/user_data/nsrikant/bbox_data/data/mmlu.jsonl", small_sample=False)
    # process_cais_mmlu_all("mmlu_sample.jsonl", small_sample=True)
