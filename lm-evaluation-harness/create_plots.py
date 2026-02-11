import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
ROOT_DIR = './eval_results'
SAVE_DIR = os.path.join(ROOT_DIR, 'plots')

# Priority for Harness v0.4 metrics
METRIC_PRIORITY = [
    'exact_match,flexible-extract', 
    'exact_match,strict-match',
    'acc_norm,none',
    'f1,none',
    'exact,none',
    'acc,none'
]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

TASK_ALIASES = {
    'naturalqs': ['naturalqs', 'nq_open', 'natural_questions'],
    'csqa': ['csqa', 'commonsense_qa'],
    'medqa': ['medqa', 'medqa_4options'],
    'squad': ['squad', 'squadv2'],
    'lambada': ['lambada', 'lambada_openai', 'lambada_standard'],
    'gsm8k': ['gsm8k', 'gsm8k_cot', 'gsm8k_main']
}

TASK_CATEGORIES = {
    'mcq': [
        'arc_challenge', 'bbh', 'csqa', 'hellaswag', 'medmcqa', 'medqa', 
        'mmlu_humanities', 'mmlu_other', 'mmlu_pro', 'mmlu_social_sciences', 
        'mmlu_stem', 'piqa', 'sciq', 'winogrande'
    ],
    'generation': ['coqa', 'drop', 'gsm8k', 'naturalqs', 'squad'],
    'language_modeling': ['blimp', 'lambada']
}

ALL_TARGET_TASKS = {task for cat in TASK_CATEGORIES.values() for task in cat}

def get_sort_key(name):
    if name == 'main': return (float('inf'), float('inf'))
    stage_match = re.search(r'stage(\d+)', name)
    step_match = re.search(r'step(\d+)', name)
    return (int(stage_match.group(1)) if stage_match else 0, 
            int(step_match.group(1)) if step_match else 0)

# 1. Sort Checkpoints
folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f)) and f != 'plots']
sorted_ckpts = sorted(folders, key=get_sort_key)

# 2. Extract Data
task_data = defaultdict(dict)
task_to_metric = {}

for ckpt in sorted_ckpts:
    ckpt_path = os.path.join(ROOT_DIR, ckpt)
    for root, _, files in os.walk(ckpt_path):
        norm_root = root.replace('\\', '/').lower()
        found_main_task = next((t for t in ALL_TARGET_TASKS if any(f"/{a}" in norm_root or norm_root.endswith(a) for a in TASK_ALIASES.get(t, [t]))), None)
        if not found_main_task: continue

        for file in files:
            if file.startswith("results_") and file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        data = json.load(f)
                        res = data.get('results', {})
                        metrics = next((res[a] for a in TASK_ALIASES.get(found_main_task, [found_main_task]) if a in res), res)
                        
                        if isinstance(metrics, dict):
                            chosen_k = next((p for p in METRIC_PRIORITY if p in metrics), None)
                            if not chosen_k:
                                chosen_k = next((k for kw in ['exact_match', 'acc_norm', 'f1', 'acc'] for k in metrics if kw in k and 'stderr' not in k), None)
                            
                            if chosen_k:
                                task_data[found_main_task][ckpt] = metrics[chosen_k]
                                task_to_metric[found_main_task] = chosen_k
                except: continue

# 3. Aggregation and Dynamic Sub-grouping
overall_json = {}
plot_subgroups = defaultdict(list)

for task in ALL_TARGET_TASKS:
    if task in task_data:
        vals = [task_data[task][ckpt] for ckpt in sorted_ckpts if ckpt in task_data[task]]
        overall_json[task] = {"metric_name": task_to_metric[task], "checkpoint_performances": vals}
        
        # Determine base category
        base_cat = next((c for c, tks in TASK_CATEGORIES.items() if task in tks), 'other')
        
        # Split based on scale (Percentage vs Decimal)
        # We check the 'main' value or the last available value
        last_val = vals[-1] if vals else 0
        scale = "percentage_scale" if last_val > 1.1 else "decimal_scale"
        
        plot_subgroups[f"{base_cat}_{scale}"].append(task)

# 4. Save and Plot with Specific Y-axis ranges
for full_cat_name, tasks in plot_subgroups.items():
    plt.figure(figsize=(12, 7))
    for t in sorted(tasks):
        x = [c for c in sorted_ckpts if c in task_data[t]]
        y = [task_data[t][c] for c in x]
        plt.plot(x, y, marker='o', markersize=4, label=t)

    plt.title(f"Evaluation: {full_cat_name.replace('_', ' ').upper()}")
    plt.xticks(rotation=45, ha='right')
    
    # Adjust Y-axis for better visibility
    if "percentage_scale" in full_cat_name:
        plt.ylabel("Score (0-100)")
        plt.ylim(-2, 102) # Standard padding for percentage
    else:
        plt.ylabel("Score (0.0-1.0)")
        plt.ylim(-0.02, 1.02)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{full_cat_name}.png"), dpi=300)
    plt.show()

with open(os.path.join(ROOT_DIR, 'overall_results.json'), 'w') as f:
    json.dump(overall_json, f, indent=4)

print(f"Success: Plots split by scale. Files saved in {SAVE_DIR}")