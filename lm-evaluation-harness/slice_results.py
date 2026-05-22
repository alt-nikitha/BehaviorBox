#!/usr/bin/env python
"""Slice each task folder's results_*.json so it only contains entries for that task + subtasks."""
import json
import re
from pathlib import Path

ROOT = Path("/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3")
MODEL = "allenai__Olmo-3-1025-7B"

RENAME = {"commonsense_qa": "csqa", "nq_open": "naturalqs", "lambada_openai": "lambada"}
REVERSE_RENAME = {v: k for k, v in RENAME.items()}
GROUP_TASKS = {"bbh", "blimp", "mmlu_stem", "mmlu_social_sciences", "mmlu_other"}

PER_TASK_DICTS = [
    "results", "groups", "group_subtasks", "configs", "versions",
    "n-shot", "higher_is_better", "n-samples", "task_hashes",
]

for ckpt_dir in sorted(ROOT.iterdir()):
    if not ckpt_dir.is_dir():
        continue
    for task_dir in sorted(ckpt_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        model_dir = task_dir / MODEL
        if not model_dir.is_dir():
            continue
        results_files = list(model_dir.glob("results_*.json"))
        sample_files = list(model_dir.glob("samples_*.jsonl"))
        if not results_files or not sample_files:
            continue
        rf = results_files[0]

        subtasks = []
        for s in sample_files:
            m = re.match(r"samples_(.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$", s.name)
            if m:
                subtasks.append(m.group(1))

        folder_name = task_dir.name
        # The "parent" key as it appears in results.json uses the actual lm_eval name
        parent_key = REVERSE_RENAME.get(folder_name, folder_name)

        if parent_key in GROUP_TASKS:
            keep_keys = {parent_key, *subtasks}
            keep_groups = {parent_key}
            keep_group_subtasks = {parent_key: list(subtasks)}
        else:
            # Non-grouped task: parent == subtask
            keep_keys = set(subtasks)
            keep_groups = set()
            keep_group_subtasks = {st: [st] for st in subtasks}

        d = json.loads(rf.read_text())
        for dict_name in PER_TASK_DICTS:
            if dict_name not in d or not isinstance(d[dict_name], dict):
                continue
            if dict_name == "groups":
                d[dict_name] = {k: v for k, v in d[dict_name].items() if k in keep_groups}
            elif dict_name == "group_subtasks":
                d[dict_name] = keep_group_subtasks
            else:
                d[dict_name] = {k: v for k, v in d[dict_name].items() if k in keep_keys}

        rf.write_text(json.dumps(d, indent=2))

    print(f"sliced {ckpt_dir.name}")

print("Done.")
