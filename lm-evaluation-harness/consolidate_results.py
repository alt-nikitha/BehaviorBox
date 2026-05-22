#!/usr/bin/env python
"""Consolidate per-subtask eval dirs into per-main-task dirs matching the prior layout."""
import re
import shutil
from pathlib import Path

SRC = Path("/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3")
DST = Path("/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3_grouped")
MODEL = "allenai__Olmo-3-1025-7B"

MMLU_TASKS_DIR = Path("/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/lm_eval/tasks/mmlu/default")
TAG_TO_GROUP = {
    "mmlu_stem_tasks": "mmlu_stem",
    "mmlu_social_sciences_tasks": "mmlu_social_sciences",
    "mmlu_other_tasks": "mmlu_other",
    "mmlu_humanities_tasks": "mmlu_humanities",
}

mmlu_subject_to_group = {}
for yaml_file in MMLU_TASKS_DIR.glob("mmlu_*.yaml"):
    text = yaml_file.read_text()
    task_m = re.search(r'"?task"?:\s*"?(mmlu_[^"\s]+)"?', text)
    tag_m = re.search(r'"?tag"?:\s*"?(mmlu_\w+_tasks)"?', text)
    if task_m and tag_m and tag_m.group(1) in TAG_TO_GROUP:
        mmlu_subject_to_group[task_m.group(1)] = TAG_TO_GROUP[tag_m.group(1)]


RENAME = {"commonsense_qa": "csqa", "nq_open": "naturalqs", "lambada_openai": "lambada"}

def parent_task(subtask: str) -> str:
    if subtask.startswith("bbh_cot_fewshot_"):
        return "bbh"
    if subtask.startswith("blimp_"):
        return "blimp"
    if subtask.startswith("mmlu_") and subtask in mmlu_subject_to_group:
        return mmlu_subject_to_group[subtask]
    return RENAME.get(subtask, subtask)


for ckpt_dir in sorted(SRC.iterdir()):
    if not ckpt_dir.is_dir():
        continue
    print(f"== {ckpt_dir.name} ==")
    results_file = None
    by_parent = {}
    for sub_dir in sorted(ckpt_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        model_dir = sub_dir / MODEL
        if not model_dir.is_dir():
            print(f"  [skip] no model dir under {sub_dir.name}")
            continue
        samples = list(model_dir.glob("samples_*.jsonl"))
        if not samples:
            print(f"  [warn] no samples in {sub_dir.name}")
        for s in samples:
            m = re.match(r"samples_(.+)_\d{4}-\d{2}-\d{2}T.*\.jsonl$", s.name)
            if not m:
                print(f"  [warn] cannot parse {s.name}")
                continue
            subtask = m.group(1)
            parent = parent_task(subtask)
            by_parent.setdefault(parent, []).append(s)
        if results_file is None:
            rfs = list(model_dir.glob("results_*.json"))
            if rfs:
                results_file = rfs[0]

    if results_file is None:
        print(f"  [error] no results_*.json found anywhere in {ckpt_dir.name}")
        continue

    for parent, sample_paths in by_parent.items():
        dest = DST / ckpt_dir.name / parent / MODEL
        dest.mkdir(parents=True, exist_ok=True)
        for s in sample_paths:
            shutil.copy2(s, dest / s.name)
        shutil.copy2(results_file, dest / results_file.name)
        print(f"  {parent}: {len(sample_paths)} samples files")

print("Done.")
