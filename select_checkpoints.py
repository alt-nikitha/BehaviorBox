#!/usr/bin/env python3
"""List all available checkpoints from HuggingFace."""

from huggingface_hub import list_repo_refs
import re
import json

def get_checkpoints(repo_id: str = "allenai/Olmo-3-1025-7B") -> list[str]:
    """
    Get all checkpoint branch names for a huggingface model.
    
    Args:
        repo_id: HuggingFace repo ID
    
    Returns:
        List of branch names (checkpoint revisions)
    """
    refs = list_repo_refs(repo_id)
    return [b.name for b in refs.branches]


def parse_olmo2_checkpoints(branches: list[str], ingredient: int = 1) -> list[dict]:
    """
    Parse OLMo2 checkpoint branches into structured format with cumulative tokens.
    Only includes stage1 + specified stage2 ingredient.
    
    Args:
        branches: List of branch names from get_checkpoints()
        ingredient: Which stage2 ingredient to include (1, 2, or 3)
    
    Returns:
        List of dicts sorted by cumulative_tokens
    """
    stage1 = []
    stage2 = []
    
    for branch in branches:
        m1 = re.match(r"stage1-step(\d+)-tokens(\d+)B", branch)
        if m1:
            stage1.append({
                "name": branch,
                "stage": 1,
                "step": int(m1.group(1)),
                "tokens": int(m1.group(2)),
            })
            continue
        
        m2 = re.match(rf"stage2-ingredient{ingredient}-step(\d+)-tokens(\d+)B", branch)
        if m2:
            stage2.append({
                "name": branch,
                "stage": 2,
                "step": int(m2.group(1)),
                "tokens": int(m2.group(2)),
            })
    
    stage1_max = max(c["tokens"] for c in stage1) if stage1 else 0
    
    for c in stage1:
        c["cumulative_tokens"] = c["tokens"]
    
    for c in stage2:
        c["cumulative_tokens"] = stage1_max + c["tokens"]
    
    all_ckpts = stage1 + stage2
    all_ckpts.sort(key=lambda x: x["cumulative_tokens"])
    
    return all_ckpts



def parse_olmo3_checkpoints(branches: list[str], ingredient: int = 1) -> list[dict]:
    """
    Parse OLMo3 checkpoint branches into structured format with cumulative tokens.
    Only includes stage1 + specified stage2 ingredient.
    
    Args:
        branches: List of branch names from get_checkpoints()
        ingredient: Which stage2 ingredient to include (1, 2, or 3)
    
    Returns:
        List of dicts sorted by cumulative_tokens
    """
    stage1 = []
    stage2 = []
    
    for branch in branches:
        m1 = re.match(r"stage1-step(\d+)", branch)
        if m1:
            stage1.append({
                "name": branch,
                "stage": 1,
                "step": int(m1.group(1)),
                
            })
            continue
        
        m2 = re.match(rf"stage2-step(\d+)", branch)
        if m2:
            stage2.append({
                "name": branch,
                "stage": 2,
                "step": int(m2.group(1)),
                
            })
    
    stage1_max = max(c["step"] for c in stage1) if stage1 else 0
    
    for c in stage1:
        c["cumulative_steps"] = c["step"]
    
    for c in stage2:
        c["cumulative_steps"] = stage1_max + c["step"]

    all_ckpts = stage1 + stage2
    all_ckpts.sort(key=lambda x: x["cumulative_steps"])

    return all_ckpts



def select_by_percent(checkpoints: list[dict], percentages: list[float]) -> list[dict]:
    """
    Select checkpoints closest to target percentages.
    
    Args:
        checkpoints: List from parse_olmo2_checkpoints()
        percentages: Target percentages (e.g., [0.1, 1, 5, 10, 25, 50, 75, 100])
    
    Returns:
        List of checkpoints at target percentages, sorted by cumulative_tokens
    """
    if not checkpoints:
        return []
    if "cumulative_tokens" in checkpoints[0]:
        max_tokens = max(c["cumulative_tokens"] for c in checkpoints)
    else:
        max_tokens = max(c["cumulative_steps"] for c in checkpoints)

    selected = []
    seen = set()
    
    for pct in sorted(percentages):
        target = (pct / 100) * max_tokens
        if "cumulative_tokens" in checkpoints[0]:
            closest = min(checkpoints, key=lambda c: abs(c["cumulative_tokens"] - target))
        else:
            closest = min(checkpoints, key=lambda c: abs(c["cumulative_steps"] - target))

        if closest["name"] not in seen:
            entry = closest.copy()
            entry["target_pct"] = pct
            if "cumulative_tokens" in closest:
                entry["actual_pct"] = (closest["cumulative_tokens"] / max_tokens) * 100
            else:
                entry["actual_pct"] = (closest["cumulative_steps"] / max_tokens) * 100
            selected.append(entry)
            seen.add(closest["name"])
    if "cumulative_tokens" in selected[0]:
        selected.sort(key=lambda x: x["cumulative_tokens"])
    else:
        selected.sort(key=lambda x: x["cumulative_steps"])
    return selected


if __name__ == "__main__":
    branches = get_checkpoints("allenai/Olmo-3-1025-7B")
    ckpts = parse_olmo3_checkpoints(branches, ingredient=1)
    
    print(f"Total checkpoints: {len(ckpts)}")
    print(f"Max steps: {ckpts[-1]['cumulative_steps']}\n")
    
    percentages = [0.1, 1, 5, 10, 25, 50, 75, 80, 90, 98]
    selected = select_by_percent(ckpts, percentages)
    checkpoints_info = {}
    checkpoints_info["name"] = "allenai/Olmo-3-1025-7B"
    checkpoints_info["checkpoints"] = selected
    for c in selected:
        print(f"{c['target_pct']:>6}% -> {c['actual_pct']:>6.2f}% | {c['cumulative_steps']:>5} | {c['name']}")
    with open("/home/nsrikant/BehaviorBoxNew/checkpoints_info/olmo3_7b_checkpoints_info.json", "w") as f:
        json.dump(checkpoints_info, f, indent=2)

    with open("/home/nsrikant/BehaviorBoxNew/checkpoints_info/olmo3_7b_checkpoints.txt", "w") as f:
        f.write("\n".join([c['name'] for c in selected]+["main"]))
        