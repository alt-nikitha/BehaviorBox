

import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= CONFIG =================
# CHECKPOINTS = {
#     "olmo2_7b": {
#         "name": "allenai/OLMo-2-1124-7B",
#         "revisions": [
#             "stage1-step850-tokens4B",
#             "stage1-step9000-tokens38B",
#             "stage1-step47000-tokens198B",
#             "stage1-step94000-tokens395B",
#             "stage1-step235000-tokens986B",
#             "stage1-step470000-tokens1972B",
#             "stage1-step705000-tokens2957B",
#             "stage1-step847000-tokens3553B",
#             "stage2-ingredient1-step1000-tokens5B",
#             "stage2-ingredient1-step7000-tokens30B",
#             "main",
#         ],
#     }
# }

# CHECKPOINTS = {
#   "amber": {
#     "revisions": [
#       "ckpt_001", "ckpt_040", "ckpt_080", "ckpt_120", "ckpt_180", "ckpt_240", "ckpt_300"
#       ],
#     }
#   }
# model_key = "amber"
# results_folder = "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_amber"



CHECKPOINTS = {
  "olmo3": {
    "revisions": [
      "stage1-step1000", "stage1-step34000", "stage1-step68000", "stage1-step103000", "stage1-step154000", "stage1-step205000", "stage1-step256000"
      ],
    }
  }
model_key = "olmo3"
results_folder = "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/eval_results_olmo3"


# CHECKPOINTS = {
#   "name": "LLM360/Amber",
#   "checkpoints": [
#     {
#       "name": "ckpt_001",
#       "tokens": "3.6B",
#       "checkpoint_number": 1
#     },
#     {
#       "name": "ckpt_040",
#       "tokens": "144B",
#       "checkpoint_number": 2
#     },
#     {
#       "name": "ckpt_080",
#       "tokens": "288B",
#       "checkpoint_number": 3
#     },
#     {
#       "name": "ckpt_120",
#       "tokens": "432B",
#       "checkpoint_number": 4
#     },
#     {
#       "name": "ckpt_180",
#       "tokens": "648B",
#       "checkpoint_number": 5
#     },
#     {
#       "name": "ckpt_240",
#       "tokens": "864B",
#       "checkpoint_number": 6
#     },
#     {
#       "name": "ckpt_300",
#       "tokens": "1.08T",
#       "checkpoint_number": 7
#     }
#   ]
# }







# results_folder = (
#     f"/home/nsrikant/BehaviorBoxNew/"
#     f"lm-evaluation-harness/eval_results/{model_key}"
# )

# results_folder = "/home/nsrikant/BehaviorBoxNew/lm-evaluation-harness/results/olmo/fast"

out_dir = f"plots/{model_key}"
os.makedirs(out_dir, exist_ok=True)
# =========================================


revisions = CHECKPOINTS[model_key]["revisions"]

dfs = []

for idx, revision in enumerate(revisions):
    revision_dir = os.path.join(results_folder, revision)
    if not os.path.isdir(revision_dir):
        print(f"[WARN] Missing revision dir: {revision_dir}")
        continue

    # Iterate over task folders
    for task_name in os.listdir(revision_dir):
        task_dir = os.path.join(revision_dir, task_name)
        if not os.path.isdir(task_dir):
            continue

        # Find results JSON inside model subfolder
        import glob as _glob
        result_files = _glob.glob(os.path.join(task_dir, "**/results_*.json"), recursive=True)
        if not result_files:
            continue

        with open(sorted(result_files)[-1]) as f:
            data = __import__('json').load(f)

        # Use the first top-level result key (folder name may differ from lm-eval task name)
        results = data.get("results", {})
        # Filter out subtask keys (start with "-") and pick the first real task
        top_keys = [k for k in results if not k.startswith("-")]
        if not top_keys:
            continue
        metrics = results[top_keys[0]]
        for metric_key, value in metrics.items():
            if metric_key == "alias" or "stderr" in metric_key:
                continue
            # Find matching stderr key: same name with "stderr" inserted
            # e.g. "acc,none" -> "acc_stderr,none", "exact_match,remove_whitespace" -> "exact_match_stderr,remove_whitespace"
            parts = metric_key.split(",", 1)
            stderr_key = parts[0] + "_stderr," + parts[1] if len(parts) == 2 else metric_key + "_stderr"
            stderr_val = metrics.get(stderr_key, None)
            # Clean metric name: strip the suffix after comma
            metric_name = parts[0]
            dfs.append({
                "Tasks": task_name,
                "Metric": metric_name,
                "Value": value,
                "Stderr": stderr_val,
                "checkpoint": revision,
                "checkpoint_idx": idx,
            })

dfs = [pd.DataFrame(dfs)]

df = pd.concat(dfs, ignore_index=True)

# Coerce Value/Stderr to numeric (non-numeric entries become NaN)
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df["Stderr"] = pd.to_numeric(df["Stderr"], errors="coerce")

# --------------------------------------------------
# Clean lm-eval quirks
# --------------------------------------------------

# drop subtasks
df = df[~df["Tasks"].astype(str).str.startswith("-")]

# drop empty task rows
df = df[df["Tasks"].notna()]

# enforce checkpoint ordering
df["checkpoint"] = pd.Categorical(
    df["checkpoint"],
    categories=revisions,
    ordered=True
)

df = df.sort_values("checkpoint_idx")

# --------------------------------------------------
# Plot: one figure per metric
# --------------------------------------------------

for metric, mdf in df.groupby("Metric"):

    plt.figure(figsize=(10, 6))

    for task, g in mdf.groupby("Tasks"):
        g = g.sort_values("checkpoint_idx")

        plt.plot(
            g["checkpoint_idx"],
            g["Value"],
            marker="o",
            linewidth=2,
            label=task
        )

        # stderr band
        if "Stderr" in g:
            plt.fill_between(
                g["checkpoint_idx"],
                g["Value"] - g["Stderr"],
                g["Value"] + g["Stderr"],
                alpha=0.2
            )

    plt.xticks(
        ticks=range(len(revisions)),
        labels=revisions,
        rotation=45,
        ha="right"
    )

    plt.xlabel("Checkpoint")
    plt.ylabel(metric)
    plt.title(f"{model_key}: {metric} over checkpoints")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{metric}_evolution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")
