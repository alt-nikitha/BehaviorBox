

import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= CONFIG =================
CHECKPOINTS = {
    "olmo2_7b": {
        "name": "allenai/OLMo-2-1124-7B",
        "revisions": [
            "stage1-step850-tokens4B",
            "stage1-step9000-tokens38B",
            "stage1-step47000-tokens198B",
            "stage1-step94000-tokens395B",
            "stage1-step235000-tokens986B",
            "stage1-step470000-tokens1972B",
            "stage1-step705000-tokens2957B",
            "stage1-step847000-tokens3553B",
            "stage2-ingredient1-step1000-tokens5B",
            "stage2-ingredient1-step7000-tokens30B",
            "main",
        ],
    }
}

model_key = "olmo2_7b"

results_folder = (
    f"/home/nsrikant/BehaviorBoxNew/"
    f"lm-evaluation-harness/eval_results/{model_key}"
)

out_dir = f"plots/{model_key}"
os.makedirs(out_dir, exist_ok=True)
# =========================================


revisions = CHECKPOINTS[model_key]["revisions"]

dfs = []

for idx, revision in enumerate(revisions):
    csv_path = os.path.join(results_folder, f"{revision}.csv")

    if not os.path.exists(csv_path):
        print(f"[WARN] Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    df["checkpoint"] = revision
    df["checkpoint_idx"] = idx  # preserves correct order

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

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
