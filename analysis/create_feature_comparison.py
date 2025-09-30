import os
import glob
import json
import pandas as pd

# Directory containing all feature_labels_validated folders
BASE_DIR = "/home/nsrikant/BehaviorBoxNew/sae_outputs"
PATTERN = os.path.join(BASE_DIR, "*", "_seed=42_ofw=_N=3000_k=50_lp=None", "feature_labels_validated", "*claude*.json")

# Find all validated label files that contain "claude" in the filename
json_files = glob.glob(PATTERN)

# For each file, store: (model_A, model_B) -> list of (feature_id, score, desc) for model_A vs model_B
pairwise_features = {}

for jf in json_files:
    with open(jf, "r") as f:
        data = json.load(f)
    # Try to get the two model names from the feature info
    # Assume all features in the file have the same "Model" field for model_A
    # Try to infer model_B from the filename (after "claude" in the filename)
    # If not possible, fallback to directory name
    model_A = None
    for finfo in data.values():
        model_A = finfo.get("Model")
        if model_A:
            break
    if not model_A:
        model_A = os.path.splitext(os.path.basename(jf))[0]
    # Try to get model_B from filename, e.g. "neulab-claude-sonnet-4-20250514_vs_modelB.json"
    fname = os.path.basename(jf)
    if "_vs_" in fname:
        model_B = fname.split("_vs_")[1].replace(".json", "")
    else:
        # fallback: use parent directory name or "unknown"
        parent = os.path.dirname(jf)
        model_B = os.path.basename(parent)
    # But if the file contains features for both models, we need to split them
    # Let's try to group features by their "Model" field
    features_by_model = {}
    for fid, finfo in data.items():
        m = finfo.get("Model")
        score = finfo.get("Score")
        desc = finfo.get("Description", "")
        if m and score is not None and desc and desc.strip():
            try:
                score_val = float(score)
            except Exception:
                continue
            features_by_model.setdefault(m, []).append((fid, score_val, desc))
    # If there are exactly two models, treat as model_A vs model_B
    models_in_file = list(features_by_model.keys())
    if len(models_in_file) == 2:
        mA, mB = models_in_file
        # For A vs B
        feats_A = sorted(features_by_model[mA], key=lambda x: (-x[1], int(x[0])))[:5]
        pairwise_features[(mA, mB)] = feats_A
        # For B vs A
        feats_B = sorted(features_by_model[mB], key=lambda x: (-x[1], int(x[0])))[:5]
        pairwise_features[(mB, mA)] = feats_B
    elif len(models_in_file) == 1:
        # Only one model, use model_A and model_B as inferred
        feats = sorted(features_by_model[models_in_file[0]], key=lambda x: (-x[1], int(x[0])))[:5]
        pairwise_features[(model_A, model_B)] = feats
    # else: skip files with no valid features

# Collect all unique models
all_models = set()
for (a, b) in pairwise_features.keys():
    all_models.add(a)
    all_models.add(b)
all_models = sorted(all_models)

# Create m x m comparison table, each cell is the features for (A,B)
comparison = pd.DataFrame(index=all_models, columns=all_models)

for model_A in all_models:
    for model_B in all_models:
        feats = pairwise_features.get((model_A, model_B), [])
        formatted = [f"{fid}: {desc} (Score={score})" for fid, score, desc in feats]
        cell_text = "\n".join(formatted)
        comparison.loc[model_A, model_B] = cell_text


comparison.to_excel("/home/nsrikant/BehaviorBoxNew/sae_outputs/feature_comparison_top5_pairwise.xlsx")
