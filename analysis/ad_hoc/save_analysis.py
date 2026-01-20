import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import re
# ============================================================================
# CONFIG
# ============================================================================
PYTHIA_MODEL = "160m"   # or "6.9b"
SEED = 42
np.random.seed(SEED)

topics_dir = f"/home/nsrikant/BehaviorBoxNew/analysis/topics/Pythia{PYTHIA_MODEL}"
json_path = f"{topics_dir}/data_with_topics.json"

output_root = f"./trend_outputs/Pythia{PYTHIA_MODEL}"
os.makedirs(output_root, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
with open(json_path, "r") as f:
    data = json.load(f)

features = data["features"]
model_names = data["model_names"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_acq_checkpoint(median_probs):
    for idx, prob in enumerate(median_probs):
        if prob is not None and prob >= 0.5:
            return idx
    return -1

def extract_step(s):
    # Extract digits after "step"
    m = re.search(r'step(\d+)', str(s))
    if m:
        return int(m.group(1))
    return None

def sort_by_checkpoint(df, col="acq_checkpoint"):
    df["step_num"] = df[col].apply(extract_step)
    df = df.sort_values("step_num", ascending=True)
    return df

def get_stagnation_checkpoint(median_probs):
    if not median_probs or len(median_probs) < 3:
        return -1
    valid = [p for p in median_probs if p is not None]
    if len(valid) < 3:
        return -1
    start = max(1, len(valid) - len(valid)//3)
    if np.std(valid[start:]) < 0.1:
        return start
    return -1

def get_peak_checkpoint(median_probs):
    valid = [(i, p) for i, p in enumerate(median_probs) if p is not None]
    if not valid:
        return -1
    return max(valid, key=lambda x: x[1])[0]

# ============================================================================
# PROCESS FEATURES
# ============================================================================
processed = []
for f in features:
    median = f.get("median_probs", [])
    acq = get_acq_checkpoint(median)
    trend = f.get("trend_median_probs")

    processed.append({
        "feature_id": f["feature_id"],
        "description": f.get("description", ""),
        "median_probs": median,
        "topic": f.get("topic", "Other"),
        "acq_checkpoint": acq,
        "stagnation_checkpoint": get_stagnation_checkpoint(median) if acq >= 0 else -1,
        "peak_checkpoint": get_peak_checkpoint(median) if acq >= 0 else -1,
        "pearson": f.get("pearson_corr"),
        "spearman": f.get("spearman_corr"),
        "trend": trend,
        "acquired": acq >= 0
    })

# ============================================================================
# GROUP BY TREND
# ============================================================================
all_trends = sorted(set([f["trend"] for f in processed]))

for trend in all_trends:
    print(f"Processing trend: {trend}")

    trend_dir = f"{output_root}/{trend}"
    os.makedirs(trend_dir, exist_ok=True)

    acquired = [f for f in processed if f["trend"] == trend and f["acquired"]]
    not_acq = [f for f in processed if f["trend"] == trend and not f["acquired"]]

    # =======================================================================
    # 1. ACQUIRED – Topic Acquisition Summary
    # =======================================================================
    if acquired:
        topic_groups = defaultdict(list)
        for f in acquired:
            topic_groups[f["topic"]].append(f)

        rows = []
        for topic, feats in topic_groups.items():
            earliest = min(f["acq_checkpoint"] for f in feats)
            per = [f["pearson"] for f in feats if f["pearson"] is not None]

            rows.append({
                "Topic": topic,
                "Num Features": len(feats),
                "First Acquired Checkpoint": earliest,
                "Checkpoint Name": model_names[earliest],
                "Avg |Pearson|": np.mean(np.abs(per)) if per else 0,
                "High Corr Count (>0.7)": sum(abs(p) > 0.7 for p in per)
            })

        df_acq = pd.DataFrame(rows)
        df_acq.sort_values("First Acquired Checkpoint", inplace=True)
        df_acq.to_csv(f"{trend_dir}/acquired_topic_summary.csv", index=False)

    # =======================================================================
    # 2. ACQUIRED – Detailed Feature List
    # =======================================================================
        df_feat = pd.DataFrame([
            {
                "Feature ID": f["feature_id"],
                "Topic": f["topic"],
                "Acquired At": model_names[f["acq_checkpoint"]],
                "Pearson": f["pearson"],
                "Spearman": f["spearman"],
                "Description": f["description"]
            }
            for f in acquired
        ])
        
        df_sorted = sort_by_checkpoint(df_feat, col="Acquired At")
        df_sorted.to_csv(f"{trend_dir}/acquired_features.csv", index=False)

    # =======================================================================
    # 3. NOT ACQUIRED – Topic Summary
    # =======================================================================
    if not_acq:
        topic_groups = defaultdict(list)
        for f in not_acq:
            topic_groups[f["topic"]].append(f)

        rows = []
        for topic, feats in topic_groups.items():
            max_probs = [max([p for p in f["median_probs"] if p is not None], default=0) for f in feats]
            rows.append({
                "Topic": topic,
                "Num Features": len(feats),
                "Avg Max Prob Reached": np.mean(max_probs)
            })
        df_na = pd.DataFrame(rows)
        df_na.to_csv(f"{trend_dir}/not_acquired_topic_summary.csv", index=False)

        # Detailed
        detailed = []
        for f in not_acq:
            median = f["median_probs"]
            maxp = max([p for p in median if p is not None], default=0)
            detailed.append({
                "Feature ID": f["feature_id"],
                "Topic": f["topic"],
                "Max Prob": maxp,
                "Gap to 0.5": 0.5 - maxp,
                "Description": f["description"]
            })
        df_det = pd.DataFrame(detailed).sort_values("Max Prob", ascending=False)
        df_det.to_csv(f"{trend_dir}/not_acquired_features.csv", index=False)

    # =======================================================================
    # 4. TREND-SPECIFIC ANALYSIS
    # =======================================================================

    # -------- increase-stagnate --------
    if trend == "increase-stagnate":
        stag = [
            f for f in acquired
            if f["stagnation_checkpoint"] >= 0
        ]
        if stag:
            df_stag = pd.DataFrame([
                {
                    "Feature ID": f["feature_id"],
                    "Topic": f["topic"],
                    "Acquired At": model_names[f["acq_checkpoint"]],
                    "Stagnates At": model_names[f["stagnation_checkpoint"]],
                    "Description": f["description"]
                }
                for f in stag
            ])
            df_stag = sort_by_checkpoint(df_stag, col="Stagnates At")
            df_stag.to_csv(f"{trend_dir}/stagnation_analysis.csv", index=False)

    # -------- increase-decrease --------
    if trend == "increase-decrease":
        # Peak
        peaks = [f for f in acquired if f["peak_checkpoint"] >= 0]
        if peaks:
            df_peaks = pd.DataFrame([
                {
                    "Feature ID": f["feature_id"],
                    "Topic": f["topic"],
                    "Acquired At": model_names[f["acq_checkpoint"]],
                    "Peaks At": model_names[f["peak_checkpoint"]],
                    "Description": f["description"]
                }
                for f in peaks
            ])
            df_peaks = sort_by_checkpoint(df_peaks, col="Peaks At")
            df_peaks.to_csv(f"{trend_dir}/peak_analysis.csv", index=False)

        # Decrease after peak
        decrows = []
        for f in peaks:
            m = f["median_probs"]
            peak = f["peak_checkpoint"]
            dc = -1
            for i in range(peak + 1, len(m)):
                if m[i] is not None and m[peak] is not None and m[i] < m[peak]:
                    dc = i
                    break
            if dc >= 0:
                decrows.append({
                    "Feature ID": f["feature_id"],
                    "Topic": f["topic"],
                    "Peaks At": model_names[peak],
                    "Decreases At": model_names[dc],
                    "Description": f["description"]
                })

        if decrows:
            df_dec = pd.DataFrame(decrows)
            df_dec.to_csv(f"{trend_dir}/decrease_analysis.csv", index=False)

print("Done! Files saved to:", output_root)
