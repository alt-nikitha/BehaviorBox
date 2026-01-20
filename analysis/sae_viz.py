import json
import numpy as np
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
PRECOMPUTED_DATA_DIR = "./precomputed_data"
OUTPUT_FILE = "feature_trends.html"
FEATURES_PER_PAGE = 999999  # no pagination for HTML

# FILTERS — edit these as needed
METRIC_CHOICE = "Median Probs"     # "Median Probs" | "Ranks" | "Avg Probs"
TREND = "increase"                 # "increase" | "decrease" | "flat"
MAX_ERROR = 999
MIN_RANGE = 0.0
GOODBAD_FILTER = "none"            # "none" | "good>bad" | "bad>good" | "balanced"

SORT_BY = "spearman_desc"          # "spearman_desc" | "spearman_asc" | "pearson_desc" | "pearson_asc" | "jump"

# ============================================================
# Load precomputed data
# ============================================================
def load_precomputed_data():
    datasets = {}
    for filename in os.listdir(PRECOMPUTED_DATA_DIR):
        if filename.endswith(".json"):
            name = filename.replace(".json", "")
            with open(os.path.join(PRECOMPUTED_DATA_DIR, filename), "r") as f:
                datasets[name] = json.load(f)
    return datasets

# ============================================================
# Filtering
# ============================================================
def filter_features(features, metric_choice, trend, max_error, min_range, goodbad_filter):
    metric_map = {
        "Median Probs": ("median_probs", "trend_median_probs", "range_median_probs"),
        "Ranks": ("avg_ranks", "trend_ranks", "range_ranks"),
        "Avg Probs": ("avg_probs", "trend_avg_probs", "range_avg_probs")
    }

    _, trend_key, range_key = metric_map[metric_choice]

    out = []
    for f in features:
        if f[trend_key] != trend:
            continue
        if f["max_error"] > max_error:
            continue
        if f[range_key] < min_range:
            continue

        good = f["good_count"]
        bad = f["bad_count"]

        if goodbad_filter == "good>bad" and not (good > bad):
            continue
        if goodbad_filter == "bad>good" and not (bad > good):
            continue
        if goodbad_filter == "balanced":
            total = good + bad
            if total > 0:
                g = good / total
                if not (0.4 <= g <= 0.6):
                    continue
        out.append(f)
    return out

# ============================================================
# Sorting
# ============================================================
def sort_features(features, sort_by, metric_choice):
    metric_map = {
        "Median Probs": ("median_probs", "jump_median_probs_idx", "jump_median_probs_value"),
        "Ranks": ("avg_ranks", "jump_ranks_idx", "jump_ranks_value"),
        "Avg Probs": ("avg_probs", "jump_avg_probs_idx", "jump_avg_probs_value")
    }
    _, jump_idx, jump_val = metric_map[metric_choice]

    if sort_by == "spearman_desc":
        return sorted(features, key=lambda f: f.get("spearman_corr", -999), reverse=True)
    if sort_by == "spearman_asc":
        return sorted(features, key=lambda f: f.get("spearman_corr", 999))
    if sort_by == "pearson_desc":
        return sorted(features, key=lambda f: f.get("pearson_corr", -999), reverse=True)
    if sort_by == "pearson_asc":
        return sorted(features, key=lambda f: f.get("pearson_corr", 999))
    if sort_by == "jump":
        return sorted(features, key=lambda f: (f[jump_idx], f[jump_val]))
    return features

# ============================================================
# Plotting utils (same logic as your Dash version)
# ============================================================
def create_distribution_charts(subset_counts, subset_good_counts, subset_bad_counts,
                               good_count, bad_count, total_samples):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2)

    # Percentages
    good_pct = good_count / total_samples * 100 if total_samples else 0
    bad_pct = bad_count / total_samples * 100 if total_samples else 0

    # Subset stacked chart
    subset_names = list(subset_counts.keys())
    good_vals = [(subset_good_counts.get(s, 0) / total_samples * 100) for s in subset_names]
    bad_vals = [(subset_bad_counts.get(s, 0) / total_samples * 100) for s in subset_names]

    fig.add_bar(row=1, col=1, x=subset_names, y=good_vals, name="Good", marker_color="green")
    fig.add_bar(row=1, col=1, x=subset_names, y=bad_vals, name="Bad", marker_color="red")

    fig.update_layout(barmode="stack")

    # Overall good/bad
    fig.add_bar(row=2, col=1, x=["good", "bad"], y=[good_pct, bad_pct],
                marker_color=["green", "red"])

    fig.update_layout(height=350, width=400, margin=dict(l=20, r=20, t=20, b=20))

    return fig

# ============================================================
# Feature card exporter
# ============================================================
def create_feature_html(feature, model_names, metric_choice):
    metric_map = {
        "Median Probs": ("median_probs", "Median Probability"),
        "Ranks": ("avg_ranks", "Average Rank"),
        "Avg Probs": ("avg_probs", "Average Probability"),
    }

    values_key, y_label = metric_map[metric_choice]

    # ---------------------- Main Plot ----------------------
    values = np.array(feature[values_key])
    le = np.array(feature["lower_errors"])
    he = np.array(feature["higher_errors"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_scatter(
        x=model_names,
        y=values,
        mode="lines+markers",
        name=y_label
    )

    # ---------------------- Distribution Plot ----------------------
    dist_fig = create_distribution_charts(
        feature["subset_counts"],
        feature.get("subset_good_counts", {}),
        feature.get("subset_bad_counts", {}),
        feature["good_count"],
        feature["bad_count"],
        len(feature["samples"])
    )

    # ---------------------- Samples Table ----------------------
    table_html = "<table border='1' style='border-collapse: collapse; width: 100%; font-size: 12px;'>"
    table_html += """
    <tr style="background:#eee;">
        <th>Activation</th><th>Cos Sim</th><th>Subset</th><th>Good/Bad</th>
        <th>Before</th><th>Word</th><th>After</th>
    </tr>
    """
    for s in feature["samples"]:
        color = "#d4edda" if s["good_bad"] == "good" else "#f8d7da"
        table_html += f"""
        <tr>
            <td>{s['activation']:.4f}</td>
            <td>{s['cos_sim']:.4f}</td>
            <td>{s['subset']}</td>
            <td style="background:{color};font-weight:bold;">{s['good_bad']}</td>
            <td>{s['before']}</td>
            <td><b>{s['word']}</b></td>
            <td>{s['after']}</td>
        </tr>
        """
    table_html += "</table>"

    # ---------------------- Combine into HTML ----------------------
    card_html = f"""
    <div style="padding:20px; margin-bottom:40px; border-bottom:2px solid #ddd;">
        <h2>Feature {feature['feature_id']} — {feature['description']}</h2>

        <h3>{y_label} Trend</h3>
        {fig.to_html(full_html=False, include_plotlyjs=False)}

        <h3>Distribution</h3>
        {dist_fig.to_html(full_html=False, include_plotlyjs=False)}

        <h3>Samples</h3>
        {table_html}
    </div>
    """

    return card_html

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    datasets = load_precomputed_data()

    if not datasets:
        print("No datasets found!")
        exit()

    # Use FIRST dataset by default
    dataset_name = list(datasets.keys())[0]
    dataset = datasets[dataset_name]

    features = dataset["features"]
    model_names = dataset["model_names"]

    # Apply filtering + sorting
    filtered = filter_features(features, METRIC_CHOICE, TREND, MAX_ERROR, MIN_RANGE, GOODBAD_FILTER)
    sorted_feats = sort_features(filtered, SORT_BY, METRIC_CHOICE)

    # Build full HTML
    html = """
    <html>
    <head>
        <title>Feature Trends Export</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style="font-family:Arial; padding:40px;">
    <h1>Feature Trends Export</h1>
    """

    for feat in sorted_feats:
        html += create_feature_html(feat, model_names, METRIC_CHOICE)

    html += "</body></html>"

    with open(OUTPUT_FILE, "w") as f:
        f.write(html)

    print(f"✔ Exported HTML to {OUTPUT_FILE}")
