

import json, re
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os

# -----------------------------
# Dataset folder configuration
# -----------------------------
DATASET_FOLDERS = {
    "Olmo-BLIMP Full default": "/home/nsrikant/bbox_outputs/sae_outputs_blimp/n_moreearly_blimp_full_olmo_seed=42_ofw=0.7_N=3000_k=50_lp=None"
}

model_names = [
        "OLMo2-step0-tokens0B",
        "OLMo2-step1000-tokens3B",
        "OLMo2-step2000-tokens5B",
        "OLMo2-step3000-tokens7B",
        "OLMo2-step4000-tokens9B",
        "OLMo2-step5000-tokens11B",
        "OLMo2-step6000-tokens13B",
        "OLMo2-step7000-tokens15B",
        "OLMo2-step8000-tokens17B",
        "OLMo2-step9000-tokens19B",
        "OLMo2-step10000-tokens21B"
    ]

# -----------------------------
# Helper functions
# -----------------------------
def max_jump(values):
    """
    Returns tuple (jump_value, jump_index)
    jump_index = index BEFORE the jump
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return (0, -1)
    diffs = np.diff(vals)
    jump_idx = np.argmax(diffs)
    jump_val = diffs[jump_idx]
    return (jump_val, jump_idx)

def parse_array_field(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=float)
    s = str(x).strip().strip('"')
    try:
        return np.array(json.loads(s), dtype=float)
    except Exception:
        pass
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return np.array([float(n) for n in nums], dtype=float)

def classify_trend(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 2:
        return "other"

    # Normalize to 0–1 range
    vals_min, vals_max = vals.min(), vals.max()
    rng = vals_max - vals_min
    if rng < 1e-8:
        return "other"
    normalized = (vals - vals_min) / rng

    # Check for peak/trough patterns
    peak_idx = np.argmax(normalized)
    trough_idx = np.argmin(normalized)
    
    # Increase-decrease: peak in middle
    if 1 < peak_idx < n - 1:
        rise = normalized[peak_idx] - normalized[0]
        fall = normalized[peak_idx] - normalized[-1]
        if rise > 0.25 and fall > 0.25:
            return "increase-decrease"
    
    # Decrease-increase: trough in middle
    if 1 < trough_idx < n - 1:
        fall = normalized[0] - normalized[trough_idx]
        rise = normalized[-1] - normalized[trough_idx]
        if fall > 0.25 and rise > 0.25:
            return "decrease-increase"
    
    # Compute general change metrics
    overall_change = normalized[-1] - normalized[0]
    diffs = np.diff(normalized)
    pos_changes = np.sum(diffs > 0.05)
    neg_changes = np.sum(diffs < -0.05)

    # Increasing trends
    if overall_change > 0.2 and pos_changes >= neg_changes:
        return "increasing"

    # Decreasing trends
    if overall_change < -0.2 and neg_changes >= pos_changes:
        return "decreasing"

    return "other"

def load_data(folder):
    """Load and process data from the specified folder"""
    # Load feature labels
    with open(os.path.join(folder, "feature_labels_validated/neulab-claude-sonnet-4-20250514.json"), "r") as f:
        raw = json.load(f)
    
    records = []
    for fid, feat in raw.items():
        avg_ranks = parse_array_field(feat.get("Mean Ranks", "[]"))
        avg_probs = parse_array_field(feat.get("Avg Probs", "[]"))
        median_probs = parse_array_field(feat.get("Median Probs", "[]"))
        records.append({
            "Feature": fid,
            "Description": feat.get("Description", ""),
            "Winning Rank": feat.get("Winning Rank"),
            "Model": feat.get("Model"),
            "Avg Ranks": avg_ranks,
            "Avg Probs": avg_probs,
            "Median Probs": median_probs
        })
    
    df = pd.DataFrame(records)
    
    # Classify trends
    df["Trend (Ranks)"] = df["Avg Ranks"].apply(classify_trend)
    df["Trend (Avg Probs)"] = df["Avg Probs"].apply(classify_trend)
    df["Trend (Median Probs)"] = df["Median Probs"].apply(classify_trend)
    
    # Load activations
    act_csv_path = os.path.join(folder, "top-50_activations.csv")
    act_df = pd.read_csv(act_csv_path)
    
    # Load embeddings
    emb_df_path = os.path.join(folder, "feature_sample_centroid-metrics.csv")
    emb_df = pd.read_csv(emb_df_path)
    
    # Map cosine similarities
    feature_to_cos = (
        emb_df[["feature", "sample_centroid_cos_sim"]]
        .assign(feature=lambda x: x["feature"].astype(str))
        .set_index("feature")["sample_centroid_cos_sim"]
        .to_dict()
    )
    df["sample_centroid_cos_sim"] = df["Feature"].astype(str).map(feature_to_cos)
    
    # Load word samples
    word_json_path = os.path.join(folder, "top-50_words_in_context.json")
    with open(word_json_path, "r") as f:
        word_samples = json.load(f)
    
    return df, act_df, emb_df, word_samples

# -----------------------------
# Dash app
# -----------------------------
app = Dash(__name__)
app.title = "Feature Trend + Activations"

app.layout = html.Div([
    html.H2("Feature Trends Across Models", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id="dataset-dropdown",
            options=[{"label": name, "value": name} for name in DATASET_FOLDERS.keys()],
            value=list(DATASET_FOLDERS.keys())[0],
            clearable=False,
            style={"width":"40%", "marginRight":"20px"}
        ),

        html.Label("Select Metric to Base Trend On:"),
        dcc.Dropdown(
            id="metric-dropdown",
            options=[
                {"label":"Median Probability","value":"Median Probs"},
                {"label":"Average Rank","value":"Ranks"},
                {"label":"Average Probability","value":"Avg Probs"},
            ],
            value="Median Probs",
            clearable=False,
            style={"width":"40%", "marginRight":"20px"}
        ),

        html.Label("Select Trend Type:"),
        dcc.Dropdown(
            id="trend-dropdown",
            value="increasing",
            clearable=False,
            style={"width":"40%"}
        ),
    ], style={"display":"flex", "justifyContent":"center", "gap":"30px", "margin":"20px"}),

    html.Div([
        html.Label("Max Allowed Error Across Models:"),
        dcc.Slider(
            id="error-slider",
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.25,
            marks={0: '0', 0.25: '0.25', 0.5:'0.5', 0.75:'0.75', 1:'1'},
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='mouseup'
        )
    ], style={"width":"50%", "margin":"20px auto"}),

    # html.Div(id="feature-container"),
    dcc.Loading(
        id="loading",
        type="default",  # Options: "graph", "cube", "circle", "dot", or "default"
        children=html.Div(id="feature-container"),
        color="#119DFF",
        fullscreen=False,
        style={"marginTop": "20px"}
    )
])

# -----------------------------
# Callback: update trend dropdown options based on metric and dataset
# -----------------------------
@app.callback(
    Output("trend-dropdown", "options"),
    Input("metric-dropdown", "value"),
    Input("dataset-dropdown", "value")
)
def update_trend_options(metric_choice, dataset_name):
    folder = DATASET_FOLDERS[dataset_name]
    df, _, _, _ = load_data(folder)
    trend_col = f"Trend ({metric_choice})"
    unique_trends = sorted(df[trend_col].unique())
    return [{"label": t, "value": t} for t in unique_trends]

# -----------------------------
# Callback: plot + table
# -----------------------------
@app.callback(
    Output("feature-container", "children"),
    Input("trend-dropdown", "value"),
    Input("metric-dropdown", "value"),
    Input("error-slider", "value"),
    Input("dataset-dropdown", "value")
)
def update_features(selected_trend, metric_choice, max_allowed_error, dataset_name):
    folder = DATASET_FOLDERS[dataset_name]
    df, act_df, emb_df, word_samples = load_data(folder)
    
    trend_col = f"Trend ({metric_choice})"
    
    metric_label = metric_choice
    if metric_choice == "Median Probs":
        y_label = "Median Probability"
    elif metric_choice == "Ranks":
        y_label = "Average Rank"
    else:
        y_label = "Average Probability"

    df[['max_jump', 'jump_idx']] = df[metric_label].apply(lambda x: pd.Series(max_jump(x)))
    df_sorted = df.sort_values(['jump_idx', 'max_jump'], ascending=[True, False])

    sub = df_sorted[df_sorted[trend_col]==selected_trend]

    children = []
    
    for _, row in sub.iterrows():
        feat_acts_all = act_df[act_df["feature"].astype(str) == str(row["Feature"])].reset_index(drop=True)
        higher_errors = []
        lower_errors = []
        for model in model_names:
            logprobs = feat_acts_all[model].values
            probs = np.exp(logprobs)
            min_prob = np.min(probs)
            max_prob = np.max(probs)
            lower_errors.append(min_prob)
            higher_errors.append(max_prob)
        max_error = np.max(np.array(higher_errors) - np.array(lower_errors))
        if max_error > max_allowed_error:
            continue
        
        values = row[metric_label]
        if len(values)==0: 
            continue

        # Plot
        plot_df = pd.DataFrame({
            "Model": model_names[:len(values)],
            y_label: values[:len(model_names)],
            "upper_error": lower_errors,
            "lower_error": higher_errors
        })
        min_prob = np.min(values[:len(model_names)])
        max_prob = np.max(values[:len(model_names)])
        probs_range = max_prob - min_prob
        
        fig = px.line(plot_df, x="Model", y=y_label, markers=True, 
                     title=f"Feature {row['Feature']} ({y_label}); Range: {probs_range}")
        
        fig.update_traces(
            error_y=dict(
                type='data',
                array=plot_df['upper_error'] - plot_df[y_label],
                arrayminus=plot_df[y_label] - plot_df['lower_error']
            )
        )
        fig.update_traces(line=dict(width=2), opacity=0.8)
        fig.update_layout(title_x=0.4)

        # Table: activations + word samples
        feat_emb = emb_df[emb_df["feature"].astype(str) == str(row["Feature"])]["sample_centroid_cos_sim"].values

        combined = feat_acts_all.copy()
        if len(feat_emb) < len(combined):
            cos_series = pd.Series(feat_emb, index=combined.index[:len(feat_emb)])
        else:
            cos_series = pd.Series(feat_emb[:len(combined)], index=combined.index)
        combined["cos_sim"] = cos_series
        combined = combined.sort_values("cos_sim", ascending=False, na_position="last")

        table_rows = []
        for _, r in combined.iterrows():
            wid = str(r.get("word_id", ""))
            word_info = word_samples.get(wid, {"before":"","word":"","after":""})
            cos_val = r.get("cos_sim", np.nan)

            table_rows.append(html.Tr([
                html.Td("" if pd.isna(r.get('act_value')) else f"{float(r['act_value']):.4f}", 
                       style={"border":"1px solid #ddd", "padding":"8px"}),
                html.Td("" if pd.isna(cos_val) else f"{float(cos_val):.4f}", 
                       style={"border":"1px solid #ddd", "padding":"8px"}),
                html.Td(word_info.get("before",""), 
                       style={"border":"1px solid #ddd", "padding":"8px"}),
                html.Td(str(word_info.get("word",""))
                        .replace("\t", "<tab>")
                        .replace(" ", "<space>"),
                        style={"border":"1px solid #ddd", "padding":"8px", "fontWeight":"bold", "fontFamily":"monospace"}),
                html.Td(word_info.get("after",""), 
                       style={"border":"1px solid #ddd", "padding":"8px"})
            ]))
        
        table = html.Table(
            [html.Tr([
                html.Th("Samples", colSpan="5", 
                       style={"border":"1px solid #ddd", "padding":"12px", "backgroundColor":"#e8e8e8", 
                              "textAlign":"center", "fontSize":"18px", "fontWeight":"bold"})
            ]),
            html.Tr([
                html.Th("Activation", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("Cos Sim", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("Before", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("Word", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("After", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"})
            ])] + table_rows,
            style={"border":"1px solid #ddd", "borderCollapse":"collapse", "width":"100%"}
        )

        # Combine plot + table side by side
        children.append(html.Div([
            html.H3([
                html.Span(f"Feature {row['Feature']}: ", style={"fontSize":"28px", "fontWeight":"bold"}),
                html.Span(row["Description"], style={"fontSize":"28px", "fontWeight":"normal"})
            ], style={"marginBottom":"20px", "lineHeight":"1.5"}),
            html.Div([
                html.Div(dcc.Graph(figure=fig, style={"height":"400px", "flex":"1"}), style={"flex":"1"}),
                html.Div(table, style={"flex":"1", "overflowY":"scroll", "maxHeight":"400px", "paddingLeft":"20px"})
            ], style={"display":"flex", "gap":"20px"})
        ], style={"marginBottom":"50px"}))

    return children

# -----------------------------
if __name__=="__main__":
    app.run(host='0.0.0.0', port=8081, debug=True)