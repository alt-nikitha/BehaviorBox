import json, re
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os
# -----------------------------
# Load + parse
# -----------------------------

folder = "/home/nsrikant/BehaviorBoxNew/sae_outputs/n_comparison/_seed=42_ofw=_N=3000_k=50_lp=None"

with open(os.path.join(folder, "feature_labels_validated/neulab-claude-sonnet-4-20250514.json"), "r") as f:
    raw = json.load(f)
model_names = [
    "pythia-160m-step1000",
    "pythia-160m-step10000",
    "pythia-160m-step70000",
    "pythia-160m-step100000",
    "pythia-160m",
]

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

records = []
for fid, feat in raw.items():
    avg_ranks = parse_array_field(feat.get("Mean Ranks", "[]"))
    avg_probs = parse_array_field(feat.get("Avg Probs", "[]"))
    records.append({
        "Feature": fid,
        "Description": feat.get("Description", ""),
        "Winning Rank": feat.get("Winning Rank"),
        "Model": feat.get("Model"),
        "Avg Ranks": avg_ranks,
        "Avg Probs": avg_probs,
    })

df = pd.DataFrame(records)

# Trend classifier (improved)
# def classify_trend(values):
#     vals = np.asarray(values, dtype=float)
#     n = len(vals)
#     if n < 2: return "stagnate"
    
#     # Normalize to 0-1 range for easier comparison
#     vals_min, vals_max = vals.min(), vals.max()
#     rng = vals_max - vals_min
#     if rng < 1e-8: return "stagnate"
#     normalized = (vals - vals_min) / rng
    
#     # Check for peak or trough patterns first (these should take priority)
#     peak_idx = np.argmax(normalized)
#     trough_idx = np.argmin(normalized)
    
#     # Increase-decrease: peak in middle, significant drop after
#     if 1 < peak_idx < n-1:
#         rise = normalized[peak_idx] - normalized[0]
#         fall = normalized[peak_idx] - normalized[-1]
#         if rise > 0.3 and fall > 0.3:
#             return "increase-decrease"
    
#     # Decrease-increase: trough in middle, significant rise after
#     if 1 < trough_idx < n-1:
#         fall = normalized[0] - normalized[trough_idx]
#         rise = normalized[-1] - normalized[trough_idx]
#         if fall > 0.3 and rise > 0.3:
#             return "decrease-increase"
    
#     # Calculate overall direction
#     diffs = np.diff(normalized)
#     thr = 0.1  # threshold for significant change
    
#     pos_changes = diffs > thr
#     neg_changes = diffs < -thr
    
#     # Strictly increasing: most changes positive, end higher than start
#     if np.mean(pos_changes) >= 0.75 and normalized[-1] > normalized[0] + 0.3:
#         return "increasing"
    
#     # Strictly decreasing: most changes negative, end lower than start
#     if np.mean(neg_changes) >= 0.75 and normalized[-1] < normalized[0] - 0.3:
#         return "decreasing"
    
#     # Check for increase then stagnate
#     mid = n // 2
#     first_half = normalized[:mid+1]
#     second_half = normalized[mid:]
    
#     first_trend = np.mean(np.diff(first_half) > thr)
#     second_stable = np.std(second_half) < 0.1
    
#     if first_trend >= 0.6 and second_stable and normalized[-1] > normalized[0] + 0.2:
#         return "increase-stagnate"
    
#     # Check for decrease then stagnate
#     first_trend = np.mean(np.diff(first_half) < -thr)
#     if first_trend >= 0.6 and second_stable and normalized[-1] < normalized[0] - 0.2:
#         return "decrease-stagnate"
    
#     return "other"
def classify_trend(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 2: return "stagnate"
    
    # Normalize to 0-1 range for easier comparison
    vals_min, vals_max = vals.min(), vals.max()
    rng = vals_max - vals_min
    if rng < 1e-8: return "stagnate"
    normalized = (vals - vals_min) / rng
    
    # Check for peak or trough patterns first (these should take priority)
    peak_idx = np.argmax(normalized)
    trough_idx = np.argmin(normalized)
    
    # Increase-decrease: peak in middle, significant drop after
    if 1 < peak_idx < n-1:
        rise = normalized[peak_idx] - normalized[0]
        fall = normalized[peak_idx] - normalized[-1]
        if rise > 0.25 and fall > 0.25:
            return "increase-decrease"
    
    # Decrease-increase: trough in middle, significant rise after
    if 1 < trough_idx < n-1:
        fall = normalized[0] - normalized[trough_idx]
        rise = normalized[-1] - normalized[trough_idx]
        if fall > 0.25 and rise > 0.25:
            return "decrease-increase"
    
    # Calculate overall direction
    overall_change = normalized[-1] - normalized[0]
    diffs = np.diff(normalized)
    
    # Check for stagnation first (very small overall change)
    if abs(overall_change) < 0.15 and np.std(normalized) < 0.15:
        return "stagnate"
    
    # Strictly increasing: net positive change and mostly upward movement
    pos_changes = np.sum(diffs > 0.05)
    neg_changes = np.sum(diffs < -0.05)
    
    if overall_change > 0.2 and pos_changes >= neg_changes:
        # Check if it stagnates at the end
        mid = n // 2
        second_half = normalized[mid:]
        if len(second_half) > 1 and np.std(second_half) < 0.1 and normalized[mid] > normalized[0] + 0.15:
            return "increase-stagnate"
        return "increasing"
    
    # Strictly decreasing: net negative change and mostly downward movement
    if overall_change < -0.2 and neg_changes >= pos_changes:
        # Check if it stagnates at the end
        mid = n // 2
        second_half = normalized[mid:]
        if len(second_half) > 1 and np.std(second_half) < 0.1 and normalized[mid] < normalized[0] - 0.15:
            return "decrease-stagnate"
        return "decreasing"
    
    # More relaxed stagnate check for mild variations
    if abs(overall_change) < 0.25 and np.std(diffs) < 0.15:
        return "stagnate"
    
    return "other"

df["Trend (Ranks)"] = df["Avg Ranks"].apply(classify_trend)
df["Trend (Probs)"] = df["Avg Probs"].apply(classify_trend)

# -----------------------------
# Load CSV of activations
# -----------------------------
act_csv_path = os.path.join(folder, "top-50_activations.csv")  # <-- replace
act_df = pd.read_csv(act_csv_path)

# Load JSON mapping word_id -> before, word, after
word_json_path = os.path.join(folder, "top-50_words_in_context.json")  # <-- replace
with open(word_json_path, "r") as f:
    word_samples = json.load(f)

# -----------------------------
# Dash app
# -----------------------------
app = Dash(__name__)
app.title = "Feature Trend + Activations"

app.layout = html.Div([
    html.H2("Feature Trends Across Models", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Metric to Base Trend On:"),
        dcc.Dropdown(
            id="metric-dropdown",
            options=[
                {"label":"Average Probability","value":"Probs"},
                {"label":"Average Rank","value":"Ranks"}
            ],
            value="Probs",
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

    html.Div(id="feature-container")
])

# -----------------------------
# Callback: update trend dropdown options based on metric
# -----------------------------
@app.callback(
    Output("trend-dropdown", "options"),
    Input("metric-dropdown", "value")
)
def update_trend_options(metric_choice):
    trend_col = f"Trend ({metric_choice})"
    unique_trends = sorted(df[trend_col].unique())
    return [{"label": t, "value": t} for t in unique_trends]

# -----------------------------
# Callback: plot + table
# -----------------------------
@app.callback(
    Output("feature-container", "children"),
    Input("trend-dropdown", "value"),
    Input("metric-dropdown", "value")
)
def update_features(selected_trend, metric_choice):
    trend_col = f"Trend ({metric_choice})"
    metric_label = "Avg Probs" if metric_choice=="Probs" else "Avg Ranks"
    y_label = "Average Probability" if metric_choice=="Probs" else "Average Rank"

    sub = df[df[trend_col]==selected_trend]
    if sub.empty:
        return html.Div(f"No features found for '{selected_trend}' trend (based on {metric_label}).", style={"padding":"20px"})

    children = []

    for _, row in sub.iterrows():
        values = row[metric_label]
        if len(values)==0: continue

        # --- Plot ---
        plot_df = pd.DataFrame({
            "Model": model_names[:len(values)],
            y_label: values[:len(model_names)]
        })
        fig = px.line(plot_df, x="Model", y=y_label, markers=True, title=f"Feature {row['Feature']} ({y_label})")
        fig.update_traces(line=dict(width=2), opacity=0.8)
        fig.update_layout(title_x=0.4)

        # --- Table: activations + word samples ---
        # FIX: Convert feature column to string for comparison
        feat_acts = act_df[act_df["feature"].astype(str) == str(row["Feature"])]
        feat_acts = feat_acts.sort_values("act_value", ascending=False)
        table_rows = []
        for _, r in feat_acts.iterrows():
            # FIX: Ensure word_id is string
            wid = str(r["word_id"])
            word_info = word_samples.get(wid, {"before":"","word":"","after":""})

            table_rows.append(html.Tr([
                html.Td(f"{r['act_value']:.4f}", style={"border":"1px solid #ddd", "padding":"8px"}),
                html.Td(word_info.get("before",""), style={"border":"1px solid #ddd", "padding":"8px"}),
                html.Td(word_info.get("word",""), style={"border":"1px solid #ddd", "padding":"8px", "fontWeight":"bold"}),
                html.Td(word_info.get("after",""), style={"border":"1px solid #ddd", "padding":"8px"})
            ]))
        table = html.Table(
            [html.Tr([
                html.Th("Samples", colSpan="4", style={"border":"1px solid #ddd", "padding":"12px", "backgroundColor":"#e8e8e8", "textAlign":"center", "fontSize":"18px", "fontWeight":"bold"})
            ]),
            html.Tr([
                html.Th("Activation", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("Before", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("Word", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"}),
                html.Th("After", style={"border":"1px solid #ddd", "padding":"10px", "backgroundColor":"#f2f2f2"})
            ])] + table_rows,
            style={"border":"1px solid #ddd", "borderCollapse":"collapse", "width":"100%"}
        )

        # --- Combine plot + table side by side ---
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
    app.run(host='0.0.0.0',debug=True)