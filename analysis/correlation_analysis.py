import json, re
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy import stats
from tqdm import tqdm
# -----------------------------
# Dataset folder configuration
# -----------------------------
DATASET_FOLDER = "/home/nsrikant/bbox_outputs/sae_outputs_blimp_trained_on_pile/blimp_trained_on_pile"
correlation_folder = "/home/nsrikant/BehaviorBoxNew/results/blimp_full"

model_names = [
    "pythia-160m-step1",
    "pythia-160m-step2",
    "pythia-160m-step4",
    "pythia-160m-step8",
    "pythia-160m-step16",
    "pythia-160m-step32",
    "pythia-160m-step64",
    "pythia-160m-step128",
    "pythia-160m-step256",
    "pythia-160m-step512",
    "pythia-160m-step1000",
    "pythia-160m-step10000",
    "pythia-160m-step70000",
    "pythia-160m-step100000",
    "pythia-160m"
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


def create_distribution_charts(subset_distributions, good_bad_distributions):
    """Create plotly charts for subset and good/bad distributions"""
    
    # Create DataFrame for easier manipulation
    df_dist = pd.DataFrame({
        'subset': subset_distributions,
        'category': good_bad_distributions
    })
    
    # Calculate subset distribution with good/bad breakdown
    subset_counts = df_dist.groupby(['subset', 'category']).size().unstack(fill_value=0)
    
    # Get total counts per subset
    subset_totals = df_dist['subset'].value_counts()
    total_samples = len(subset_distributions)
    
    # Calculate percentages for good and bad within each subset
    good_percentages = {}
    bad_percentages = {}
    
    for subset in subset_totals.index:
        total_for_subset = subset_totals[subset]
        good_count = subset_counts.loc[subset, 'good'] if 'good' in subset_counts.columns and subset in subset_counts.index else 0
        bad_count = subset_counts.loc[subset, 'bad'] if 'bad' in subset_counts.columns and subset in subset_counts.index else 0
        
        good_percentages[subset] = (good_count / total_samples * 100)
        bad_percentages[subset] = (bad_count / total_samples * 100)
    
    # Sort by total percentage
    sorted_subsets = sorted(subset_totals.index, key=lambda x: subset_totals[x], reverse=True)
    
    # Calculate overall good/bad distribution percentages
    goodbad_counts = pd.Series(good_bad_distributions).value_counts()
    goodbad_percentages = (goodbad_counts / len(good_bad_distributions) * 100).round(1)
    
    # Create stacked vertical layout
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Subset Distribution (Good/Bad)', 'Overall Good/Bad'),
        vertical_spacing=0.15,
        specs=[[{"type": "bar"}], [{"type": "bar"}]]
    )
    
    # Add stacked bar chart for subset distribution
    # Good portion (green)
    fig.add_trace(
        go.Bar(
            name='Good',
            x=sorted_subsets,
            y=[good_percentages[s] for s in sorted_subsets],
            marker_color='#28a745',
            text=[f'{good_percentages[s]:.0f}%' if good_percentages[s] > 3 else '' for s in sorted_subsets],
            textposition='inside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Bad portion (red)
    fig.add_trace(
        go.Bar(
            name='Bad',
            x=sorted_subsets,
            y=[bad_percentages[s] for s in sorted_subsets],
            marker_color='#dc3545',
            text=[f'{bad_percentages[s]:.0f}%' if bad_percentages[s] > 3 else '' for s in sorted_subsets],
            textposition='inside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add good/bad distribution bar chart
    colors = ['#28a745' if 'good' in str(x).lower() else '#dc3545' for x in goodbad_percentages.index]
    fig.add_trace(
        go.Bar(
            x=goodbad_percentages.index.tolist(),
            y=goodbad_percentages.values.tolist(),
            marker_color=colors,
            text=[f'{v:.0f}%' for v in goodbad_percentages.values],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Update layout for stacked bars
    fig.update_layout(
        barmode='stack',
        height=300,
        width=350,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=40),
        font=dict(size=10)
    )
    
    # Update axes
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=10), row=2, col=1)
    
    max_subset_pct = max([good_percentages[s] + bad_percentages[s] for s in sorted_subsets])
    fig.update_yaxes(range=[0, max_subset_pct * 1.15], showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[0, max(goodbad_percentages.values) * 1.2], showticklabels=False, row=2, col=1)
    
    return fig.to_html(include_plotlyjs=False, div_id=None, config={'displayModeBar': False})


def get_correlated_trends(max_allowed_error):
    

    df, act_df, emb_df, word_samples = load_data(DATASET_FOLDER)
    
    
    
    
    # if metric_choice == "Median Probs":
    #     y_label = "Median Probability"
    # elif metric_choice == "Ranks":
    #     y_label = "Average Rank"
    # else:
    #     y_label = "Average Probability"
    metric_label = "Median Probs"
    y_label = "Median Probability"
    trend_col = f"Trend ({metric_label})"

    df[['max_jump', 'jump_idx']] = df[metric_label].apply(lambda x: pd.Series(max_jump(x)))
    df_sorted = df.sort_values(['jump_idx', 'max_jump'], ascending=[True, False])

    

    results = []
    
    for _, row in tqdm(df_sorted.iterrows()):
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

        correlation_values = []

        for model_name in model_names[:len(values)]:
            with open(f"{correlation_folder}/{model_name}/blimp_summary_seq_logprob.json", "r") as fp:
                c_json = json.load(fp)
            correlation_values.append(c_json["overall_accuracy"])
            

        
        
        corr, p = stats.pearsonr(values[:len(model_names)], correlation_values)
        corr_spear, p_new = stats.spearmanr(values[:len(model_names)], correlation_values)
        desc = row['Description']
        trend = row[trend_col]
        
    
    
        # Table: activations + word samples
        feat_emb = emb_df[emb_df["feature"].astype(str) == str(row["Feature"])]["sample_centroid_cos_sim"].values

        combined = feat_acts_all.copy()
        if len(feat_emb) < len(combined):
            cos_series = pd.Series(feat_emb, index=combined.index[:len(feat_emb)])
        else:
            cos_series = pd.Series(feat_emb[:len(combined)], index=combined.index)
        combined["cos_sim"] = cos_series
        combined = combined.sort_values("cos_sim", ascending=False, na_position="last")

        # table_rows = []
        
        word_infos = []
        subset_distributions = []
        good_bad_distributions = []
        for _, r in combined.iterrows():
            
            wid = str(r.get("word_id", ""))
            subset_goodbad = wid.split("_")[:-2]
            subset = "_".join(subset_goodbad[:-1])
            goodbad = subset_goodbad[-1]
            subset_distributions.append(subset)
            good_bad_distributions.append(goodbad)
            word_info = word_samples.get(wid, {"before":"","word":"","after":""})
            cos_val = r.get("cos_sim", np.nan)
            word_infos.append(word_info)
        
        # Create distribution charts
        chart_html = create_distribution_charts(subset_distributions, good_bad_distributions)
        
        results.append([corr, corr_spear, desc, trend, word_infos[:5], chart_html])
    return results


def create_html_table(results, output_file="correlation_results.html"):
    """Create an HTML table from results and save to file"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Correlation Analysis Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 20px 0;
            }
            th {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .number {
                text-align: right;
                font-family: monospace;
            }
            .trend {
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
                display: inline-block;
            }
            .trend-increasing {
                background-color: #d4edda;
                color: #155724;
            }
            .trend-decreasing {
                background-color: #f8d7da;
                color: #721c24;
            }
            .trend-other {
                background-color: #e7e7e7;
                color: #383838;
            }
            .word-context {
                font-family: monospace;
                font-size: 0.9em;
                background-color: #f8f9fa;
                padding: 4px;
                margin: 2px 0;
                border-left: 3px solid #007bff;
            }
            .word-highlight {
                background-color: #fff3cd;
                font-weight: bold;
            }
            .rank {
                background-color: #007bff;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-weight: bold;
                margin-right: 10px;
            }
            .chart-container {
                margin: 5px 0;
                background-color: white;
                padding: 5px;
                border-radius: 4px;
                border: 1px solid #ddd;
                min-width: 350px;
                max-width: 400px;
            }
            .chart-cell {
                min-width: 380px;
            }
        </style>
    </head>
    <body>
        <h1>Feature Correlation Analysis Results</h1>
        <p style="text-align: center; color: #666;">
            Sorted by Spearman Correlation (Descending)
        </p>
        <table>
            <thead>
                <tr>
                    <th style="width: 60px;">Rank</th>
                    <th style="width: 100px;">Pearson</th>
                    <th style="width: 100px;">Spearman</th>
                    <th style="width: 300px;">Description</th>
                    <th style="width: 100px;">Trend</th>
                    <th style="width: 300px;">Top Word Contexts</th>
                    <th style="width: 400px;">Distribution Charts</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for idx, (pearson, spearman, description, trend, word_infos, chart_html) in enumerate(results, 1):
        # Determine trend class
        trend_class = "trend-other"
        if "increasing" in trend.lower():
            trend_class = "trend-increasing"
        elif "decreasing" in trend.lower():
            trend_class = "trend-decreasing"
        
        # Format word contexts
        word_contexts_html = ""
        for word_info in word_infos:
            before = word_info.get("before", "")
            word = word_info.get("word", "")
            after = word_info.get("after", "")
            word_contexts_html += f'<div class="word-context">{before}<span class="word-highlight">{word}</span>{after}</div>'
        
        html_content += f"""
                <tr>
                    <td><span class="rank">{idx}</span></td>
                    <td class="number">{pearson:.4f}</td>
                    <td class="number">{spearman:.4f}</td>
                    <td>{description}</td>
                    <td><span class="trend {trend_class}">{trend}</span></td>
                    <td>{word_contexts_html}</td>
                    <td class="chart-cell"><div class="chart-container">{chart_html}</div></td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML table saved to: {output_file}")
    return output_file


# -----------------------------
if __name__=="__main__":
    
    results = get_correlated_trends(0.9)
    sorted_feats = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Create HTML table
    output_file = create_html_table(sorted_feats, "correlation_results.html")
    print(f"\nTotal features analyzed: {len(sorted_feats)}")
    print(f"Results saved to: {output_file}")