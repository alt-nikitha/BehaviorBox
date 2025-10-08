import base64
import dash
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

from dash import html, dcc, callback, Output, Input, dash_table, State
from io import BytesIO
from math import ceil

viz_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if viz_root not in sys.path:
    sys.path.insert(0, viz_root)

from visualizations.feature_container import FeatureContainer
from visualizations.plots import plot_word_probs_hist

pd.options.mode.chained_assignment = None  # default='warn'

#########
# Change this to the directory where your SAE data is stored
SAE_DIR = f"/home/nsrikant/BehaviorBoxNew/sae_outputs/n_comparison/_seed=42_ofw=_N=3000_k=50_lp=None"

# Change this to change how words per feature are ordered
# SORT_BY = "act_value"   # default
# SORT_BY = "centroid_embedding_dist"
SORT_BY = "centroid_cos_sim"
#########

LABEL_MODEL = "neulab-claude-sonnet-4-20250514"
CONFIG_FILE = f"{SAE_DIR}/config.json"
assert os.path.exists(CONFIG_FILE), f"Config file {CONFIG_FILE} does not exist."
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
SAE_NAME = config["name"]
k = 50

feature_metrics_to_display = [
    "num_samples_considered",
    "embedding_avg_dist",
    "embedding_avg_cos_sim",
    "prob_avg_dist",
    "label_valid"
]

# Load data
feature_label_info = pd.read_json(f"{SAE_DIR}/feature_labels_validated/{LABEL_MODEL}.json")
fc = FeatureContainer(SAE_DIR)

# Only look at validated features
validated_features = [x for x in feature_label_info if int(feature_label_info[x]["Score"]) > 0]

# Set pages
FEATURES_PER_PAGE = 20
TOTAL_FEATURES = len(validated_features)
TOTAL_PAGES = ceil(TOTAL_FEATURES / FEATURES_PER_PAGE)

# Initialize the Dash app
app = dash.Dash(__name__)

# Function to create base64 encoded images from Seaborn plots
def create_encoded_image(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64


# Removed create_prob_diff_histogram - not needed for n-model comparison


def get_label(feature):
    # Get the label for the feature
    if feature not in feature_label_info:
        return "NONE"
    label = feature_label_info[feature]["Description"]
    return label


def get_label_model(feature):
    # Get the label model for the feature
    if feature not in feature_label_info:
        return "NONE"
    label_model = feature_label_info[feature]["Model"]
    return label_model


def parse_array_string(array_str):
    """Parse array string from JSON into numpy array"""
    if isinstance(array_str, str):
        try:
            # Handle different formats:
            # Format 1: "[1.0, 2.8, 3.45]" (with brackets and commas)
            # Format 2: "\"[1.0 3.0 4.0]\" (with escaped quotes and spaces)
            
            # Remove outer quotes and brackets
            array_str = array_str.strip('"').strip("'").strip('[]')
            
            # Clean up newlines and extra whitespace
            array_str = array_str.replace('\n', ' ').replace('\r', ' ')
            
            # Try comma-separated first
            if ',' in array_str:
                return np.array([float(x.strip()) for x in array_str.split(',') if x.strip()])
            else:
                # Handle space-separated format
                return np.array([float(x.strip()) for x in array_str.split() if x.strip()])
        except (ValueError, AttributeError):
            # If parsing fails, return empty array
            return np.array([])
    return np.array(array_str)


def get_model_names_from_metrics():
    """Extract model names from the feature metrics CSV filename"""
    try:
        # Find the metrics CSV file
        metrics_files = [f for f in os.listdir(SAE_DIR) if f.startswith('feature_metrics-') and f.endswith('.csv')]
        if not metrics_files:
            return []
        
        # Extract model names from filename
        # Format: feature_metrics-model1_model2_model3_model4_model5.csv
        filename = metrics_files[0].replace('feature_metrics-', '').replace('.csv', '')
        model_names = filename.split('_')
        
        return model_names
            
    except Exception as e:
        print(f"Error getting model names: {e}")
        return []


def get_feature_trend(feature):
    """Determine the trend of a feature across models"""
    if feature not in feature_label_info:
        return "unknown"
    
    feature_data = feature_label_info[feature]
    avg_probs = parse_array_string(feature_data.get("Avg Probs", "[]"))
    
    if len(avg_probs) < 2:
        return "unknown"
    
    # Calculate trend
    diff = np.diff(avg_probs)
    positive_diffs = np.sum(diff > 0.01)  # threshold for significant increase
    negative_diffs = np.sum(diff < -0.01)  # threshold for significant decrease
    zero_diffs = np.sum(np.abs(diff) <= 0.01)  # threshold for stagnation
    
    if positive_diffs > 0 and negative_diffs == 0:
        return "increasing"
    elif negative_diffs > 0 and positive_diffs == 0:
        return "decreasing"
    elif positive_diffs > 0 and negative_diffs > 0:
        # Check if it's increase-decrease or decrease-increase
        if diff[0] > 0:
            return "increase-decrease"
        else:
            return "decrease-increase"
    elif positive_diffs > 0 and zero_diffs > 0:
        return "increase-stagnate"
    elif negative_diffs > 0 and zero_diffs > 0:
        return "decrease-stagnate"
    else:
        return "stagnate"


def group_features_by_trend():
    """Group features by their probability trends"""
    groups = {
        "increasing": [],
        "decreasing": [],
        "increase-decrease": [],
        "decrease-increase": [],
        "increase-stagnate": [],
        "decrease-stagnate": [],
        "stagnate": [],
        "unknown": []
    }
    
    for feature in validated_features:
        trend = get_feature_trend(feature)
        groups[trend].append(feature)
    
    return groups


# Initialize feature groups after function definitions
feature_groups = group_features_by_trend()


def create_group_visualization(group_name, features):
    """Create visualization for a group of features"""
    if not features:
        return None
    
    # Collect data for all features in the group
    all_avg_probs = []
    all_avg_ranks = []
    model_names = []
    
    for feature in features:
        feature_data = feature_label_info[feature]
        avg_probs = parse_array_string(feature_data.get("Avg Probs", "[]"))
        avg_ranks = parse_array_string(feature_data.get("Mean Ranks", "[]"))
        
        if len(avg_probs) > 0:
            all_avg_probs.append(avg_probs)
            all_avg_ranks.append(avg_ranks)
            if not model_names:
                # Get actual model names from the feature metrics CSV
                model_names = get_model_names_from_metrics()
                # If we don't have enough model names, use generic ones
                if len(model_names) < len(avg_probs):
                    model_names.extend([f"Model {i+1}" for i in range(len(model_names), len(avg_probs))])
    
    if not all_avg_probs:
        return None
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average probabilities
    avg_probs_array = np.array(all_avg_probs)
    mean_probs = np.mean(avg_probs_array, axis=0)
    std_probs = np.std(avg_probs_array, axis=0)
    
    x_pos = np.arange(len(model_names))
    bars1 = ax1.bar(x_pos, mean_probs, yerr=std_probs, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Average Probability')
    ax1.set_title(f'{group_name.title()} Features - Average Probabilities')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45)
    
    # Plot 2: Average ranks
    avg_ranks_array = np.array(all_avg_ranks)
    mean_ranks = np.mean(avg_ranks_array, axis=0)
    std_ranks = np.std(avg_ranks_array, axis=0)
    
    bars2 = ax2.bar(x_pos, mean_ranks, yerr=std_ranks, capsize=5, alpha=0.7, color='orange')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Average Rank')
    ax2.set_title(f'{group_name.title()} Features - Average Ranks')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45)
    
    plt.tight_layout()
    return fig

# Function to create feature container
def create_feature_container(feature):
    label = get_label(feature)    
    
    feature_df = fc.get_feature_info(feature)[0]
    if SORT_BY == "centroid_embedding_dist":
        feature_df = feature_df.sort_values(SORT_BY, ascending=True)
    elif SORT_BY == "centroid_cos_sim":
        feature_df = feature_df.sort_values(SORT_BY, ascending=False)
    feature_df = feature_df.round(3)
    feature_metrics = fc.get_feature_info(feature)[1]
    feature_metrics = {k: feature_metrics[k] for k in feature_metrics_to_display if k in feature_metrics}
    feature_metrics["Num Samples"] = feature_metrics.pop("num_samples_considered")
    feature_metrics["Embedding Avg Dist"] = feature_metrics.pop("embedding_avg_dist")
    feature_metrics["Embedding Avg Cos Sim"] = feature_metrics.pop("embedding_avg_cos_sim")
    feature_metrics["Prob Avg Dist"] = feature_metrics.pop("prob_avg_dist")
    if "label_valid" in feature_metrics:
        feature_metrics["Percent Valid"] = feature_metrics.pop("label_valid")
    
    valid_sample_feature_metrics = fc.get_valid_sample_feature_metrics(feature, feature_df)
    
    model_probs_hist = create_encoded_image(plot_word_probs_hist(feature_df))
    
    # Get n-model data for this feature
    n_model_data = None
    if feature in feature_label_info:
        feature_data = feature_label_info[feature]
        avg_probs = parse_array_string(feature_data.get("Avg Probs", "[]"))
        avg_ranks = parse_array_string(feature_data.get("Mean Ranks", "[]"))
        median_ranks = parse_array_string(feature_data.get("Median Ranks", "[]"))
        
        if len(avg_probs) > 0:
            # Get actual model names
            model_names = get_model_names_from_metrics()
            
            # Check if all arrays have the same length
            if (len(avg_probs) == len(avg_ranks) == len(median_ranks)):
                # Map model indices to names: Model 1 -> first model, Model 2 -> second model, etc.
                display_names = []
                for i in range(len(avg_probs)):
                    if i < len(model_names):
                        display_names.append(model_names[i])
                    else:
                        display_names.append(f"Model {i+1}")
                
                n_model_data = {
                    'Model': display_names,
                    'Avg Probability': avg_probs,
                    'Avg Rank': avg_ranks,
                    'Median Rank': median_ranks
                }
                n_model_df = pd.DataFrame(n_model_data).round(4)
            else:
                # If arrays don't match, skip the n-model table
                print(f"Warning: Array length mismatch for feature {feature}")
                print(f"avg_probs: {len(avg_probs)}, avg_ranks: {len(avg_ranks)}, median_ranks: {len(median_ranks)}")
                n_model_data = None
    
    # Feature header and description
    feature_header = html.Div(
        [
            html.H2(f"[{get_label_model(feature)}] Feature {feature}: {label}", style={'text-align': 'left', 'margin-bottom': '10px'}),\
        ],
        style={'padding': '10px', 'border-bottom': '1px solid #ccc'}
    )

    # Histogram container on the left
    histograms_container = html.Div(
        [
            html.H3('Model Probabilities Distribution', style={'text-align': 'left', 'margin-bottom': '10px'}),
            html.Img(src='data:image/png;base64,{}'.format(model_probs_hist), style={'width': '100%', 'margin': '5px 0'}),
        ],
        style={'display': 'flex', 'flex-direction': 'column', 'width': '30%'}
    )

    # Placeholder for other information
    other_info_components = [
        html.H3('Feature Information', style={'text-align': 'left', 'margin-bottom': '10px'}),
        dash_table.DataTable(
            data=[feature_metrics],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'fontWeight': 'bold'}    
        ),
        html.H3('Feature Information (filtered for valid samples)', style={'text-align': 'left', 'margin-bottom': '10px'}),
        dash_table.DataTable(
            data=[valid_sample_feature_metrics],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'fontWeight': 'bold'}    
        )
    ]
    
    # Add n-model data table if available
    if n_model_data is not None:
        other_info_components.extend([
            html.H3('N-Model Comparison Data', style={'text-align': 'left', 'margin-bottom': '10px'}),
            dash_table.DataTable(
                data=n_model_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'}
            )
        ])
    
    other_info_components.append(
        dash_table.DataTable(
            data=feature_df.to_dict('records'),
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'fontWeight': 'bold'}
        )
    )
    
    other_info = html.Div(
        other_info_components,
        style={'width': '70%', 'padding-left': '10px'}
    )

    # Combined container
    combined_container = html.Div(
        [histograms_container, other_info],
        style={'display': 'flex', 'justify-content': 'space-between'}
    )

    # Full feature container
    container = html.Div(
        [feature_header, combined_container],
        style={
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'margin': '20px',
            'padding': '10px',
            'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
            'background-color': '#FFFFFF'
        }
    )

    return container


# Define the app layout
app.layout = html.Div(
    [
        html.H1(f"{SAE_NAME}", style={'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
        html.Div(
            [
                html.Div([
                    html.H2(f"Number of features: {len(validated_features)}", style={'text-align': 'left', 'font-family': 'Arial, sans-serif'}),
                ],
                style={'padding': '10px', 'text-align': 'left', 'border-bottom': '1px solid #ccc'}
                ),
                # Removed diffs histograms - not needed for n-model comparison
            ],
            style={
                'border': '1px solid #ccc',
                'border-radius': '5px',
                'margin': '20px',
                'padding': '10px',
                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'background-color': '#FFFFFF'
            }
        ),
        # Add group visualization section
        html.Div(
            [
                html.H2("Feature Groups by Trend", style={'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
                html.Div(id='group-visualizations-content'),
            ],
            style={
                'border': '1px solid #ccc',
                'border-radius': '5px',
                'margin': '20px',
                'padding': '10px',
                'box-shadow': '2px 2px 5px rgba(0,0,0,0.1)',
                'background-color': '#FFFFFF'
            }
        ),
        dcc.Loading(
            id="loading-container",
            type="circle",
            children=html.Div(id='feature-containers-content'),
            style={'margin-top': '20px'}
        ),
        html.Div(
            [
                html.Div(id='page-buttons', style={'text-align': 'center', 'margin': '20px 0'}),
            ],
            style={'text-align': 'center', 'margin-top': '20px'}
        ),
        dcc.Store(id='app-initialized', data=True)  # Simple trigger for initial load
    ],
    style={'padding': '20px', 'background-color': '#FAF8F4'}
)


# Removed pagination callback since we're not using pagination anymore


# Callback to update feature containers (no pagination needed)
@callback(
    Output('feature-containers-content', 'children'),
    [Input('app-initialized', 'data')]  # Simple trigger for initial load
)
def update_feature_containers(trigger):
    
    # Create grouped feature containers
    all_containers = []
    
    # Define the order of trend groups to display
    trend_order = ["increasing", "decreasing", "increase-decrease", "decrease-increase", 
                   "increase-stagnate", "decrease-stagnate", "stagnate", "unknown"]
    
    for trend in trend_order:
        if trend in feature_groups and feature_groups[trend]:
            # Add trend header
            trend_header = html.Div(
                [
                    html.H2(f"{trend.title()} Features ({len(feature_groups[trend])} features)", 
                           style={'text-align': 'left', 'margin': '20px 0 10px 0', 
                                 'padding': '10px', 'background-color': '#f0f0f0', 
                                 'border-left': '4px solid #007BFF'})
                ]
            )
            all_containers.append(trend_header)
            
            # Add feature containers for this trend
            for feature in feature_groups[trend]:
                container = create_feature_container(feature)
                all_containers.append(container)
    
    return all_containers


# Callback to update page buttons
@callback(
    Output('page-buttons', 'children'),
    [Input('app-initialized', 'data')]  # Simple trigger for initial load
)
def update_page_buttons(trigger):
    # Since we're showing all features grouped by trend, we don't need pagination
    # But we can show a summary of the groups
    group_summary = []
    
    for trend in ["increasing", "decreasing", "increase-decrease", "decrease-increase", 
                  "increase-stagnate", "decrease-stagnate", "stagnate", "unknown"]:
        if trend in feature_groups and feature_groups[trend]:
            count = len(feature_groups[trend])
            group_summary.append(
                html.Span(f"{trend.title()}: {count}", 
                         style={'margin': '0 10px', 'padding': '5px 10px', 
                               'background-color': '#e9ecef', 'border-radius': '3px'})
            )
    
    return group_summary


# Callback to update group visualizations
@callback(
    Output('group-visualizations-content', 'children'),
    [Input('app-initialized', 'data')]  # Simple trigger for initial load
)
def update_group_visualizations(trigger):
    # Group features by trend
    groups = group_features_by_trend()
    
    visualizations = []
    
    for group_name, features in groups.items():
        if not features:
            continue
            
        # Create visualization for this group
        fig = create_group_visualization(group_name, features)
        if fig is not None:
            img_base64 = create_encoded_image(fig)
            
            group_container = html.Div(
                [
                    html.H3(f"{group_name.title()} Features ({len(features)} features)", 
                           style={'text-align': 'left', 'margin-bottom': '10px'}),
                    html.Img(src='data:image/png;base64,{}'.format(img_base64), 
                           style={'max-width': '100%', 'height': 'auto', 'margin': '10px 0'}),
                ],
                style={
                    'border': '1px solid #ddd',
                    'border-radius': '5px',
                    'margin': '10px 0',
                    'padding': '10px',
                    'background-color': '#f9f9f9'
                }
            )
            visualizations.append(group_container)
    
    return visualizations


def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()