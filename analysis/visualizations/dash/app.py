import base64
import dash
import json
import matplotlib.pyplot as plt
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
from visualizations.plots import plot_feature_diffs_hist, plot_word_diffs_hist, plot_word_probs_hist

pd.options.mode.chained_assignment = None  # default='warn'

#########
# Change this to the directory where your SAE data is stored
SAE_DIR = f"/data/tir/projects/tir3/users/lindiat/ltjuatja/bbox/sae/arxiv/llama2-7b_llama2-13b_seed=42_ofw=0.7_N=3000_k=50_lp=None"

# Change this to change how words per feature are ordered
# SORT_BY = "act_value"   # default
# SORT_BY = "centroid_embedding_dist"
SORT_BY = "centroid_cos_sim"
#########

LABEL_MODEL = "claude-3-5-sonnet-20241022"
CONFIG_FILE = f"{SAE_DIR}/config.json"
assert os.path.exists(CONFIG_FILE), f"Config file {CONFIG_FILE} does not exist."
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
SAE_NAME = config["name"]
k = 50

feature_metrics_to_display = [
    "num_samples_considered",
    "prob_avg_diff",
    "prob_median_diff",
    "logprob_avg_diff",
    "logprob_median_diff",
    "prob_diff_consistency",
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


def create_prob_diff_histogram(logprob: bool = False):
    hist = plot_feature_diffs_hist(
        feature_label_info,
        logprob=logprob
    )
    hist_fig = hist.get_figure()
    img_base64 = create_encoded_image(hist_fig)
    return img_base64


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
    feature_metrics = {k: feature_metrics[k] for k in feature_metrics_to_display}
    feature_metrics["Num Samples"] = feature_metrics.pop("num_samples_considered")
    feature_metrics["Prob Avg Diff"] = feature_metrics.pop("prob_avg_diff")
    feature_metrics["Prob Median Diff"] = feature_metrics.pop("prob_median_diff")
    feature_metrics["LogProb Avg Diff"] = feature_metrics.pop("logprob_avg_diff")
    feature_metrics["LogProb Median Diff"] = feature_metrics.pop("logprob_median_diff")
    feature_metrics["Consistency"] = feature_metrics.pop("prob_diff_consistency")
    feature_metrics["Percent Valid"] = feature_metrics.pop("label_valid")
    
    valid_sample_feature_metrics = fc.get_valid_sample_feature_metrics(feature, feature_df)
    
    model_probs_hist = create_encoded_image(plot_word_probs_hist(feature_df))
    model_diffs_hist = create_encoded_image(plot_word_diffs_hist(feature_df))
    
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
            html.Img(src='data:image/png;base64,{}'.format(model_probs_hist), style={'width': '100%', 'margin': '5px 0'}),
            html.Img(src='data:image/png;base64,{}'.format(model_diffs_hist), style={'width': '100%', 'margin': '5px 0'}),
        ],
        style={'display': 'flex', 'flex-direction': 'column', 'width': '30%'}
    )

    # Placeholder for other information
    other_info = html.Div(
        [   
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
            ),
            dash_table.DataTable(
                data=feature_df.to_dict('records'),
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'fontWeight': 'bold'}
            ),
        ],
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
                html.Div(
                    [
                        html.Div(
                            html.Img(src='data:image/png;base64,{}'.format(create_prob_diff_histogram(logprob=False)), style={'max-width': '100%', 'height': 'auto'}),
                            style={'flex': '1', 'padding': '10px'}
                        ),
                        html.Div(
                            html.Img(src='data:image/png;base64,{}'.format(create_prob_diff_histogram(logprob=True)), style={'max-width': '100%', 'height': 'auto'}),
                            style={'flex': '1', 'padding': '10px'}
                        ),
                    ],
                    style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}
                ),
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
                html.Button("Previous", id='prev-btn', n_clicks=0),
                html.Div(id='page-buttons', style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Button("Next", id='next-btn', n_clicks=0),
            ],
            style={'text-align': 'center', 'margin-top': '20px'}
        ),
        dcc.Store(id='current-page', data=1)  # Store to track the current page
    ],
    style={'padding': '20px', 'background-color': '#FAF8F4'}
)


# Callback for pagination controls
@callback(
    Output("current-page", "data"),
    [Input("prev-btn", "n_clicks"), Input("next-btn", "n_clicks")],
    [State("current-page", "data")]
)
def update_page_number(prev_clicks, next_clicks, current_page):
    current_page = current_page or 1
    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "prev-btn" and prev_clicks > 0 and current_page > 1:
        return current_page - 1
    elif triggered_id == "next-btn" and next_clicks > 0 and current_page < TOTAL_PAGES:
        return current_page + 1
    return current_page


# Callback to update feature containers based on the current page
@callback(
    Output('feature-containers-content', 'children'),
    [Input('current-page', 'data')]
)
def update_feature_containers(page):
    if page is None:
        page = 1
    
    start_idx = (page - 1) * FEATURES_PER_PAGE
    end_idx = min(start_idx + FEATURES_PER_PAGE, TOTAL_FEATURES)
    
    current_features = validated_features[start_idx:end_idx]
    
    containers = [create_feature_container(feature) for feature in current_features]
        
    return containers


# Callback to update page buttons
@callback(
    Output('page-buttons', 'children'),
    [Input('current-page', 'data')]
)
def update_page_buttons(current_page):
    if current_page is None:
        current_page = 1
    
    # Create page buttons (show at most 5 pages around the current page)
    start_page = max(1, current_page - 2)
    end_page = min(TOTAL_PAGES, start_page + 4)
    
    buttons = []
    for i in range(start_page, end_page + 1):
        if i == current_page:
            style = {'margin': '0 5px', 'padding': '5px 10px', 'background-color': '#007BFF', 'color': 'white'}
        else:
            style = {'margin': '0 5px', 'padding': '5px 10px'}
        
        buttons.append(html.Button(str(i), id=f'page-{i}', style=style))
    
    return buttons


def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()