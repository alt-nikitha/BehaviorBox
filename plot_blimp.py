#!/usr/bin/env python3
"""
Script to plot BLiMP results across model checkpoints.
Generates separate HTML visualizations for OLMo and Pythia models.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_checkpoint_step(folder_name: str) -> Tuple[str, int]:
    """
    Extract model name and step number from folder name.
    
    Args:
        folder_name: Name of the checkpoint folder (e.g., 'pythia-160m-step1000')
    
    Returns:
        Tuple of (model_name, step_number)
    """
    # Match pattern: model-name-step<number>
    match = re.match(r'(.+?)-step(\d+)', folder_name)
    if match:
        model_name = match.group(1)
        step = int(match.group(2))
        return model_name, step
    return folder_name, 0


def load_blimp_results(base_dir: str) -> Dict[str, Dict]:
    """
    Load all BLiMP results from checkpoint directories.
    
    Args:
        base_dir: Base directory containing checkpoint folders
    
    Returns:
        Dictionary mapping checkpoint names to their results
    """
    results = {}
    base_path = Path(base_dir)
    
    # Find all directories that might contain BLiMP results
    for item in base_path.iterdir():
        if item.is_dir():
            # Look for the JSON summary file
            json_file = item / "blimp_summary_seq_logprob.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[item.name] = data
    
    return results


def organize_by_model(results: Dict[str, Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
    """
    Organize results by model, sorted by checkpoint step.
    
    Args:
        results: Dictionary of checkpoint results
    
    Returns:
        Dictionary mapping model names to sorted lists of (step, data) tuples
    """
    models = {}
    
    for checkpoint_name, data in results.items():
        model_name, step = extract_checkpoint_step(checkpoint_name)
        
        if model_name not in models:
            models[model_name] = []
        
        models[model_name].append((step, data))
    
    # Sort by step number for each model
    for model_name in models:
        models[model_name].sort(key=lambda x: x[0])
    
    return models


def create_category_plots(model_data: List[Tuple[int, Dict]], model_name: str) -> go.Figure:
    """
    Create interactive plots for all categories of a single model.
    
    Args:
        model_data: List of (step, data) tuples for a model
        model_name: Name of the model
    
    Returns:
        Plotly figure with subplots
    """
    if not model_data:
        return None
    
    # Extract steps and categories
    steps = [step for step, _ in model_data]
    categories = sorted(model_data[0][1]['per_split_accuracy'].keys())
    
    # Calculate number of rows needed (4 plots per row)
    plots_per_row = 4
    num_rows = (len(categories) + plots_per_row - 1) // plots_per_row
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=plots_per_row,
        subplot_titles=categories,
        vertical_spacing=0.05,
        horizontal_spacing=0.08
    )
    
    # Plot each category
    for idx, category in enumerate(categories):
        row = idx // plots_per_row + 1
        col = idx % plots_per_row + 1
        
        # Extract accuracy values for this category across checkpoints
        accuracies = [
            data['per_split_accuracy'][category] 
            for _, data in model_data
        ]
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=accuracies,
                mode='lines+markers',
                name=category,
                showlegend=False,
                marker=dict(size=6),
                line=dict(width=2),
                hovertemplate='Step: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
            ),
            row=row,
            col=col
        )
        
        # Add a horizontal line at 50% (random chance)
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="red",
            opacity=0.3,
            row=row,
            col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Training Step", row=row, col=col, type='log')
        fig.update_yaxes(title_text="Accuracy (%)", row=row, col=col, range=[0, 100])
    
    # Update layout
    fig.update_layout(
        height=400 * num_rows,
        title_text=f"BLiMP Results: {model_name}<br><sub>Per-Category Accuracy Across Training Steps</sub>",
        title_font_size=20,
        showlegend=False
    )
    
    return fig


def create_overall_accuracy_plot(models_data: Dict[str, List[Tuple[int, Dict]]]) -> go.Figure:
    """
    Create a plot showing overall accuracy for all models.
    
    Args:
        models_data: Dictionary mapping model names to their data
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for model_name, model_data in sorted(models_data.items()):
        steps = [step for step, _ in model_data]
        overall_accuracies = [data['overall_accuracy'] for _, data in model_data]
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=overall_accuracies,
                mode='lines+markers',
                name=model_name,
                marker=dict(size=8),
                line=dict(width=3),
                hovertemplate='%{fullData.name}<br>Step: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
            )
        )
    
    # Add horizontal line at 50%
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="Random Chance (50%)"
    )
    
    fig.update_layout(
        title="Overall BLiMP Accuracy Across Training",
        xaxis_title="Training Step",
        yaxis_title="Overall Accuracy (%)",
        xaxis_type='log',
        yaxis_range=[0, 100],
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def main():
    """Main execution function."""
    # Directory containing checkpoint folders
    base_dir = "/home/nsrikant/BehaviorBoxNew/results/blimp_full"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        return
    
    print(f"Loading BLiMP results from {base_dir}...")
    results = load_blimp_results(base_dir)
    
    if not results:
        print("No BLiMP results found! Make sure the directory contains checkpoint folders with 'blimp_summary_seq_logprob.json' files.")
        return
    
    print(f"Found {len(results)} checkpoint(s)")
    
    # Organize by model
    models_data = organize_by_model(results)
    print(f"Found {len(models_data)} model(s): {', '.join(models_data.keys())}")
    
    # Separate OLMo and Pythia models
    olmo_models = {k: v for k, v in models_data.items() if 'olmo' in k.lower()}
    pythia_models = {k: v for k, v in models_data.items() if 'pythia' in k.lower()}
    
    # Create visualizations for Pythia models
    if pythia_models:
        print("\nGenerating Pythia visualizations...")
        
        # Overall accuracy plot
        pythia_overall = create_overall_accuracy_plot(pythia_models)
        
        # Individual model plots
        pythia_html_parts = [pythia_overall.to_html(full_html=False, include_plotlyjs='cdn')]
        
        for model_name, model_data in sorted(pythia_models.items()):
            print(f"  Creating plots for {model_name}...")
            fig = create_category_plots(model_data, model_name)
            if fig:
                pythia_html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        
        # Save Pythia HTML
        pythia_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pythia BLiMP Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 100%;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Pythia Model BLiMP Results</h1>
                {''.join(pythia_html_parts)}
            </div>
        </body>
        </html>
        """
        
        output_file = "pythia_blimp_results.html"
        with open(output_file, 'w') as f:
            f.write(pythia_html)
        print(f"✓ Saved Pythia results to {output_file}")
    
    # Create visualizations for OLMo models
    if olmo_models:
        print("\nGenerating OLMo visualizations...")
        
        # Overall accuracy plot
        olmo_overall = create_overall_accuracy_plot(olmo_models)
        
        # Individual model plots
        olmo_html_parts = [olmo_overall.to_html(full_html=False, include_plotlyjs='cdn')]
        
        for model_name, model_data in sorted(olmo_models.items()):
            print(f"  Creating plots for {model_name}...")
            fig = create_category_plots(model_data, model_name)
            if fig:
                olmo_html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        
        # Save OLMo HTML
        olmo_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>OLMo BLiMP Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 100%;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>OLMo Model BLiMP Results</h1>
                {''.join(olmo_html_parts)}
            </div>
        </body>
        </html>
        """
        
        output_file = "olmo_blimp_results.html"
        with open(output_file, 'w') as f:
            f.write(olmo_html)
        print(f"✓ Saved OLMo results to {output_file}")
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    main()