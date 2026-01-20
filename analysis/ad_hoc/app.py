import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

st.set_page_config(page_title="Feature Specificity Explorer", layout="wide")

# Load data
@st.cache_data
def load_data(model):
    if model == "160m":
        json_path = "/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_blimp_full/Pythia-piletrainedonpile.json"
    else:
        json_path = "/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data_blimp_full/Pythia6.9b-piletrainedonpile.json"
    
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_specificity(feature):
    """Calculate feature specificity"""
    subset_correlations = feature.get('subset_correlations', {})
    
    if not subset_correlations:
        return None, None, {}
    
    subset_corrs = {}
    for subset_name, corr_data in subset_correlations.items():
        residual_corr = corr_data.get('residual_pearson_correlation')
        raw_corr = corr_data.get('raw_pearson_correlation')
        corr_to_use = residual_corr if residual_corr is not None else raw_corr
        
        if corr_to_use is not None:
            subset_corrs[subset_name] = abs(corr_to_use)
    
    if len(subset_corrs) < 2:
        return None, None, subset_corrs
    
    corr_values = list(subset_corrs.values())
    max_corr = np.max(corr_values)
    mean_corr = np.mean(corr_values)
    
    specificity = max_corr - mean_corr
    specificity_ratio = max_corr / (mean_corr + 0.01)
    
    return specificity, specificity_ratio, subset_corrs

# Sidebar
st.sidebar.title("Configuration")
model_choice = st.sidebar.radio("Select Model", ["160m", "6.9b"])

# Load data
data = load_data(model_choice)
features = data['features']
model_names = data['model_names']

# Calculate specificity for all features
feature_specs = []

for feature in features:
    spec, spec_ratio, subset_corrs = calculate_specificity(feature)
    
    if spec is None:
        continue
    
    max_subset = max(subset_corrs.items(), key=lambda x: x[1])
    subset_name, max_corr = max_subset
    
    other_corrs = [c for s, c in subset_corrs.items() if s != subset_name]
    mean_other = np.mean(other_corrs) if other_corrs else 0
    
    feature_specs.append({
        'feature_id': feature.get('feature_id'),
        'description': feature.get('description', ''),
        'topic': feature.get('topic', 'Other'),
        'specificity': spec,
        'specificity_ratio': spec_ratio,
        'best_subset': subset_name,
        'best_corr': max_corr,
        'mean_other_corr': mean_other,
        'subset_correlations': subset_corrs,
        'feature_obj': feature,
    })

# Sort by specificity
feature_specs.sort(key=lambda x: x['specificity'], reverse=True)

# Filter for truly specific
truly_specific = [
    f for f in feature_specs 
    if f['best_corr'] > 0.6 and f['mean_other_corr'] < 0.3 and f['specificity_ratio'] > 2.0
]

# Main UI
st.title("🎯 Feature Specificity Explorer")
st.markdown(f"**Model:** Pythia-{model_choice} | **Total Features:** {len(feature_specs)}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Features Analyzed", len(feature_specs))
with col2:
    st.metric("Truly Specific Features", len(truly_specific))
with col3:
    avg_specificity = np.mean([f['specificity'] for f in feature_specs])
    st.metric("Avg Specificity", f"{avg_specificity:.4f}")

st.divider()

# Filter options
st.subheader("🔍 Filter Features")

col1, col2, col3 = st.columns(3)

with col1:
    show_truly_specific = st.checkbox(
        "Show only truly specific features",
        value=True,
        help="Best subset > 0.6, others < 0.3, ratio > 2.0"
    )

with col2:
    min_best_corr = st.slider("Min correlation with best subset", 0.0, 1.0, 0.5)

with col3:
    max_other_corr = st.slider("Max correlation with others", 0.0, 1.0, 0.4)

# Apply filters
display_features = feature_specs.copy()

if show_truly_specific:
    display_features = truly_specific

display_features = [
    f for f in display_features
    if f['best_corr'] >= min_best_corr and f['mean_other_corr'] <= max_other_corr
]

st.info(f"Showing {len(display_features)} features")

st.divider()

# Display features
st.subheader("📊 Highly Specific Features")

for idx, feat_spec in enumerate(display_features):
    feature_id = feat_spec['feature_id']
    description = feat_spec['description']
    best_subset = feat_spec['best_subset']
    best_corr = feat_spec['best_corr']
    
    with st.expander(
        f"⭐ **[{feature_id}]** {description} → {best_subset} (r={best_corr:.3f})",
        expanded=(idx == 0)
    ):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Subset", best_subset)
        with col2:
            st.metric("Best Correlation", f"{best_corr:.4f}")
        with col3:
            st.metric("Others (avg)", f"{feat_spec['mean_other_corr']:.4f}")
        with col4:
            st.metric("Specificity", f"{feat_spec['specificity']:.4f}")
        
        st.markdown(f"**Topic:** {feat_spec['topic']}")
        
        st.divider()
        
        # Plot subset correlations
        st.markdown("**Correlation with all BLIMP subsets:**")
        
        subset_corrs = feat_spec['subset_correlations']
        
        # Create bar chart
        subset_names = sorted(subset_corrs.items(), key=lambda x: x[1], reverse=True)
        names = [s[0] for s in subset_names]
        corrs = [s[1] for s in subset_names]
        colors = [
            '#2ca02c' if name == best_subset else '#ff7f0e' if corr > 0.5 else '#d62728'
            for name, corr in subset_names
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names,
            x=corrs,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{c:.3f}" for c in corrs],
            textposition='auto',
        ))
        
        fig.update_layout(
            height=max(300, len(names) * 20),
            xaxis_title="Residual Correlation",
            yaxis_title="BLIMP Subset",
            template='plotly_white',
            margin=dict(l=150, r=50, t=20, b=50),
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{feature_id}")
        
        # Plot feature trajectory
        st.markdown("**Feature Probability Trajectory:**")
        
        feature = feat_spec['feature_obj']
        median_probs = feature.get('median_probs', [])
        subset_corr_data = feature.get('subset_correlations', {}).get(best_subset, {})
        subset_perf = subset_corr_data.get('subset_performance_at_checkpoints', [])
        
        if median_probs and any(median_probs) and subset_perf:
            fig2 = go.Figure()
            
            # Feature probability
            fig2.add_trace(go.Scatter(
                x=model_names,
                y=median_probs,
                mode='lines+markers',
                name='Feature Prob',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8),
            ))
            
            # Subset performance
            subset_perf_norm = [x / 100.0 for x in subset_perf]
            residual_corr = subset_corr_data.get('residual_pearson_correlation')
            
            fig2.add_trace(go.Scatter(
                x=model_names,
                y=subset_perf_norm,
                mode='lines+markers',
                name=f'{best_subset} Performance',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                marker=dict(size=8),
            ))
            
            fig2.update_layout(
                height=400,
                template='plotly_white',
                hovermode='x unified',
                yaxis_title='Probability / Performance',
                xaxis_title='Checkpoint',
                margin=dict(l=50, r=20, t=30, b=40),
            )
            
            st.plotly_chart(fig2, use_container_width=True, key=f"line_{feature_id}")

# Footer
st.divider()
st.caption("🎯 Finding linguistically meaningful features through specificity analysis")