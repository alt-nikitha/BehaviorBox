import streamlit as st
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Trend-Based Topic Acquisition Analysis", layout="wide")

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42

np.random.seed(SEED)

# ============================================================================
# MODEL SELECTION
# ============================================================================

st.sidebar.title("Configuration")
PYTHIA_MODEL = st.sidebar.radio(
    "Select Pythia Model",
    ["160m", "6.9b"],
    key="pythia_model_selector",
    help="Choose which model to analyze"
)

# ============================================================================
# LOAD DATA WITH TOPICS
# ============================================================================

topics_dir = f"/home/nsrikant/BehaviorBoxNew/analysis/topics/Pythia{PYTHIA_MODEL}"
json_path = f"{topics_dir}/data_with_topics.json"

@st.cache_data(hash_funcs={str: lambda x: x})
def load_data(model_path):
    with open(model_path, 'r') as f:
        return json.load(f)

try:
    data = load_data(json_path)
except FileNotFoundError:
    st.error(f"❌ Data not found: {json_path}")
    st.info(f"Run `python generate_topics_llm_only.py` with PYTHIA_MODEL='{PYTHIA_MODEL}' first")
    st.stop()

features = data['features']
model_names = data['model_names']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_acq_checkpoint(median_probs):
    for idx, prob in enumerate(median_probs):
        if prob is not None and prob >= 0.5:
            return idx
    return -1

def get_trend(median_probs):
    """Determine trend: increasing, increase-stagnate, increase-decrease, decreasing, other"""
    if not median_probs or len(median_probs) < 2:
        return "other"
    
    valid_probs = [p for p in median_probs if p is not None]
    if len(valid_probs) < 2:
        return "other"
    
    # Check if monotonically increasing
    is_increasing = all(valid_probs[i] <= valid_probs[i+1] for i in range(len(valid_probs)-1))
    
    # Check if monotonically decreasing
    is_decreasing = all(valid_probs[i] >= valid_probs[i+1] for i in range(len(valid_probs)-1))
    
    if is_increasing:
        return "increasing"
    elif is_decreasing:
        return "decreasing"
    else:
        # Check for increase-stagnate vs increase-decrease
        if valid_probs[-1] > valid_probs[0]:
            return "increase-stagnate"
        else:
            return "increase-decrease"

def get_trend(median_probs):
    """Determine trend: increasing, increase-stagnate, increase-decrease, decreasing, other"""
    if not median_probs or len(median_probs) < 2:
        return "other"
    
    valid_probs = [p for p in median_probs if p is not None]
    if len(valid_probs) < 2:
        return "other"
    
    # Check if monotonically increasing
    is_increasing = all(valid_probs[i] <= valid_probs[i+1] for i in range(len(valid_probs)-1))
    
    # Check if monotonically decreasing
    is_decreasing = all(valid_probs[i] >= valid_probs[i+1] for i in range(len(valid_probs)-1))
    
    if is_increasing:
        return "increasing"
    elif is_decreasing:
        return "decreasing"
    else:
        # Check for increase-stagnate vs increase-decrease
        if valid_probs[-1] > valid_probs[0]:
            return "increase-stagnate"
        else:
            return "increase-decrease"

def get_stagnation_checkpoint(median_probs):
    """Find where stagnation begins (last 30% with low variance)"""
    if not median_probs or len(median_probs) < 3:
        return -1
    
    valid_probs = [p for p in median_probs if p is not None]
    if len(valid_probs) < 3:
        return -1
    
    # Check last 30% of points
    stag_threshold = max(1, len(valid_probs) - len(valid_probs)//3)
    stag_portion = valid_probs[stag_threshold:]
    
    if len(stag_portion) > 0 and np.std(stag_portion) < 0.1:
        return stag_threshold
    return -1

def get_peak_checkpoint(median_probs):
    """Find where peak occurs"""
    if not median_probs:
        return -1
    
    valid_probs = [(i, p) for i, p in enumerate(median_probs) if p is not None]
    if not valid_probs:
        return -1
    
    peak_idx, _ = max(valid_probs, key=lambda x: x[1])
    return peak_idx

# ============================================================================
# PROCESS ALL FEATURES
# ============================================================================

all_features = []
for feature in features:
    median_probs = feature.get('median_probs', [])
    acq_checkpoint = get_acq_checkpoint(median_probs)
    
    # Get trend from trend_median_probs field
    trend = feature.get('trend_median_probs')
    if not trend:
        trend = get_trend(median_probs)
    
    all_features.append({
        'feature_id': feature.get('feature_id'),
        'description': feature.get('description', ''),
        'acq_checkpoint': acq_checkpoint,
        'stagnation_checkpoint': get_stagnation_checkpoint(median_probs) if acq_checkpoint >= 0 else -1,
        'peak_checkpoint': get_peak_checkpoint(median_probs) if acq_checkpoint >= 0 else -1,
        'pearson': feature.get('pearson_corr'),
        'spearman': feature.get('spearman_corr'),
        'topic': feature.get('topic', 'Other'),
        'trend': trend,
        'acquired': acq_checkpoint >= 0,
        'median_probs': median_probs,
    })

st.sidebar.write("---")
st.sidebar.write(f"**Selected Model:** Pythia-{PYTHIA_MODEL}")
st.sidebar.write(f"**Seed:** {SEED}")

# ============================================================================
# PAGE
# ============================================================================

st.title("Topic Acquisition Analysis by Feature Trend")
st.markdown("**RQ1: Understanding the order of topic acquisition across training**")

# ============================================================================
# GET UNIQUE TRENDS
# ============================================================================

unique_trends = sorted(set(f['trend'] for f in all_features))

# ============================================================================
# PROCESS EACH TREND
# ============================================================================

for trend in unique_trends:
    st.subheader(f"📊 Trend: {trend.upper()}")
    
    # Split into acquired and not acquired
    trend_acquired = [f for f in all_features if f['trend'] == trend and f['acquired']]
    trend_not_acquired = [f for f in all_features if f['trend'] == trend and not f['acquired']]
    
    # ========================================================================
    # SECTION 1: ACQUIRED TOPICS - ORDERED BY ACQUISITION
    # ========================================================================
    
    if trend_acquired:
        st.markdown(f"**✅ Acquired Features ({len(trend_acquired)} total)**")
        
        # Group by topic
        topics_acquired = defaultdict(list)
        for feature in trend_acquired:
            topic = feature['topic']
            topics_acquired[topic].append(feature)
        
        # Sort topics by when they're first acquired
        topic_acq_order = []
        for topic, feats in topics_acquired.items():
            earliest_ckpt = min([f['acq_checkpoint'] for f in feats])
            topic_acq_order.append((earliest_ckpt, topic, len(feats)))
        
        topic_acq_order.sort()
        
        # Display ordered topics
        st.markdown("**Topics in order of acquisition (earliest → latest):**")
        
        acq_data = []
        for earliest_ckpt, topic, count in topic_acq_order:
            feats = topics_acquired[topic]
            pearson_vals = [f['pearson'] for f in feats if f['pearson']]
            avg_pearson = np.mean([abs(p) for p in pearson_vals]) if pearson_vals else 0
            high_corr = len([p for p in pearson_vals if abs(p) > 0.7])
            
            acq_data.append({
                'Topic': topic,
                'Features': count,
                'First Acquired': model_names[earliest_ckpt],
                'Checkpoint': earliest_ckpt,
                'Avg |Pearson|': f"{avg_pearson:.4f}",
                'High Corr': high_corr,
            })
        
        df_acq = pd.DataFrame(acq_data)
        df_acq = df_acq.drop('Checkpoint', axis=1)
        st.dataframe(df_acq, use_container_width=True)
    
    else:
        st.info("No acquired features in this trend")
    
    # ========================================================================
    # SECTION 2: NOT ACQUIRED TOPICS
    # ========================================================================
    
    if trend_not_acquired:
        st.markdown(f"**❌ Not Acquired Features ({len(trend_not_acquired)} total)**")
        st.markdown("_Features that never reached 50% median probability_")
        
        # Group by topic
        topics_not_acq = defaultdict(list)
        for feature in trend_not_acquired:
            topics_not_acq[feature['topic']].append(feature)
        
        # Sort topics by count
        topic_not_acq_order = []
        for topic, feats in topics_not_acq.items():
            topic_not_acq_order.append((len(feats), topic))
        
        topic_not_acq_order.sort(reverse=True)
        
        # Display ordered topics
        st.markdown("**Topics not acquired (sorted by feature count):**")
        
        not_acq_data = []
        for count, topic in topic_not_acq_order:
            feats = topics_not_acq[topic]
            max_probs = [max([p for p in f['median_probs'] if p is not None], default=0) for f in feats]
            avg_max_prob = np.mean(max_probs) if max_probs else 0
            
            not_acq_data.append({
                'Topic': topic,
                'Features': count,
                'Avg Max Prob': f"{avg_max_prob:.4f}",
            })
        
        df_not_acq = pd.DataFrame(not_acq_data)
        st.dataframe(df_not_acq, use_container_width=True)
        
        # Show detailed features table
        with st.expander(f"📋 Detailed Not-Acquired Features ({len(trend_not_acquired)})"):
            detail_data = []
            for feature in trend_not_acquired:
                # Get max probability reached
                median_probs = feature['median_probs']
                max_prob = max([p for p in median_probs if p is not None], default=0) if median_probs else 0
                
                # Only include if never reached 0.5
                if max_prob < 0.5:
                    detail_data.append({
                        'Feature ID': feature['feature_id'],
                        'Topic': feature['topic'],
                        'Max Prob Reached': f"{max_prob:.4f}",
                        'Gap to 0.5': f"{0.5 - max_prob:.4f}",
                        'Description': feature['description'][:100],
                    })
            
            if detail_data:
                df_details = pd.DataFrame(detail_data).sort_values('Max Prob Reached', ascending=False)
                st.dataframe(df_details, use_container_width=True, height=400)
            else:
                st.info("No features with max prob < 0.5")
    
    else:
        st.info("All features were acquired in this trend")
    
    st.divider()
    
    # ========================================================================
    # TREND-SPECIFIC SECTIONS
    # ========================================================================
    
    if trend == "increase-stagnate":
        st.subheader(f"🛑 Stagnation Analysis")
        st.markdown("**Features ordered by when they begin to stagnate:**")
        
        trend_with_stag = [f for f in trend_acquired if f['stagnation_checkpoint'] >= 0]
        
        if trend_with_stag:
            # Group by topic and stagnation point
            stag_data = []
            for feature in trend_with_stag:
                stag_data.append({
                    'Feature ID': feature['feature_id'],
                    'Topic': feature['topic'],
                    'Acquired': model_names[feature['acq_checkpoint']],
                    'Stagnates': model_names[feature['stagnation_checkpoint']],
                    'Stag Checkpoint': feature['stagnation_checkpoint'],
                    'Description': feature['description'][:80],
                })
            
            df_stag = pd.DataFrame(stag_data).sort_values('Stag Checkpoint')
            df_stag = df_stag.drop('Stag Checkpoint', axis=1)
            st.dataframe(df_stag, use_container_width=True, height=400)
        else:
            st.info("No features with detectable stagnation point")
    
    elif trend == "increase-decrease":
        st.subheader(f"📈 Peak Analysis")
        st.markdown("**Features ordered by when they peak:**")
        
        trend_with_peak = [f for f in trend_acquired if f['peak_checkpoint'] >= 0]
        
        if trend_with_peak:
            # Sort by peak checkpoint
            peak_data = []
            for feature in trend_with_peak:
                peak_data.append({
                    'Feature ID': feature['feature_id'],
                    'Topic': feature['topic'],
                    'Acquired': model_names[feature['acq_checkpoint']],
                    'Peaks': model_names[feature['peak_checkpoint']],
                    'Peak Checkpoint': feature['peak_checkpoint'],
                    'Description': feature['description'][:80],
                })
            
            df_peak = pd.DataFrame(peak_data).sort_values('Peak Checkpoint')
            df_peak = df_peak.drop('Peak Checkpoint', axis=1)
            st.dataframe(df_peak, use_container_width=True, height=400)
        else:
            st.info("No features with detectable peak point")
        
        # Add decrease analysis
        st.subheader(f"📉 Decrease Analysis")
        st.markdown("**Features ordered by when they begin to decrease (after peak):**")
        
        trend_with_decrease = [f for f in trend_acquired if f['peak_checkpoint'] >= 0]
        
        if trend_with_decrease:
            # Find when decrease starts (first point after peak that's lower)
            decrease_data = []
            for feature in trend_with_decrease:
                median_probs = feature['median_probs']
                peak_idx = feature['peak_checkpoint']
                
                # Find first point after peak where prob decreases
                decrease_checkpoint = -1
                if peak_idx >= 0 and peak_idx < len(median_probs) - 1:
                    for idx in range(peak_idx + 1, len(median_probs)):
                        if median_probs[idx] is not None and median_probs[peak_idx] is not None:
                            if median_probs[idx] < median_probs[peak_idx]:
                                decrease_checkpoint = idx
                                break
                
                if decrease_checkpoint >= 0:
                    decrease_data.append({
                        'Feature ID': feature['feature_id'],
                        'Topic': feature['topic'],
                        'Acquired': model_names[feature['acq_checkpoint']],
                        'Peaks': model_names[peak_idx],
                        'Decreases': model_names[decrease_checkpoint],
                        'Decrease Checkpoint': decrease_checkpoint,
                        'Description': feature['description'][:80],
                    })
            
            if decrease_data:
                df_decrease = pd.DataFrame(decrease_data).sort_values('Decrease Checkpoint')
                df_decrease = df_decrease.drop('Decrease Checkpoint', axis=1)
                st.dataframe(df_decrease, use_container_width=True, height=400)
            else:
                st.info("No features with detectable decrease point")
        else:
            st.info("No features with peak points")
    
    st.divider()


