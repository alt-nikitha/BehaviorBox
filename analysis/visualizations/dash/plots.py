import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

MODEL_ORDER = [
    "llama2-7b",
    "llama2-13b",
    "olmo2-7b",
    "olmo2-13b",
    "llama2-7b-chat",
    "llama2-13b-chat",
    "olmo2-7b-dpo",
    "olmo2-13b-dpo",
    "pythia-160m-step1000",
    "pythia-160m"
]

MODEL_DISPLAY_NAMES = {
    "llama2-7b": "Llama-7B",
    "llama2-13b": "Llama-13B",
    "olmo2-7b": "OLMo-7B",
    "olmo2-13b": "OLMo-13B",
    "llama2-7b-chat": "Llama-7B-Chat",
    "llama2-13b-chat": "Llama-13B-Chat",
    "olmo2-7b-dpo": "OLMo-7B-DPO",
    "olmo2-13b-dpo": "OLMo-13B-DPO",
    "olmo2-7b-sft": "OLMo-7B-SFT",
    "olmo2-13b-sft": "OLMo-13B-SFT",
    "pythia-160m-step1000": "Pythia-160M-STEP1000",
    "pythia-160m": "Pythia-160M":
}

# Removed plot_feature_diffs_hist - not needed for n-model comparison

# Removed plot_feature_prob_diffs_cum_count - not needed for n-model comparison

# Removed plot_word_diffs_hist - not needed for n-model comparison

def plot_word_probs_hist(
    feature_df: pd.DataFrame,
):
    # For n-model data, look for model-specific probability columns
    model_prob_cols = [col for col in feature_df.columns if col.endswith('_prob_variance')]
    
    if model_prob_cols:
        # Use the n-model probability variance columns
        models = [col.replace('_prob_variance', '') for col in model_prob_cols]
        probs_df = pd.DataFrame()
        probs_df["probs"] = feature_df[model_prob_cols].values.flatten()
        probs_df["model"] = [model for model in models for _ in range(len(feature_df))]
    else:
        # Fallback to old behavior
        models = [x for x in feature_df.columns if x in MODEL_ORDER]
        models = sorted(models, key=lambda x: MODEL_ORDER.index(x))
        probs_df = pd.DataFrame()
        probs_df["probs"] = feature_df[models].values.flatten()
        probs_df["model"] = [model for model in models for _ in range(len(feature_df))]
    
    if len(probs_df) == 0:
        # If no data available, create a dummy plot
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No probability data available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Number of Words")
        return fig
    
    hist = sns.histplot(
        data=probs_df,
        x="probs",
        hue="model",
        kde=True,
    )
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Probability")
    plt.ylabel("Number of Words")
    plt.tight_layout()
    return hist.get_figure()