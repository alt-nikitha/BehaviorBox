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
}

def plot_feature_diffs_hist(
    feature_label_info: pd.DataFrame,
    logprob: bool = False,
):
    df = pd.DataFrame()
    validated_features = [x for x in feature_label_info if int(feature_label_info[x]["Score"]) > 0]
    df["model"] = [feature_label_info[x]["Model"] for x in validated_features]
    models = df["model"].unique()
    models = sorted(models, key=lambda x: MODEL_ORDER.index(x))
    print(models)
    
    if not logprob:
        df["diff"] = [feature_label_info[x]["Median Prob Diff"] for x in validated_features]
        x_label = "Median Probability Difference"
        y_label = "Number of Features"
    else:
        df["diff"] = [feature_label_info[x]["Median Logprob Diff"] for x in validated_features]
        x_label = "Median Log Probability Difference"
        y_label = "Number of Features"
    
    # Create figure and axis explicitly
    fig, ax = plt.subplots()
    
    # Create custom palette to ensure consistency
    palette = sns.color_palette("tab10", n_colors=len(models))
    
    # Plot with explicit palette
    hist = sns.histplot(
        data=df, 
        x="diff", 
        hue="model", 
        kde=True, 
        ax=ax,
    )
    
    # Create custom legend patches
    import matplotlib.patches as mpatches
    patches = []
    for i, model in enumerate(models):
        count = len(df[df['model'] == model])
        label = f"{MODEL_DISPLAY_NAMES[model]} ({count})"
        patch = mpatches.Patch(color=palette[i], label=label)
        patches.append(patch)
    
    # Add custom legend with patches
    ax.legend(handles=patches, title="Better Model")
    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()
    
    return fig

def plot_feature_prob_diffs_cum_count(
    feature_label_info: dict,
    logprob: bool = False,
):
    """
    Plots the cumulative count of validated features,
    with a separate line for each model showing features where that model wins
    """
    df = pd.DataFrame()
    valiadated_features = [x for x in feature_label_info if int(feature_label_info[x]["Score"]) > 0]
    df["model"] = [feature_label_info[x]["Model"] for x in valiadated_features]
    models = df["model"].unique()
    models = sorted(models, key=lambda x: MODEL_ORDER.index(x))
    model_order = [x for x in MODEL_ORDER if x in models]
    if not logprob:
        df["diff"] = [feature_label_info[x]["Median Prob Diff"] for x in valiadated_features]
        x_label = "Median Probability Difference"
    else:
        df["diff"] = [feature_label_info[x]["Median Logprob Diff"] for x in valiadated_features]
        x_label = "Median Log Probability Difference"
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    all_sorted_diffs = []
    all_cum_counts = []
    model_col = []
    # Plot cumulative count for each model
    for model in models:
        model_df = df[df["model"] == model]
        # Sort the diffs and calculate cumulative count
        sorted_diffs = sorted(model_df["diff"])
        cum_count = np.arange(1, len(sorted_diffs) + 1)
        all_sorted_diffs += sorted_diffs
        all_cum_counts += cum_count.tolist()
        model_col += [model] * len(sorted_diffs)
    
    plot_df = pd.DataFrame()
    plot_df["diff"] = all_sorted_diffs
    plot_df["cum_count"] = all_cum_counts
    plot_df["model"] = model_col
    # Plot the line for this model
    plot = sns.lineplot(data=plot_df, x="diff", y="cum_count", hue="model", hue_order=model_order)
    labels = [f"{MODEL_DISPLAY_NAMES[model]} ({len(df[df['model'] == model])})" for model in models]
    plt.legend(title="Better Model", labels=labels)
    plt.xlabel(x_label)
    plt.ylabel("Cumulative Count of Features")
    plt.legend(title="Better Model")
    plt.tight_layout()
    return plot

def plot_word_diffs_hist(
    feature_df: pd.DataFrame,
    logprob: bool = False,
):
    if logprob:
        diff = "logprob_diff"
        x_label = "Log Probability Difference"
    else:
        diff = "prob_diff"
        x_label = "Probability Difference"
    hist = sns.histplot(
        feature_df[diff],
        kde=True,
    )
    # add a dashed line at 0
    plt.axvline(0, linestyle="--", color="red")
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel(x_label)
    plt.ylabel("Number of Words")
    plt.tight_layout()
    return hist.get_figure()

def plot_word_probs_hist(
    feature_df: pd.DataFrame,
):
    models = [x for x in feature_df.columns if x in MODEL_ORDER]
    models = sorted(models, key=lambda x: MODEL_ORDER.index(x))
    probs_df = pd.DataFrame()
    probs_df["probs"] = feature_df[models].values.flatten()
    probs_df["model"] = [model for model in models for _ in range(len(feature_df))]
    hist = sns.histplot(
        data=probs_df,
        x="probs",
        hue="model",
        hue_order=models,
        kde=True,
    )
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Probability")
    plt.ylabel("Number of Words")
    plt.tight_layout()
    return hist.get_figure()