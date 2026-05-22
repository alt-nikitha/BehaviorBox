import pandas as pd
import numpy as np

def classify_trend_soft(arr, tol=1e-5):
    arr = np.array(arr)
    try:
        diffs = np.diff(arr)
    except:
        print(arr)
    
    pos = np.sum(diffs > tol)
    neg = np.sum(diffs < -tol)
    zero = np.sum(np.abs(diffs) <= tol)
    
    n = len(diffs)
    
    # Roughly monotone
    if pos / n > 0.8 and neg / n < 0.1:
        return "roughly increasing"
    elif neg / n > 0.8 and pos / n < 0.1:
        return "roughly decreasing"
    
    # Roughly monotone with stagnant part
    elif pos / n > 0.5 and zero / n > 0.2 and neg / n < 0.1:
        return "roughly increasing to stagnant"
    elif neg / n > 0.5 and zero / n > 0.2 and pos / n < 0.1:
        return "roughly decreasing to stagnant"
    
    # Increasing then decreasing
    peak_idx = np.argmax(arr)
    if peak_idx > 0 and peak_idx < len(arr)-1:
        before_peak = arr[:peak_idx+1]
        after_peak = arr[peak_idx:]
        if np.sum(np.diff(before_peak) > -tol)/len(before_peak) > 0.6 and \
           np.sum(np.diff(after_peak) < tol)/len(after_peak) > 0.6:
            return "roughly increasing then decreasing"
    
    # Decreasing then increasing
    trough_idx = np.argmin(arr)
    if trough_idx > 0 and trough_idx < len(arr)-1:
        before_trough = arr[:trough_idx+1]
        after_trough = arr[trough_idx:]
        if np.sum(np.diff(before_trough) < tol)/len(before_trough) > 0.6 and \
           np.sum(np.diff(after_trough) > -tol)/len(after_trough) > 0.6:
            return "roughly decreasing then increasing"
    
    # If nothing matches
    return "other"



def get_relevant_features(
    feature_metrics: pd.DataFrame,
):
    return feature_metrics["feature"].values


def get_activations_and_wic(
    top_acts: pd.DataFrame,
    top_wic: pd.DataFrame,
    feature_id: int
) -> pd.DataFrame:
    feature_acts = top_acts[top_acts['feature'] == feature_id]
    # drop activations that are 0
    feature_acts = feature_acts[feature_acts["act_value"] != 0]
    if len(feature_acts.index) == 0:
        return None
    max_act = feature_acts["act_value"].max()
    # keep activation and associated sample if
    # activation value is in the top 3 quartiles AND >= 0.25 * max_act
    # (drops weak activations whose value is small relative to the feature's peak)
    max_act_threshold = 0.25 * max_act
    feature_acts = feature_acts[
        (feature_acts["act_value"] >= max_act_threshold) | (feature_acts["act_value"] >= feature_acts["act_value"].quantile(0.25))]
    if len(feature_acts.index) == 0:
        return None
    feature_word_ids = feature_acts["word_id"]
    before = []
    word = []
    after = []
    for word_id in feature_word_ids:
        before.append(top_wic[word_id]["before"])
        word.append(top_wic[word_id]["word"])
        after.append(top_wic[word_id]["after"])
    feature_acts["before"] = before
    feature_acts["word"] = word
    feature_acts["after"] = after
    return feature_acts


def sample_context_string(
    feature_act: pd.Series
):
    def format_word_string(
        word: str,
    ) -> str:
        word = word.replace(" ", "<SPACE>")
        word = word.replace("\t", "<TAB>")
        word = word.replace("\n", "<NEWLINE>")
        return word
    word = feature_act["word"]
    before = feature_act["before"].lstrip()
    after = feature_act["after"].rstrip()
    context_string = f"{word}: {before} *{format_word_string(word)}* {after}"
    return context_string


def format_context_string(
    feature_acts: pd.DataFrame,
    num_words: int = 20
) -> str:
    feature_acts = feature_acts.head(num_words)
    lines = []
    for i in range(min(num_words, len(feature_acts.index))):
        context_string = sample_context_string(feature_acts.iloc[i])
        line = f"- {context_string}"
        lines.append(line)
    return "\n".join(lines)