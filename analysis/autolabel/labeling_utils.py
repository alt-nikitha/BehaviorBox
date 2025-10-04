import pandas as pd

def get_relevant_features(
    feature_metrics: pd.DataFrame,
):
    """
    Gets the indices of the features that have a absolute median logprob diff of > 1
    """
    feature_metrics = feature_metrics[(feature_metrics["logprob_median_diff"].abs() > 1) | (feature_metrics["prob_median_diff"].abs() > 0.1)]
    return feature_metrics["feature"].values


def get_activations_and_wic(
    top_acts: pd.DataFrame,
    top_wic: pd.DataFrame,
    feature_id: int
) -> pd.DataFrame:
    feature_acts = top_acts[top_acts['feature'] == feature_id]
    # drop activations that are 0
    feature_acts = feature_acts[feature_acts["act_value"] != 0]
    max_act = feature_acts["act_value"].max()
    # keep activation and associated sample if
    # activation value is in the top 3 quartiles or >= 0.25 * max_act
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