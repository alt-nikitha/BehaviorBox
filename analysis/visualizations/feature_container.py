import numpy as np
import pandas as pd
import os
import json

class FeatureContainer:
    def __init__(self, sae_dir: str):
        self.sae_dir = sae_dir
        self.top_activations = pd.read_csv(f"{self.sae_dir}/top-50_activations.csv")
        self.top_wic = pd.read_json(f"{self.sae_dir}/top-50_words_in_context.json")
        self.model_names = self.__get_model_names()
        self.feature_metrics = pd.read_csv(f"{self.sae_dir}/feature_metrics-{'_'.join(self.model_names)}.csv")
        self.sample_centroid_metrics = pd.read_csv(f"{self.sae_dir}/feature_sample_centroid-metrics.csv")
        self.feature_label_info = feature_label_info = pd.read_json(f"{self.sae_dir}/feature_labels_validated/neulab-claude-sonnet-4-20250514.json")
        if os.path.exists(f"{self.sae_dir}/feature_labels_validated/sample_annotation.csv"):
            self.sample_annotations = pd.read_csv(f"{self.sae_dir}/feature_labels_validated/sample_annotation.csv")
        else:
            self.sample_annotations = None
    
    def __get_model_names(self):
        sae_config = json.load(open(f"{self.sae_dir}/config.json", "r"))
        models = sae_config["model_names"]
        return models
    
    def __get_sample_annotation(self, x, feature_id):
        word_id = x["word_id"]
        sample_annotation = self.sample_annotations[
            (self.sample_annotations["feature"] == feature_id) & (self.sample_annotations["word_id"] == word_id)
        ]
        if len(sample_annotation.index) > 0:
            return sample_annotation.iloc[0]["valid"]
        else:
            return None
    
    def get_feature_info(self, feature_id: int) -> tuple[pd.DataFrame, dict]:
        feature_df = self.top_activations[self.top_activations["feature"] == feature_id]
        # filter activations
        # keep activation and associated sample if
        # activation value is in the top 3 quartiles or >= 0.25 * max_act
        feature_df = feature_df[feature_df["act_value"] > 0]
        max_act_threshold = 0.25 * feature_df["act_value"].max()
        feature_df = feature_df[(feature_df["act_value"] >= max_act_threshold) | (feature_df["act_value"] >= feature_df["act_value"].quantile(0.25))]
        feature_centroid_metrics = self.sample_centroid_metrics[self.sample_centroid_metrics["feature"] == feature_id]
        feature_centroid_metrics.drop(columns=["feature"], inplace=True)
        word_ids = feature_df["word_id"].values
        act_values = feature_df["act_value"].values
        
        model_logprobs = {
            model: feature_df[model].values for model in self.model_names
        }
        model_probs = {
            model: np.exp(model_logprobs[model]) for model in self.model_names
        }
        model_probs = pd.DataFrame(model_probs)
        model_diffs = pd.DataFrame()
        model_diffs["prob_diff"] = model_probs[self.model_names[0]] - model_probs[self.model_names[1]]
        model_diffs["logprob_diff"] = model_logprobs[self.model_names[0]] - model_logprobs[self.model_names[1]]
        
        before = []
        word = []
        after = []
        for word_id in word_ids:
            before.append(self.top_wic[word_id]["before"])
            word.append(self.top_wic[word_id]["word"])
            after.append(self.top_wic[word_id]["after"])
        feature_df = pd.DataFrame({
            "word_id": word_ids,
            "act_value": act_values,
            "before": before,
            "word": word,
            "after": after,
            "centroid_embedding_dist": feature_centroid_metrics["sample_centroid_embedding_dist"].values,
            "centroid_cos_sim": feature_centroid_metrics["sample_centroid_cos_sim"].values,
        })
        feature_df["act_value"] = feature_df["act_value"].round(3)
        if self.sample_annotations is not None:
            feature_df["label_valid"] = feature_df.apply(lambda x: self.__get_sample_annotation(x, feature_id), axis=1)
        feature_df = pd.concat([feature_df, model_probs, model_diffs], axis=1)
        feature_metric_df = self.feature_metrics[self.feature_metrics["feature"] == feature_id]
        if self.sample_annotations is not None:
            if "label_valid" in feature_metric_df.columns:
                feature_metric_df["label_valid"] = (feature_df["label_valid"] == "YES").sum() / len(feature_df)
        feature_metric_df = feature_metric_df.round(3)
        feature_metrics = feature_metric_df.to_dict(orient="records")[0]
        
        return feature_df, feature_metrics

    def get_valid_sample_feature_metrics(
        self,
        feature_id: int,
        feature_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        if feature_df is None:
            feature_df = self.get_feature_info(feature_id)[0]
        if "label_valid" in feature_df.columns:
            feature_df = feature_df[feature_df["label_valid"] == "YES"]
            feature_metrics = {}
            feature_metrics["Num Samples"] = len(feature_df)
            feature_metrics["Prob Avg Diff"] = round(feature_df["prob_diff"].mean(), 3)
            feature_metrics["Prob Median Diff"] = round(feature_df["prob_diff"].median(), 3)
            feature_metrics["LogProb Avg Diff"] = round(feature_df["logprob_diff"].mean(), 3)
            feature_metrics["LogProb Median Diff"] = round(feature_df["logprob_diff"].median(), 3)
            feature_metrics["Consistency"] = round(max(np.sum(feature_df["prob_diff"] > 0), np.sum(feature_df["prob_diff"] < 0)) / feature_df["prob_diff"].shape[0], 3)
            return feature_metrics
        return {}