# X_probs: (N, V) probabilities or logits reduced (e.g., top-k prob vector)
# y_space: 1 if the original text/word had leading space else 0
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

model_name = "pythia-160m-step1000"
logprobs_path = f"/mnt/labshare/nsrikant/bbox_outputs/.cache/pythia-160m_pythia-160m-step1000/validation_split_1000/{model_name}/logprobs.pkl"
word_ids_path = "/mnt/labshare/nsrikant/bbox_outputs/.cache/pythia-160m_pythia-160m-step1000/validation_split_1000/word_ids.pkl"

mapping_file = "/home/nsrikant/BehaviorBoxNew/output/validation_split_1000/input_features/file_to_doc.csv"
word_embeddings_folder = "/home/nsrikant/BehaviorBoxNew/output/validation_split_1000/input_features"

logprobs = pd.read_pickle(logprobs_path)
word_ids = pd.read_pickle(word_ids_path)
mapping_df = pd.read_csv(mapping_file)


doc_to_file = dict(zip(mapping_df["doc_id"], mapping_df["file"]))

all_parquet_files = [os.path.join(word_embeddings_folder, f) for f in os.listdir(word_embeddings_folder) if f.endswith(".parquet")]
print(f"Loading {len(all_parquet_files)} parquet files...")

embed_df_list = []
for fpath in tqdm(all_parquet_files):
    df = pd.read_parquet(fpath)
    embed_df_list.append(df)

embed_df = pd.concat(embed_df_list, ignore_index=True)


word_df = pd.DataFrame({"word_id": word_ids, "logprob": logprobs})

merged_df = word_df.merge(embed_df, on="word_id", how="inner")


embedding_cols = [c for c in merged_df.columns if c.startswith("embedding_")]

def clean_string(s):
    return s.replace("Ġ", " ").replace("Ċ", "\t")

merged_df["cleaned"] = merged_df["word"].str.replace("Ġ", " ").str.replace("Ċ", "\t")
merged_df["label"] = (merged_df["word"] != merged_df["cleaned"]).astype(int)


logprobs_correlation_X = merged_df["logprob"].to_numpy().reshape(-1, 1)
logprobs_correlation_Y = merged_df["label"].to_numpy()

embeds_correlation_X = embed_df[embedding_cols].to_numpy(dtype=np.float32)



        



clf = LogisticRegression(max_iter=200).fit(logprobs_correlation_X, logprobs_correlation_Y)
auc = roc_auc_score(logprobs_correlation_Y, clf.decision_function(logprobs_correlation_X))
print("Pythia-prob whitespace AUC:", auc)






# X_embeds: (N, D)
# clf2 = LogisticRegression(max_iter=1000, verbose=1).fit(embeds_correlation_X, logprobs_correlation_Y)
# print("Longformer-embed whitespace AUC:", roc_auc_score(logprobs_correlation_Y, clf2.decision_function(embeds_correlation_X)))

# import numpy as np
print("mean ||emb||:", np.mean(np.linalg.norm(embeds_correlation_X, axis=1)))
# print("mean ||probs||:", np.mean(np.linalg.norm(logprobs_correlation_X, axis=1)))
