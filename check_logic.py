import pandas as pd
import os
input_feature_dir = "/home/nsrikant/BehaviorBoxNew/output/validation_split_1000/input_features"


file_df = pd.read_csv(f"{input_feature_dir}/file_to_doc.csv")
file_df.drop(columns=["num_words"], inplace=True)
doc_pos = {}

doc_id = "valid_294"
word_pos = 48
if doc_id in doc_pos:
    doc_pos[doc_id].append(word_pos)
else:
    doc_pos[doc_id] = [word_pos]
doc_ids = list(doc_pos.keys())
print(doc_ids)
file_df = file_df[file_df["doc_id"].isin(doc_ids)]
print(file_df.head())
words_in_context = {}
for file in file_df["file"].to_list():
    docs = file_df[file_df["file"] == file]["doc_id"].to_list()
    print(docs)
    file = os.path.join(input_feature_dir, file)
    docs_df = pd.read_parquet(file, columns=["doc_id", "word"], filters=[("doc_id", 'in', docs)])
    print(docs_df.head())
    for doc in docs:
        doc_df = docs_df[docs_df["doc_id"] == doc]
        doc_words = doc_df["word"].tolist()
        print(doc_words)


'''
models = [
"stage1-step1000-tokens3B",
"stage1-step2000-tokens5B",
"stage1-step3000-tokens7B",
"stage1-step4000-tokens9B",
"stage1-step5000-tokens11B",
"stage1-step6000-tokens13B",
"stage1-step7000-tokens15B",
"stage1-step8000-tokens17B",
"stage1-step9000-tokens19B",
"stage1-step10000-tokens21B",
]
'''