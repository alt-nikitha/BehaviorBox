import numpy as np
from tslearn.clustering import KShape
from tslearn.utils import to_time_series_dataset
from tslearn.metrics import cdist_shape_based
from sklearn.metrics import silhouette_score
import json
def auto_kshape_clustering(
    features,
    series_key="avg_probs",
    k_min=2,
    k_max=10
):
    """
    Automatically selects best k for k-Shape using silhouette score.
    """

    # ------------------------------
    # 1. Collect time series
    # ------------------------------
    feature_ids = []
    series_list = []

    for feat in features:
        fid = feat["feature_id"]
        ts = feat.get(series_key)
        if ts is None:
            continue
        ts = np.array(ts, dtype=float)
        if len(ts) == 0:
            continue

        feature_ids.append(fid)
        series_list.append(ts)

    series_array = np.vstack(series_list)
    print(f"Collected {len(series_array)} time series of length {series_array.shape[1]}")

    # ------------------------------
    # 2. Try different k values
    # ------------------------------
    best_k = None
    best_score = -np.inf
    best_model = None
    best_labels = None

    print("\nEvaluating cluster counts...\n")

    for k in range(k_min, k_max + 1):
        
        ks = KShape(n_clusters=k, n_init=5, verbose=False)
        labels = ks.fit_predict(series_array)

        # k-Shape uses normalized cross-correlation, so we compute a similarity matrix:
        series_ts = to_time_series_dataset(series_array)
        dist_matrix = cdist_dtw(series_ts)   # Already (1 - NCC)

        # silhouette_score expects distances, but NCC is similarity.
        # Convert similarity to distance:
        # distance = 1 - similarity
        sil_score = silhouette_score(1 - dist_matrix, labels, metric="precomputed")

        print(f"k={k}: silhouette={sil_score:.4f}")

        if sil_score > best_score:
            best_score = sil_score
            best_k = k
            best_model = ks
            best_labels = labels


    print("\nBest k =", best_k, "with silhouette =", best_score)

    # ------------------------------
    # 3. Produce clusters
    # ------------------------------
    clusters = {i: [] for i in range(best_k)}
    for fid, label in zip(feature_ids, best_labels):
        clusters[label].append(fid)

    # ------------------------------
    # 4. Print results
    # ------------------------------
    print("\n================= FINAL CLUSTER RESULTS =================\n")
    for c in range(best_k):
        print(f"\n### Cluster {c} ###\n")
        for fid in clusters[c]:
            desc = features[fid].get("description", "")
            print(f"- Feature {fid}: {desc}")
        print("\n--------------------------------------------------")

    return clusters, best_labels, best_model, best_k


path_to_features_file = "/home/nsrikant/BehaviorBoxNew/analysis/precomputed_data/Pythia-piletrainedonpile_6400.json"
with open(path_to_features_file, "r") as f:
    all_features = json.load(f)["features"]

clusters, labels, model, best_k = auto_kshape_clustering(
    all_features,
    series_key="median_probs",  # or "median_probs", "trend_avg_probs", etc.
    k_min=2,
    k_max=10
)
print(clusters)
print(labels)
print(best_k)
