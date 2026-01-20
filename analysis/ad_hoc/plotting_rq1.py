import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def plot_feature_acquisition_3groups(csv_path, n_clusters_per_group=4, save_path="acquisition_plot.png"):

    # -------------------------
    # Load + sort by acquisition
    # -------------------------
    df = pd.read_csv(csv_path)
    df = df.sort_values("acq_checkpoint").reset_index(drop=True)

    # -------------------------
    # Split into early / mid / late
    # -------------------------
    N = len(df)
    df["stage"] = "mid"

    df.loc[: N//3, "stage"] = "early"      # first third
    df.loc[N//3 : 2*N//3, "stage"] = "mid" # middle third
    df.loc[2*N//3 : , "stage"] = "late"    # final third

    # -------------------------
    # Cluster descriptions within each stage
    # -------------------------
    df["cluster"] = -1  # default

    for stage in ["early", "mid", "late"]:
        subset = df[df["stage"] == stage]

        if len(subset) == 0:
            continue

        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(subset["description"])

        # KMeans
        kmeans = KMeans(
            n_clusters=min(n_clusters_per_group, len(subset)),
            n_init=20,
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(X)

        df.loc[subset.index, "cluster"] = cluster_labels

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(14, 18))

    y_positions = range(len(df))

    # Assign colors from tab20
    import numpy as np
    max_cluster = df["cluster"].max() + 1
    colors = plt.cm.tab20(df["cluster"] / max_cluster)

    # Scatter
    plt.scatter(df["acq_checkpoint"], y_positions, c=colors, s=80, edgecolor="black", linewidth=0.3)

    # Labels
    for i, row in df.iterrows():
        plt.text(
            row["acq_checkpoint"] + 0.05,
            y_positions[i],
            f"{row['feature_group_id']} ({row['stage']})",
            fontsize=8,
            verticalalignment="center",
        )

    plt.xlabel("Acquisition Checkpoint", fontsize=12)
    plt.ylabel("Feature Order", fontsize=12)
    plt.title("Feature Acquisition — Grouped into Early, Mid, Late (with Clustering)", fontsize=15)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")


# Example:
# plot_feature_acquisition_3groups("features.csv", n_clusters_per_group=4)



# Example call:
plot_feature_acquisition_3groups("/home/nsrikant/BehaviorBoxNew/analysis/rq1_qualitative_6400/acquired_increase-stagnate_Pythia-160m.csv")
