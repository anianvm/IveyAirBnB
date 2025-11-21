import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. FIND OPTIMAL k (ELBOW, SILHOUETTE, CH, DB)
# ============================================================

def find_optimal_k(data, max_k=15, method='all', explain=True):
    k_range = range(2, max_k + 1)
    inertia_values, silhouette_scores, ch_scores, db_scores = [], [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(data)

        inertia_values.append(km.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))
        ch_scores.append(calinski_harabasz_score(data, labels))
        db_scores.append(davies_bouldin_score(data, labels))

        if explain:
            print(f"Evaluated k={k} | Silhouette={silhouette_scores[-1]:.3f}")

    # ---- Plotting ----
    plt.figure(figsize=(18, 10))

    # Elbow
    plt.subplot(2, 2, 1)
    plt.plot(k_range, inertia_values, 'bo-')
    plt.title("Elbow Method (WCSS)")
    plt.grid(True)

    kl = KneeLocator(k_range, inertia_values, curve="convex", direction="decreasing")
    elbow_k = kl.elbow

    if elbow_k:
        plt.axvline(elbow_k, color='r', linestyle='--')

    # Silhouette
    plt.subplot(2, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.title("Silhouette Score")
    plt.grid(True)

    best_sil = k_range[np.argmax(silhouette_scores)]
    plt.axvline(best_sil, color='r', linestyle='--')

    # CH
    plt.subplot(2, 2, 3)
    plt.plot(k_range, ch_scores, 'mo-')
    plt.title("Calinski-Harabasz Score")
    plt.grid(True)

    best_ch = k_range[np.argmax(ch_scores)]
    plt.axvline(best_ch, color='r', linestyle='--')

    # DB
    plt.subplot(2, 2, 4)
    plt.plot(k_range, db_scores, 'co-')
    plt.title("Davies-Bouldin Score (Lower = Better)")
    plt.grid(True)

    best_db = k_range[np.argmin(db_scores)]
    plt.axvline(best_db, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()

    votes = [best_sil, best_ch, best_db]
    if elbow_k:
        votes.append(elbow_k)

    final_k = Counter(votes).most_common(1)[0][0]

    if explain:
        print("\nRecommended k from metrics:")
        print(f"Silhouette best k: {best_sil}")
        print(f"Calinski-Harabasz best k: {best_ch}")
        print(f"Davies-Bouldin best k: {best_db}")
        print(f"Elbow best k: {elbow_k}")
        print(f"\nFinal recommended k: {final_k}")

    return final_k


# ============================================================
# 2. FIND OPTIMAL EPS FOR DBSCAN
# ============================================================

def find_optimal_eps(data, min_samples=5, explain=True):
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)

    k_dist = np.sort(distances[:, min_samples - 1])

    plt.figure(figsize=(12, 6))
    plt.plot(k_dist)
    plt.title(f"K-distance Plot (k={min_samples})")
    plt.grid(True)

    kl = KneeLocator(range(len(k_dist)), k_dist,
                     curve='convex', direction='increasing')

    if kl.knee:
        eps = k_dist[kl.knee]
        if explain:
            print(f"Suggested eps ≈ {eps:.3f}")
        return eps

    if explain:
        print("Automatic eps not detected. Inspect the plot.")
    return None


# ============================================================
# 3. CLUSTERING RUNNERS (ALL HAVE explain=True OPTION)
# ============================================================

def run_kmeans(data, n_clusters=3, explain=True):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(data)

    if explain:
        sil = silhouette_score(data, labels)
        print(f"\nKMeans ({n_clusters} clusters) Silhouette: {sil:.3f}")

        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Cluster {u}: {c} samples ({c/len(labels):.1%})")

    return model, labels


def run_hierarchical(data, n_clusters=3, linkage_method='ward', explain=True):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(data)

    if explain:
        sil = silhouette_score(data, labels)
        print(f"\nHierarchical ({linkage_method}) Silhouette: {sil:.3f}")

    return model, labels


def run_dbscan(data, eps=0.5, min_samples=5, explain=True):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    if explain:
        non_noise = labels != -1
        if len(np.unique(labels[non_noise])) > 1:
            sil = silhouette_score(data[non_noise], labels[non_noise])
            print(f"\nDBSCAN Silhouette (non-noise): {sil:.3f}")

        # Print cluster sizes
        for lbl in np.unique(labels):
            count = sum(labels == lbl)
            name = "Noise" if lbl == -1 else f"Cluster {lbl}"
            print(f"{name}: {count} samples ({count/len(labels):.1%})")

    return model, labels


def run_gmm(data, n_components=3, explain=True):
    model = GaussianMixture(n_components=n_components, random_state=42)
    labels = model.fit_predict(data)

    if explain:
        sil = silhouette_score(data, labels)
        bic = model.bic(data)
        aic = model.aic(data)
        print(f"\nGMM ({n_components} components)")
        print(f"Silhouette: {sil:.3f}")
        print(f"BIC: {bic:.1f}")
        print(f"AIC: {aic:.1f}")

    return model, labels


# ============================================================
# 4. DENDROGRAM
# ============================================================

def plot_dendrogram(data, max_samples=150):
    sample = data.sample(min(len(data), max_samples))
    linked = linkage(sample, method='ward')

    plt.figure(figsize=(16, 6))
    dendrogram(linked)
    plt.title("Dendrogram (Hierarchical Clustering)")
    plt.show()


# ============================================================
# 5. FEATURE IMPORTANCE & PLOTS
# ============================================================

def get_feature_importance(data, labels, top_n=10):
    from sklearn.feature_selection import f_classif
    f_vals, p_vals = f_classif(data, labels)
    return pd.DataFrame({
        "Feature": data.columns,
        "F-Value": f_vals,
        "P-Value": p_vals
    }).sort_values("F-Value", ascending=False).head(top_n)


def plot_cluster_heatmap(data, labels, top_n=10):
    df = data.copy()
    df["Cluster"] = labels

    top_feats = get_feature_importance(data, labels, top_n)["Feature"]
    heat = df.groupby("Cluster")[top_feats].mean()

    plt.figure(figsize=(12, 8))
    sns.heatmap(heat, annot=True, cmap="viridis")
    plt.title("Cluster Heatmap (Top Features)")
    plt.show()


def plot_radar_chart(data, labels, top_n=6):
    df = data.copy()
    df["Cluster"] = labels

    top_feats = get_feature_importance(data, labels, top_n)["Feature"]
    centers = df.groupby("Cluster")[top_feats].mean()

    scaled = MinMaxScaler().fit_transform(centers)
    centers_scaled = pd.DataFrame(scaled, index=centers.index, columns=top_feats)

    N = len(top_feats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(15, 8))

    for idx, (cluster, row) in enumerate(centers_scaled.iterrows()):
        ax = plt.subplot(1, len(centers_scaled), idx + 1, polar=True)
        values = row.tolist()
        values.append(values[0])

        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_feats)
        ax.set_title(f"Cluster {cluster}")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. TEXTUAL CLUSTER EXPLANATION
# ============================================================

def explain_clusters(data, labels, top_n=10):
    df = data.copy()
    df['Cluster'] = labels

    important = get_feature_importance(data, labels, top_n)['Feature']

    for c in sorted(df['Cluster'].unique()):
        if c == -1:
            print("\nCluster = -1 (Noise)")
            continue

        print("\n" + "="*50)
        print(f"Cluster {c} Profile")
        print("="*50)

        cluster_df = df[df['Cluster'] == c]

        for feat in important:
            c_mean = cluster_df[feat].mean()
            overall_mean = df[feat].mean()
            diff_pct = ((c_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0

            direction = "higher" if diff_pct > 0 else "lower"
            print(f"- {feat}: {c_mean:.2f} ({abs(diff_pct):.1f}% {direction} than overall)")


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

def compare_clustering_models(data, n_clusters=3, eps=0.5, min_samples=5, explain=True):
    results = {}

    # KMeans
    km, km_labels = run_kmeans(data, n_clusters, explain=False)
    results["KMeans"] = (km, km_labels)

    # Hierarchical
    hc, hc_labels = run_hierarchical(data, n_clusters, explain=False)
    results["Hierarchical"] = (hc, hc_labels)

    # GMM
    gmm, gmm_labels = run_gmm(data, n_clusters, explain=False)
    results["GMM"] = (gmm, gmm_labels)

    # DBSCAN
    db, db_labels = run_dbscan(data, eps=eps, min_samples=min_samples, explain=False)
    results["DBSCAN"] = (db, db_labels)

    scores = {}
    for name, (model, labels) in results.items():
        valid = labels != -1
        if name == "DBSCAN" and (len(np.unique(labels[valid])) < 2):
            scores[name] = None
            continue

        try:
            score = silhouette_score(data[valid], labels[valid])
            scores[name] = score
        except:
            scores[name] = None

    if explain:
        print("\nSilhouette Score Comparison:")
        for name, score in scores.items():
            print(f"{name}: {score}")

    return scores
