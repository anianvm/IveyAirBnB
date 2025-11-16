import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from kneed import KneeLocator
from collections import Counter

pd.set_option("display.max_columns", None)

def preprocess_data_for_clustering(raw_data):
    data = raw_data.copy()

    categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    binary_features = data.select_dtypes(include=['bool']).columns.tolist()

    print("Categorical Features:", categorical_features)
    print("Numerical Features:", numerical_features)
    print("Binary Features:", binary_features)

    preprocess_pipeline = Pipeline([
        ('numerical_imputer', MeanMedianImputer(
            imputation_method='median',
            variables=numerical_features
        )),
        # Handle outliers
        ('outlier_handler', Winsorizer(
            capping_method='gaussian',
            tail='both',
            fold=3,
            variables=numerical_features
        )),
        # Normalize numeric data
        ('scaler', SklearnTransformerWrapper(
            transformer=StandardScaler(),
            variables=numerical_features
        ))
    ])

    if categorical_features:
        preprocess_pipeline.steps.insert(1, ('categorical_imputer',
                                             CategoricalImputer(
                                                 imputation_method='frequent',
                                                 variables=categorical_features
                                             )
                                             ))

        preprocess_pipeline.steps.append(('encoder',
                                          OneHotEncoder(
                                              variables=categorical_features,
                                              drop_last=True
                                          )
                                          ))

    processed_data = preprocess_pipeline.fit_transform(data)

    print(f"Original data shape: {data.shape}")
    print(f"Processed data shape: {data.shape}")

    return processed_data

def find_optimal_k(data, max_k=15, method='all'):
    k_range = range(2, max_k + 1)
    inertia_values = []
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    for k in k_range:
        print(f"Evaluating k={k}...")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)

        inertia_values.append(kmeans.inertia_)

        if method in ['silhouette', 'all']:
            if k > 1:
                sil_score = silhouette_score(data, kmeans.labels_)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)

        if method in ['all']:
            if k > 1:
                ch_score = calinski_harabasz_score(data, kmeans.labels_)
                ch_scores.append(ch_score)
            else:
                ch_scores.append(0)

            db_score = davies_bouldin_score(data, kmeans.labels_)
            db_scores.append(db_score)

    plt.figure(figsize=(18, 10))

    if method in ['elbow', 'all']:
        plt.subplot(2, 2, 1)
        plt.plot(k_range, inertia_values, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid(True)

        try:
            kl = KneeLocator(k_range, inertia_values, curve='convex', direction='decreasing')
            elbow_k = kl.elbow
            plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow at k={elbow_k}')
            plt.legend()
        except:
            print("Could not automatically determine elbow point")

    if method in ['silhouette', 'all']:
        plt.subplot(2, 2, 2)
        plt.plot(k_range, silhouette_scores, 'go-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')
        plt.grid(True)

        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        plt.axvline(x=best_silhouette_k, color='r', linestyle='--',
                    label=f'Best k={best_silhouette_k}')
        plt.legend()

    if method in ['all']:
        plt.subplot(2, 2, 3)
        plt.plot(k_range, ch_scores, 'mo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Method')
        plt.grid(True)

        best_ch_k = k_range[np.argmax(ch_scores)]
        plt.axvline(x=best_ch_k, color='r', linestyle='--',
                    label=f'Best k={best_ch_k}')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(k_range, db_scores, 'co-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Davies-Bouldin Score')
        plt.title('Davies-Bouldin Method (Lower is Better)')
        plt.grid(True)

        best_db_k = k_range[np.argmin(db_scores)]
        plt.axvline(x=best_db_k, color='r', linestyle='--',
                    label=f'Best k={best_db_k}')
        plt.legend()

    plt.tight_layout()
    plt.show()

    if method == 'elbow':
        try:
            return elbow_k
        except:
            return k_range[1]  # Default to 3 if elbow can't be found
    elif method == 'silhouette':
        return best_silhouette_k
    else:  # 'all'
        votes = [best_silhouette_k, best_ch_k, best_db_k]
        votes.append(elbow_k)

        counter = Counter(votes)
        most_common = counter.most_common(1)[0][0]

        print("\nRecommended number of clusters by different methods:")
        print(f"Silhouette Method: {best_silhouette_k}")
        print(f"Calinski-Harabasz Method: {best_ch_k}")
        print(f"Davies-Bouldin Method: {best_db_k}")
        try:
            print(f"Elbow Method: {elbow_k}")
        except:
            print("Elbow Method: Could not determine")

        print(f"\nFinal recommendation: {most_common} clusters")
        return most_common

def find_optimal_eps(data, min_samples=5): # for DBSCAN
    neigh = NearestNeighbors(n_neighbors=min_samples)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)

    k_dist = np.sort(distances[:, min_samples - 1])

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(k_dist)), k_dist)
    plt.xlabel('Points sorted by distance to kth neighbor')
    plt.ylabel(f'Distance to {min_samples}th nearest neighbor')
    plt.title(f'K-Distance Graph (k={min_samples})')
    plt.grid(True)

    x = np.arange(len(k_dist))
    kl = KneeLocator(x, k_dist, curve='convex', direction='increasing')
    knee_point = kl.knee

    if knee_point:
        eps = k_dist[knee_point]
        plt.axvline(x=knee_point, color='r', linestyle='--',
                        label=f'Knee at eps≈{eps:.3f}')
        plt.axhline(y=eps, color='r', linestyle='--')
        plt.legend()

        print(f"Recommended eps value: {eps:.3f}")
        return eps

    plt.show()
    print("Could not automatically determine optimal eps.")
    print("Please inspect the k-distance plot and select a value at the 'elbow' point.")
    eps = float(input("Enter your chosen eps value: "))
    return eps

def run_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    sil_score = silhouette_score(data, labels)
    print(f"K-means with {n_clusters} clusters:")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Inertia: {kmeans.inertia_:.3f}")

    unique, counts = np.unique(labels, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Cluster {i}: {count} samples ({count / len(labels):.1%})")

    return kmeans, labels

def run_hierarchical(data, n_clusters=3, linkage_method='ward'):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(data)

    sil_score = silhouette_score(data, labels)
    print(f"Hierarchical clustering with {n_clusters} clusters (linkage: {linkage_method}):")
    print(f"Silhouette Score: {sil_score:.3f}")

    unique, counts = np.unique(labels, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Cluster {i}: {count} samples ({count / len(labels):.1%})")

    return model, labels

def plot_dendrogram(data, max_samples=100): # for hierarchical clustering
    if len(data) > max_samples:
        sample_idx = np.random.choice(len(data), max_samples, replace=False)
        data_sample = data.iloc[sample_idx]
    else:
        data_sample = data

    linked = linkage(data_sample, method='ward')

    plt.figure(figsize=(16, 8))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

def run_dbscan(data, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    unique, counts = np.unique(labels, return_counts=True)

    if len(unique) > 1:
        valid_points = labels != -1
        if sum(valid_points) > 1:
            non_noise_labels = labels[valid_points]
            unique_non_noise = np.unique(non_noise_labels)
            if len(unique_non_noise) >= 2:
                sil_score = silhouette_score(data.iloc[valid_points], non_noise_labels)
                print(f"DBSCAN (eps={eps}, min_samples={min_samples}):")
                print(f"Silhouette Score (excluding noise): {sil_score:.3f}")
            else:
                print(f"DBSCAN (eps={eps}, min_samples={min_samples}):")
                print("Cannot calculate silhouette score: only one non-noise cluster found")
        else:
            print(f"DBSCAN (eps={eps}, min_samples={min_samples}):")
            print("Cannot calculate silhouette score: not enough non-noise points")

    for i, count in zip(unique, counts):
        if i == -1:
            print(f"Noise: {count} samples ({count / len(labels):.1%})")
        else:
            print(f"Cluster {i}: {count} samples ({count / len(labels):.1%})")

    return model, labels

def run_gmm(data, n_components=3):
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(data)
    labels = model.predict(data)

    sil_score = silhouette_score(data, labels)
    print(f"Gaussian Mixture Model with {n_components} components:")
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"BIC: {model.bic(data):.3f}")
    print(f"AIC: {model.aic(data):.3f}")

    unique, counts = np.unique(labels, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"Cluster {i}: {count} samples ({count / len(labels):.1%})")

    return model, labels

def get_feature_importance(data, labels, top_n=10):
    f_values, p_values = f_classif(data, labels)

    importance_df = pd.DataFrame({
        'Feature': data.columns,
        'F-Value': f_values,
        'P-Value': p_values
    }).sort_values('F-Value', ascending=False)
    return importance_df

def plot_cluster_heatmap(data, labels, top_n=10):
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = labels

    cluster_centers = data_with_labels.groupby('Cluster').mean()

    importance_df = get_feature_importance(data, labels, top_n=top_n)

    top_features = importance_df.head(top_n)['Feature'].tolist()

    plt.figure(figsize=(14, 10))

    sns.heatmap(cluster_centers[top_features], cmap='viridis', annot=True, fmt='.2f',
                linewidths=0.5, cbar_kws={'label': 'Normalized Value'})

    plt.title(f'Cluster Profiles: Top Differentiating Features')
    plt.ylabel('Cluster')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_radar_chart(data, labels, top_n=6):
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = labels
    importance_df = get_feature_importance(data, labels, top_n=top_n)

    top_features = importance_df.head(top_n)['Feature'].tolist()

    cluster_centers = data_with_labels.groupby('Cluster').mean()

    scaler = MinMaxScaler()
    scaled_centers = pd.DataFrame(
        scaler.fit_transform(cluster_centers[top_features]),
        index=cluster_centers.index,
        columns=top_features
    )

    n_clusters = len(np.unique(labels))
    if -1 in np.unique(labels):
        n_clusters -= 1

    fig = plt.figure(figsize=(15, 10))

    if n_clusters <= 3:
        n_rows, n_cols = 1, n_clusters
    else:
        n_rows = (n_clusters + 2) // 3
        n_cols = min(n_clusters, 3)

    for i, cluster in enumerate(sorted(np.unique(labels))):
        if cluster == -1:
            continue

        if i >= n_rows * n_cols:
            break
        ax = fig.add_subplot(n_rows, n_cols, i + 1, polar=True)

        values = scaled_centers.loc[cluster].values

        N = len(top_features)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        values = np.append(values, values[0])  # Close the loop

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_features, fontsize=8)
        ax.set_title(f'Cluster {cluster} Profile', size=11, y=1.1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def explain_clusters(data, labels, top_n=10):
    data_with_labels = data.copy()
    data_with_labels['Cluster'] = labels
    cluster_sizes = data_with_labels['Cluster'].value_counts().sort_index()
    print("Cluster Sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} samples ({size / len(labels):.1%})")

    importance_df = get_feature_importance(data, labels, top_n=top_n)

    top_features = importance_df.head(top_n)['Feature'].tolist()
    for cluster in sorted(np.unique(labels)):
        if cluster == -1:
            continue
        print(f"\n{'=' * 50}")
        print(f"Cluster {cluster} Profile:")
        print(f"{'=' * 50}")

        cluster_samples = data_with_labels[data_with_labels['Cluster'] == cluster]

        for feature in top_features:
            if feature in data.columns:
                cluster_mean = cluster_samples[feature].mean()
                overall_mean = data[feature].mean()
                diff_pct = ((cluster_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else float('inf')
                direction = "higher" if diff_pct > 0 else "lower"
                print(f"- {feature}: {cluster_mean:.2f} ({abs(diff_pct):.1f}% {direction} than average)")

    print(f"\n{'=' * 50}")
    print("Visualizing Cluster Profiles")
    print(f"{'=' * 50}")

    selected_features = [f for f in top_features if f in data.columns]
    selected_data = data_with_labels[selected_features + ['Cluster']]

    cluster_means = selected_data.groupby('Cluster').mean()

    if 'Cluster' in cluster_means.columns:
        cluster_means = cluster_means.drop('Cluster', axis=1)

    cluster_means_transposed = cluster_means.T

    fig, ax = plt.subplots(figsize=(16, 10))
    cluster_means_transposed.plot(kind='bar', ax=ax, width=0.8)

    plt.title('Mean Feature Values by Cluster', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Mean Value', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Cluster', title_fontsize=12, fontsize=10, loc='best')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=3)

    plt.tight_layout()

    plt.show()

def compare_clustering_models(data, n_clusters=3, eps=0.5, min_samples=5):
    results = {}

    print("Running K-means clustering...")
    kmeans, kmeans_labels = run_kmeans(data, n_clusters)
    results['K-means'] = (kmeans, kmeans_labels)

    print("\nRunning Hierarchical clustering...")
    hc, hc_labels = run_hierarchical(data, n_clusters)
    results['Hierarchical'] = (hc, hc_labels)

    print("\nRunning DBSCAN clustering...")
    dbscan, dbscan_labels = run_dbscan(data, eps, min_samples)
    results['DBSCAN'] = (dbscan, dbscan_labels)

    print("\nRunning Gaussian Mixture Model clustering...")
    gmm, gmm_labels = run_gmm(data, n_clusters)
    results['GMM'] = (gmm, gmm_labels)

    plt.figure(figsize=(12, 6))
    scores = []
    methods = []

    print("\nSilhouette Scores:")
    for name, (model, labels) in results.items():
        if name == 'DBSCAN' and all(l == -1 for l in labels):
            print(f"{name}: N/A (all points classified as noise)")
            continue

        if name == 'DBSCAN':
            valid_points = labels != -1
            if sum(valid_points) > 1:
                # Count non-noise clusters
                non_noise_labels = labels[valid_points]
                unique_non_noise = np.unique(non_noise_labels)

                # Only calculate silhouette if we have at least 2 clusters after removing noise
                if len(unique_non_noise) >= 2:
                    score = silhouette_score(data.iloc[valid_points], non_noise_labels)
                    print(f"{name}: {score:.3f}")
                    scores.append(score)
                    methods.append(name)
                else:
                    print(f"{name}: N/A (only one non-noise cluster found)")
            else:
                print(f"{name}: N/A (not enough non-noise points)")
        else:
            # For other algorithms, check if we have at least 2 clusters
            unique_labels = np.unique(labels)
            if len(unique_labels) >= 2:
                score = silhouette_score(data, labels)
                print(f"{name}: {score:.3f}")
                scores.append(score)
                methods.append(name)
            else:
                print(f"{name}: N/A (fewer than 2 clusters found)")
    return results
