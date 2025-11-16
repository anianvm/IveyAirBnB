import pandas as pd
import clustering_workbook as cw

# TODO 1 Load the data
# Assuming you have a cleaned dataset
# data = pd.read_csv('you dataset')
# TODO 2 Transform data
# Assuming the dataset doesn't have any index columns or columns with unique IDs.
#transformed_data = cw.preprocess_data_for_clustering(data)

# TODO 3 Find optimal number of clusters using multiple methods
# k= cw.find_optimal_k(transformed_data, max_k=15, method='all')
# TODO 3.1 CAUTION: Verify that the suggested k value is appropriate for your specific use case.
#   If it doesn't make sense, manually overwrite it with a different value.
# k = ...

# TODO 4 Run a specific clustering model
# k-means clustering
# kmeans_model, k_cluster_labels = cw.run_kmeans(transformed_data, n_clusters=k)

# # Hierarchical clustering
# hierarchical_model, h_cluster_labels = cw.run_hierarchical(transformed_data, n_clusters=k, linkage_method='ward')
# cw.plot_dendrogram (transformed_data)

# # DBSCAN clustering
# eps = cw.find_optimal_eps(transformed_data) # for DBSCAN only
# dbscan_model, d_cluster_labels = cw.run_dbscan(transformed_data, eps=eps, min_samples=5)

# # Gaussian Mixture Model
# gmm_model, gm_cluster_labels = cw.run_gmm(transformed_data, n_components=k)

# TODO 5 EXplain the clustering results from a specific model
# cluster_labels=...
# # cw.explain_clusters(data, cluster_labels, 10)
# # cw.plot_cluster_heatmap(data, cluster_labels)
# # cw.plot_radar_chart(data, cluster_labels)

# TODO 6 Compare multiple clustering models with a fixed k
# clustering_results = cw.compare_clustering_models(transformed_data, n_clusters=k, eps=eps, min_samples=5)