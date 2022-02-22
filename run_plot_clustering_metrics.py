import pandas as pd
from matplotlib import pyplot as plt

from results.clustering_metrics import plot_clustering_metrics
from results.compare_clustering_results import get_project_tcp_results

cluster_nums = [2]+list(range(25,501,5))
projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [1, 1, 1, 1, 1]

tcp_results_cluster_nums = [1]+list(range(50, 501, 50))
tcp_to_versions = [21, 128, 60, 101, 22]

data_path = './aggregate/main_results'
output_path = './aggregate/compare_clustering_agg2_max_with_fp'

dist_functions = ['eucl']
dist_complete_names = ['Euclidean', 'Cosine']
#base_alg_name = '_agg2_xgb14001115_online_c0999_max_clus'

metrics = ['silhouette', 'calinski']
metric_labels = ['Silhouette score', 'Calinski-Harabasz score', 'Davies-Bouldin score']
#  metrics = ['silhouette', 'calinski', 'davies']

for index, project in enumerate(projects):
    for version_number in range(from_version[index], to_version[index] + 1):
        print("Project: ", project, " Version: ", version_number)

        plot_clustering_metrics(data_path, project, version_number, '_fp0_clus', "_at", metrics, metric_labels, cluster_nums,
                                    tcp_results_cluster_nums, dist_functions, dist_complete_names, tcp_to_versions[index], False, 'at_nofp')

        plot_clustering_metrics(data_path, project, version_number, '_agg2_xgb14001115_online_c0999_max_clus', "", metrics, metric_labels, cluster_nums,
                                    tcp_results_cluster_nums, dist_functions, dist_complete_names, tcp_to_versions[index], True, 'max_fp')