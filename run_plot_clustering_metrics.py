import pandas as pd
from matplotlib import pyplot as plt

from results.clustering_metrics import plot_clustering_metrics, plot_clustering_metrics_multi
from results.compare_clustering_results import get_project_tcp_results

metrics_cluster_nums_filter = list(range(25, 501, 25))
projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [1, 1, 1, 1, 1]

tcp_to_versions = [21, 128, 60, 101, 22]

data_path = './aggregate/main_results'
output_path = './clustering_metrics/plots/4'
metrics_path = './clustering_metrics/4'

dist_functions = ['eucl']
dist_complete_names = ['Euclidean', 'Cosine']
# base_alg_name = '_agg2_xgb14001115_online_c0999_max_clus'

# metrics = ['silhouette', 'calinski']
# metric_labels = ['Silhouette score', 'Calinski-Harabasz score', 'Davies-Bouldin score']
# metrics = ['silhouette', 'calinski', 'davies']

metric_labels = ['Davies-Bouldin index']
metrics = ['davies']

for index, project in enumerate(projects):
    for version_number in range(from_version[index], to_version[index] + 1):
        print("Project: ", project, " Version: ", version_number)

        plot_clustering_metrics_multi(data_path, metrics_path, output_path, project, version_number, '_fp0_clus', "_at",
                                      metrics, metric_labels, metrics_cluster_nums_filter,
                                      list(range(50, 501, 50)), dist_functions, dist_complete_names,
                                      tcp_to_versions[index], False, 'metrics_at_nofp')

        plot_clustering_metrics_multi(data_path, metrics_path, output_path, project, version_number,
                                      '_agg2_xgb14001115_online_c0999_max_clus', "", metrics, metric_labels,
                                      metrics_cluster_nums_filter,
                                      list(range(25, 501, 25)), dist_functions, dist_complete_names,
                                      tcp_to_versions[index], True, 'metrics_max_fp')
