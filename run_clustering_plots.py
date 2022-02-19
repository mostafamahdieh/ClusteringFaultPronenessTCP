import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import Normalizer
from prioritization.prioritization_clustering import clustering_agg1, clustering_agg2
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances

from results.clustering_plots import plot_clustering_num

cluster_nums = [2]+list(range(5,1001,5))
projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [1, 1, 1, 1, 1]


for index, project in enumerate(projects):
    bug_prediction_data_path = 'defect_prediction_results/xgb_results_14001115_online/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)

        plot_clustering_num(bug_prediction_data, 'xgb_score_online', project, version_number, clustering_agg2, euclidean_distances, 'average', cluster_nums, [0, 0.999],
                ['eucl_agg2_no_fp', 'eucl_agg2_no_fp'])
