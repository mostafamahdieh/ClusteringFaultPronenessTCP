import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import Normalizer
from prioritization.prioritization_clustering import clustering_agg
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
#to_version = [26, 133, 65, 106, 27]
to_version = [26, 133, 65, 106, 26]


for index, project in enumerate(projects):
    bug_prediction_data_path = 'defect_prediction_results/xgb_results_14001115_online/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
#        run_prioritization_clustering_fp(bug_prediction_data, 'xgb_score_online', project, version_number, clustering_agg, cosine_distances, 'average', 'total', 
#                list(range(50,501,50)), [0, 0.999], 
#                'cosd_xgb_results_14001115_online_c0c0999_tt_50_500.csv', ['cosd_xgb_results_14001115_online_c0_tt', 'cosd_xgb_results_14001115_online_c0999_tt'])

#        run_prioritization_clustering_fp(bug_prediction_data, 'xgb_score_online', project, version_number, clustering_agg, manhattan_distances, 'average', 'total', 
#                list(range(50,501,50)), [0, 0.999], 
#                'manh_xgb_results_14001115_online_c0c0999_tt_50_500.csv', ['manh_xgb_results_14001115_online_c0_tt', 'manh_xgb_results_14001115_online_c0999_tt'])

        run_prioritization_clustering_fp(bug_prediction_data, 'xgb_score_online', project, version_number, clustering_agg, euclidean_distances, 'average', 'total', 
                list(range(50,1001,50)), [0], 
                'eucl_xgb_results_14001115_online_c0_tt_50_1000.csv', ['eucl_xgb_results_14001115_online_c0_tt'])
        print()