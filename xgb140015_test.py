import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import Normalizer
from prioritization.prioritization_clustering import clustering_agg
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances

projects = ['Chart']
#projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [26, 133, 65, 106, 26]
#to_version = [26, 133, 65, 106, 27]


for index, project in enumerate(projects):
#    bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
#    bug_prediction_data_path = 'defect_prediction_results/xgboost_res1/' + project + '.csv'
#    bug_prediction_data_path = 'bug_prediction_data/xgb_results_14001108_offline/' + project + '.csv'
#    bug_prediction_data_path = 'bug_prediction_data/cb_results_14001109_online/' + project + '.csv'
    bug_prediction_data_path = 'defect_prediction_results/xgb_results_14001115_online/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_prioritization_clustering_fp(bug_prediction_data, 'xgb_score_online', project, version_number, clustering_agg, euclidean_distances, 'average', ['total', 'additional'], [350],
                                         [0.999], 'eucl_xgb14001115_test_350.csv',
                                         ['eucl_xgb14001115_test_350_c999'])
        print()
