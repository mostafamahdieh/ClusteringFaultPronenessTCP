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
#    bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
    bug_prediction_data_path = 'bug_prediction_data/xgboost_res1/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_agg, cosine_distances, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                                         [0], 'fp0_cosd_clus50_500.csv',
                                         ['cosd_fp0'])
#        run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_agg, manhattan_distances, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
#                                         [0], 'fp0_manh_clus50_500.csv',
#                                         ['manh_fp0'])
#        run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_agg, euclidean_distances, [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
#                                         [0], 'fp0_eucl_clus50_500.csv',
#                                         ['eucl_fp0'])
        print()
