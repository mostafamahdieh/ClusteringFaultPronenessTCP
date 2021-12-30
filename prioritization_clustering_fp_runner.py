import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp
from prioritization.prioritization_clustering import clustering_agg11
from prioritization.prioritization_clustering import clustering_agg12

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [13, 50, 33, 50, 14]

for index, project in enumerate(projects):
#    bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
    bug_prediction_data_path = 'bug_prediction_data/xgboost_res1/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_agg11, 200,
                                         [0.95], 'agg11_res1_c95.csv',
                                         ['a11_res1_c95'])
        print()
