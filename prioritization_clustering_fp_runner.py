import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp
from prioritization.prioritization_clustering import clustering_agg1

projects = ['Time', 'Chart', 'Math', 'Lang', 'Closure']
from_version = [1, 1, 1, 1, 1]
to_version = [14, 13, 50, 33, 50]


index = 0

for project in projects:
    bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_agg1, 200,
                                         [0, 0.999], 'agg1_200.csv',
                                         ['agg1_200_c0', 'agg1_200_c0999'])
        print()
    index = index + 1
