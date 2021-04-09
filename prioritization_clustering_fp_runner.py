import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp
from prioritization.prioritization_clustering import clustering_agg13

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
        for distance_metric in ['jaccard', 'matching', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath']:
            run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_agg13, distance_metric, 200,
                                         [0.999], 'agg13_200.csv',
                                         ['a13_%s' % distance_metric])
        print()
    index = index + 1
