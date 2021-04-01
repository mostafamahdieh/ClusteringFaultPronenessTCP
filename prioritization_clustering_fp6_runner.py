import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp6

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
        run_prioritization_clustering_fp6(bug_prediction_data, project, version_number, 200, [0, 0.99, 0.999, 1],
                                          'clfp6_agg200.csv', ['clfp6_agg200_c0', 'clfp6_agg200_c099', 'clfp6_agg200_c0999','clfp6_agg200_c1' ])
        print()
    index = index + 1
