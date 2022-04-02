import pandas as pd
from prioritization.prioritization_manager import run_max_prioritization_fp

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [26, 133, 65, 106, 26]

for index, project in enumerate(projects):
    bug_prediction_data_path = 'defect_prediction_results/xgb_results_online/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_max_prioritization_fp(bug_prediction_data, 'xgb_score_online', project, version_number, 0.999,
                                  'max_xgb_online.csv', 'max_xgb_online')
        print()
