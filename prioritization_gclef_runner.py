import pandas as pd
from prioritization.prioritization_manager import run_gclef_prioritization

#projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
#from_version = [1, 1, 1, 1, 1]
#to_version = [26, 133, 65, 106, 26]

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [1, 1, 1, 1, 1]

for index, project in enumerate(projects):
#    bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
#    bug_prediction_data_path = 'bug_prediction_data/xgboost_res2/' + project + '.csv'
#    bug_prediction_data_path = 'bug_prediction_data/xgboost_res1/' + project + '.csv'
#    bug_prediction_data_path = 'bug_prediction_data/cb_results_14001109_online/' + project + '.csv'
    bug_prediction_data_path = 'bug_prediction_data/xgb_results_14001115_online/' + project + '.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_gclef_prioritization(bug_prediction_data, 'xgb_score_online', project, version_number, 'gclef_test.csv',
                                         'xgb_results_14001115_online')
        print()
