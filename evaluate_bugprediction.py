import numpy as np
import pandas as pd
import h5py


def extract_bug_prediction_for_units_version(bug_prediction_data, version, unit_names):
    unit_num = len(unit_names)
    unit_dp_prob = np.zeros((unit_num,))
    for u in range(0, unit_num):
        unit_class = unit_names[u]
        b_all = bug_prediction_data[bug_prediction_data.LongName == unit_class]
        b = b_all[b_all.version == version]
        if not b.empty:
            unit_dp_prob[u] = b.xgb_score.max()
    return unit_dp_prob


def run_bugprediction_evaluation(bug_prediction_data, project, version_number):
    data_path = "../WTP-data/%s/%d" % (project, version_number)


#    print("Reading real bugs...")
    faultClassFile = ("%s/bugfix_sources.txt" % data_path)
    with open(faultClassFile) as f:
        faultClasses = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    faultClasses = [x.strip() for x in faultClasses]

    unit_prob = extract_bug_prediction_for_units_version(bug_prediction_data, version_number, faultClasses)

    evaluation = 0
    evaluationProb = 0
    found = 0

    for index, faultClass in enumerate(faultClasses):
        found = found + 1
        prob = unit_prob[index]
#        print("%s --> %f" % (faultClass, prob))
        if (prob >= 0.3):
            evaluation = evaluation + 1
        evaluationProb = evaluationProb + prob

    if (found == 0):
        print("No classes found in bug prediction results")
        return

    evaluation = evaluation / found
    evaluationProb = evaluationProb / found
#    print("evaluation: ", evaluation)
    return (evaluation, evaluationProb)


evaluationSumProjects = []


projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [13, 50, 33, 50, 14]

for index, project in enumerate(projects):
    bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
    print("Reading " + bug_prediction_data_path)
    bug_prediction_data = pd.read_csv(bug_prediction_data_path)
    print("done.")
    evaluation_sum = 0
    for version_number in range(from_version[index], to_version[index] + 1):
#        print("* Version %d" % version_number)
        evaluation, evaluation_prob = run_bugprediction_evaluation(bug_prediction_data, project, version_number)
        evaluation_sum = evaluation_sum + evaluation
#        print("%d,%f,%f\n" % (version_number, evaluation, evaluation_prob))
    evaluationSumProjects.append((project, evaluation_sum, to_version[index]-from_version[index]+1))

print(evaluationSumProjects)
