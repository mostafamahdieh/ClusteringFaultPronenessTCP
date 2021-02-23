import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering_fp
from prioritization import prioritization_clustering as pr_cl

def clustering_method(coverage, cluster_num):
    return pr_cl.clustering_agg(coverage, 'complete', cluster_num)

def main():
    projects = ['Time', 'Chart', 'Math', 'Lang', 'Closure']
    from_version = [1, 1, 1, 1, 1]
    to_version = [14, 13, 50, 33, 50]


    index = 0

    for project in projects:
        print("Stargin project " + project)
        bug_prediction_data_path = '../WTP-data/' + project + '/xgb.csv'
        print("Reading " + bug_prediction_data_path)
        bug_prediction_data = pd.read_csv(bug_prediction_data_path)
        print("done.")
        for version_number in range(from_version[index], to_version[index] + 1):
            print("* Version %d" % version_number)
            run_prioritization_clustering_fp(bug_prediction_data, clustering_method, project, version_number, 200, [0, 0.99, 0.999, 1],
                'aggcomp.csv', ['aggcomp200_c0', 'aggcomp200_c099', 'aggcomp200_c0999','aggcomp200_c1full' ])
            print()
        index = index + 1

main()
