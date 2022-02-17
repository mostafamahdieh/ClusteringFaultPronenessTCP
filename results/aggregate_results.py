import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import scipy.stats as stats
import itertools as it

def aggregate_results(file_names, projects, from_version, to_version, results_path):
    try:
        os.stat(results_path)
    except:
        os.mkdir(results_path)

    first_fail = pd.DataFrame(columns=['project', 'version'])
    apfd = pd.DataFrame(columns=['project', 'version'])
    for index, project in enumerate(projects):
        for version_number in range(from_version[index], to_version[index] + 1):
            data_path = "../WTP-data/%s/%d" % (project, version_number)
            results_dict_first_fail = {'version': version_number, 'project': project}
            results_dict_apfd = {'version': version_number, 'project': project}
            skipped = False

            for file_name in file_names:
                file_path = '%s/%s' % (data_path, file_name)

                if os.path.isfile(file_path):
                    print("Reading %s" % file_path)
                    results = pd.read_csv(file_path, delimiter=',')

                    for i, row in results.iterrows():
                        alg = row['alg']

                        if alg in results_dict_first_fail:
                            print('Name clash occured for algorithm: ', alg)
                            assert(not alg in results_dict_first_fail)

                        if alg in results_dict_apfd:
                            print('Name clash occured for algorithm: ', alg)
                            assert(not alg in results_dict_apfd)

                        results_dict_first_fail[alg] = row['first_fail'] * 100
                        results_dict_apfd[alg] = row['apfd']
                else:
                    print("Skipping %s" % file_path)
                    skipped = True

            if not skipped:
                first_fail = first_fail.append(results_dict_first_fail, ignore_index=True)
                apfd = apfd.append(results_dict_apfd, ignore_index=True)

#    first_fail = first_fail.reset_index()
#    apfd = apfd.reset_index()

    first_fail.to_csv(results_path + '/first_fail_all.csv', index_label="index")
    apfd.to_csv(results_path + '/apfd_all.csv', index_label="index")
