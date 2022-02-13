import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import scipy.stats as stats
import itertools as it


def effect_size(lst1, lst2):
    return improvement(lst1, lst2)


def improvement(x, y):
    n = len(x)
    a = 0
    for i in range(0, n):
#        print("x",x[i],"y",y[i],"x-y",x[i]-y[i])
        a = a + (x[i] - y[i])
    improvement = (a/n)
    return improvement


def read_results(file_names, project, from_version, to_version):
    first_fail = pd.DataFrame(columns=['version'])
    apfd = pd.DataFrame(columns=['version'])

    for version_number in range(from_version, to_version):
        data_path = "../../WTP-data/%s/%d" % (project, version_number)
        results_dict_first_fail = {'version': version_number}
        results_dict_apfd = {'version': version_number}
        skipped = False

        for file_name in file_names:
            file_path = '%s/%s' % (data_path, file_name)

            if os.path.isfile(file_path):
                print("Reading %s" % file_path)
                results = pd.read_csv(file_path, delimiter=',')

                for i, row in results.iterrows():
                    results_dict_first_fail[row['alg']] = row['first_fail'] * 100
                    results_dict_apfd[row['alg']] = row['apfd']
            else:
                print("Skipping %s" % file_path)
                skipped = True

        if not skipped:
            first_fail = first_fail.append(results_dict_first_fail, ignore_index=True)
            apfd = apfd.append(results_dict_apfd, ignore_index=True)

    return first_fail, apfd


def aggregate_results(file_names, results_folder):
    projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
    from_version = [1, 1, 1, 1, 1]
    to_version = [26, 133, 65, 106, 26]

    results_path = '../../WTP-data/aggregate/'+results_folder
    try:
        os.stat(results_path)
    except:
        os.mkdir(results_path)       

    data_vals_stats = pd.DataFrame()
    improvement_stats = pd.DataFrame(columns=["project", "improvement_clustering", "improvement_clustering_fp"])
    first_fail_all = pd.DataFrame()

    for index, project in enumerate(projects):
        first_fail, apfd = read_results(file_names,
                                        project, from_version[index], to_version[index] + 1)

        if index == 0:
            apfd_all = apfd
            first_fail_all = first_fail
        else:
            apfd_all = apfd_all.append(apfd)
            first_fail_all = first_fail_all.append(first_fail)

    first_fail_all = first_fail_all.reset_index()
    apfd_all = apfd_all.reset_index()

    first_fail_all.to_csv(results_path+'/first_fail_all.csv')