import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools as it


def effect_size(lst1, lst2):
    return improvement(lst1, lst2)


def improvement(x, y):
    n = len(x)
    a = 0
    for i in range(0, n):
        a = a + x[i] / y[i]
    improvement = (a / n - 1) * 100
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


def main():
    projects = ['Time', 'Chart', 'Math', 'Lang', 'Closure']
    from_version = [1, 1, 1, 1, 1]
    to_version = [14, 13, 50, 33, 50]


    results_path = '../../WTP-data/aggregate'
    matplotlib.rcParams.update({'font.size': 14})

    pd.set_option('display.max_columns', 1000)

    data_vals_stats = pd.DataFrame()
    effect_size_vals = pd.DataFrame(columns=['project', 'additional', 'total'])



    for index, project in enumerate(projects):
        first_fail, apfd = read_results(['clfp5_agg200.csv', 'clfp5_agg200_c0.csv', 'clfp6_agg200.csv', 'aggcomp.csv', 'agg1_200.csv', 'agg2_200.csv', 'agg3_200.csv'],
                                        project, from_version[index], to_version[index] + 1)

        plt.close('all')

        # 'fp2_1__1_aa': 'fp_0_aa' --> c_dp1=1.0, c_dp2=0

        apfd = apfd.rename(columns={"cl11": "cl_tt", "cl12": "cl_ta", "cl21": "cl_at", "cl22": "cl_aa", "fp2_1__1_at" : "fp_at",
                             "fp2_1__1_aa": "fp_aa"})
        first_fail = first_fail.rename(columns={"cl11": "cl_tt", "cl12": "cl_ta", "cl21": "cl_at", "cl22": "cl_aa", "fp2_1__1_at" : "fp_at",
                             "fp2_1__1_aa": "fp_aa"})

#        print(first_fail)
#        print(apfd)

        first_fail_mean = first_fail.mean()
        first_fail_mean = first_fail_mean.drop('version')

        data_vals_stats = data_vals_stats.append(first_fail_mean, ignore_index=True)

        columns = ['tot_', 'tot_fp99', 'tot_fp999', 'tot_fp1', 'add_', 'add_fp99', 'add_fp999', 'add_fp1', "cl_tt",
                   "cl_ta", "cl_at", "cl_aa"]

#        plot1 = first_fail.boxplot(column=columns)
#        plot1.set_ylabel('First Fail (%)')
#        plot1.set_ylim(0, 100)

#        plot1.set_title(project)

#        fig1 = plot1.get_figure()
#        fig1.autofmt_xdate(rotation=90)
#        fig1.savefig('%s/first_fail/%s.first_fail.boxplot.png' % (results_path, project))

#        plt.close('all')

    data_vals_stats.insert(0, 'project', projects)
    data_vals_stats.to_csv(results_path+'/first_fail/stats.csv')

main()
