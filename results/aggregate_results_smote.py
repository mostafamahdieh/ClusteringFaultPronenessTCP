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


def main():
#    projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
#    from_version = [1, 1, 1, 1, 1]
#    to_version = [26, 119, 65, 106, 26]
    projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
    from_version = [1, 1, 1, 1, 1]
    to_version = [13, 50, 33, 50, 14]

    results_path = '../../WTP-data/aggregate'
    matplotlib.rcParams.update({'font.size': 14})
    pd.set_option('display.max_columns', 1000)

    data_vals_stats = pd.DataFrame()
    improvement_stats = pd.DataFrame(columns=["project", "improvement_clustering", "improvement_clustering_fp"])
    first_fail_all = pd.DataFrame()

    for index, project in enumerate(projects):
        first_fail, apfd = read_results(['std2_smote.csv', 'agg11_smote_200.csv'],
                                        project, from_version[index], to_version[index] + 1)

        plt.close('all')

        # 'fp2_1__1_aa': 'fp_0_aa' --> c_dp1=1.0, c_dp2=0

        first_fail = first_fail.rename(columns={"a11_c0_200_tt":"Clustering","a11_c0999_200_tt":"Clustering+FP",
                                                "add_smote_c0":"Additional","tot_smote_c0":"Total",
                                                "add_smote_c0999":'Additional+FP', "tot_smote_c0999":"Total+FP"})

#        print(first_fail)
#        print(apfd)

#        improvement_clustering = improvement(first_fail[['Additional', 'Total']].min(axis=1), first_fail["Clustering"])
#        improvement_clustering_fp = improvement(first_fail[['Additional+FP', 'Total+FP']].min(axis=1), first_fail["Clustering+FP"])
#        print("improvement_clustering", improvement_clustering)
#        print("improvement_clustering_fp", improvement_clustering_fp)
#        improvement_stats.add([project, improvement_clustering, improvement_clustering_fp])

        if index == 0:
            first_fail_all = first_fail
        else:
            first_fail_all = first_fail_all.append(first_fail)
        first_fail_mean = first_fail.mean()
        first_fail_mean = first_fail_mean.drop('version')

        data_vals_stats = data_vals_stats.append(first_fail_mean, ignore_index=True)

#        columns = ['Total', 'Additional', 'Total+FP', 'Additional+FP', 'Clustering', 'Clustering+FP']
        columns = ['Total', 'Additional', 'Total+FP', 'Additional+FP']

        plot1 = first_fail.boxplot(column=columns)
        plot1.set_ylabel('First Fail (%)')
        plot1.set_ylim(0, 100)

        #plot1.set_title(project)

        fig1 = plot1.get_figure()
        fig1.autofmt_xdate(rotation=32)
        #fig1.savefig('%s/first_fail/%s.first_fail.boxplot.png' % (results_path, project), bbox_inches='tight')
        fig1.savefig('%s/first_fail/%s.first_fail.boxplot.png' % (results_path, project))

        plt.close('all')

    first_fail_all = first_fail_all.reset_index()
    print(first_fail_all)
#    print("first_fail_total", stats.wilcoxon(first_fail_all["Total"],first_fail_all["Clustering"]))
#    print("first_fail_additional", stats.wilcoxon(first_fail_all["Additional"],first_fail_all["Clustering"]))
#    print("first_fail_tolal+fp", stats.wilcoxon(first_fail_all["Total+FP"],first_fail_all["Clustering+FP"]))
#    print("first_fail_additional+fp", stats.wilcoxon(first_fail_all["Additional+FP"],first_fail_all["Clustering+FP"]))

    data_vals_stats.insert(0, 'project', projects)
    data_vals_stats.to_csv(results_path+'/smote/stats.csv')

main()
