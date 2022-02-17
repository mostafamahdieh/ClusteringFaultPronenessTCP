import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools as it


def compare_clustering_results(projects, to_version, data_path, output_path, cluster_nums, dist_functions, dist_complete_names, base_filename, after_name):
    matplotlib.rcParams.update({'font.size': 12})
    pd.set_option('display.max_columns', 1000)

    first_fail = pd.read_csv(data_path + "/first_fail_all.csv")
    apfd = pd.read_csv(data_path + "/apfd_all.csv")

    vals = first_fail
    all_proj_stats = pd.DataFrame(columns=['Cluster #'], data=cluster_nums)

    for (index, project) in enumerate(projects):
        print("Project ", project)
        vals_proj = vals.where(vals.project == project).where(vals.version <= to_version[index])
        vals_proj_mean = vals_proj.mean()
        plt.close('all')

        proj_stats_data = []
        for cluster_num in cluster_nums:
            stat = [cluster_num]
            for dist in dist_functions:
                stat.append(vals_proj_mean[dist + base_filename + str(cluster_num) + after_name])

            print(stat)
            proj_stats_data.append(stat)

        proj_stats = pd.DataFrame(columns=['Cluster #'] + dist_functions, data=proj_stats_data)
        all_proj_stats = all_proj_stats.join(pd.DataFrame(columns=[project], data=[x[1] for x in proj_stats_data]))
        fig1 = plt.figure()
        for (ind_dist, dist) in enumerate(dist_functions):
            plt.plot(proj_stats["Cluster #"], proj_stats[dist], 'o--', label=dist_complete_names[ind_dist])

        plt.legend()
        #plt.legend(title='Distance function')
        #plt.ylabel('APFD (%)')
        plt.ylabel('First fail (%)')
        plt.ylim(0, 100)
        plt.grid(axis='x', color='0.95')
        plt.suptitle(project)

        fig1.savefig('%s/%s_clustering.first_fail.png' % (output_path, project))
        plt.close('all')

        print(proj_stats)
        print()

    all_proj_stats.to_csv('%s/ClusteringStatistics.csv' % output_path)
