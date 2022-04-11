import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools as it


def compare_clustering_results(projects, to_versions, data_path, output_path, cluster_nums, dist_functions,
                               dist_complete_names, base_alg_name, after_name, output_name):
    matplotlib.rcParams.update({'font.size': 12})
    pd.set_option('display.max_columns', 1000)

    first_fail = pd.read_csv(data_path + "/first_fail_all.csv")
    apfd = pd.read_csv(data_path + "/apfd_all.csv")

    vals = first_fail
    all_proj_stats = pd.DataFrame(columns=['Cluster #'], data=cluster_nums)

    marker_style = ['o', 'X', '^']

    for (index, project) in enumerate(projects):
        print("Project ", project)
        proj_stats, proj_stats_data = get_project_tcp_results(base_alg_name, after_name, cluster_nums, dist_functions,
                                                              project, to_versions[index], vals)
        all_proj_stats = all_proj_stats.join(pd.DataFrame(columns=[project], data=[x[1] for x in proj_stats_data]))
        plt.close('all')
        fig1 = plt.figure()
        min_tcp = []
        max_tcp = []
        for (ind_dist, dist) in enumerate(dist_functions):
            plt.plot(proj_stats["Cluster #"], proj_stats[dist], linestyle='--', marker=marker_style[ind_dist], label=dist_complete_names[ind_dist])
            min_tcp.append(min(proj_stats[dist]))
            max_tcp.append(max(proj_stats[dist]))

        plt.legend()
        # plt.legend(title='Distance function')
        # plt.ylabel('APFD (%)')
        plt.ylabel('First fail (%)')
        #plt.xlim(10, max(cluster_nums))
        plt.ylim(min(min_tcp)-3, max(max_tcp)+3)
        plt.xticks([1] + list(range(50, 501, 50)))

        plt.grid(axis='x', color='0.95')
        plt.suptitle(project)

        fig1.savefig('%s/%s_%s.png' % (output_path, project, output_name))
        plt.close('all')

        # print(proj_stats)
        print()

    all_proj_stats.to_csv('%s/ClusteringStatistics.csv' % output_path)


def get_project_tcp_results(base_alg_name, after_name, cluster_nums, dist_functions, project, to_version, vals):
    vals_proj = vals.where(vals.project == project).where(vals.version <= to_version)
    vals_proj_mean = vals_proj.mean()
    proj_stats_data = []
    for cluster_num in cluster_nums:
        stat = [cluster_num]

        print("cluster_num: ", cluster_num)

        if cluster_num == 1:
            print("len(dist_functions): ", len(dist_functions))
            if base_alg_name in ["_fp0_clus"]:
                stat = stat + [vals_proj_mean["add_c0"]] * len(dist_functions)
            elif base_alg_name in ["_xgb_results_14001115_online_c0_tt_clus"]:
                stat = stat + [vals_proj_mean["tot_c0"]] * len(dist_functions)
            elif base_alg_name in ["_agg2_xgb14001115_online_c0999_max_clus"]:
                stat = stat + [vals_proj_mean["eucl_agg2_xgb14001115_online_c0999_max_clus1"]] * len(dist_functions)
        else:
            for dist in dist_functions:
                # print("dist: ", dist)
                # print("base_alg_name: ", base_alg_name)
                # print("cluster_num: ", cluster_num)
                # print("after_name: ", after_name)
                stat_index_str = dist + base_alg_name + str(cluster_num) + after_name
                # print("stat_index_str: ", stat_index_str)
                stat.append(vals_proj_mean[stat_index_str])

        print(stat)
        proj_stats_data.append(stat)
    proj_stats = pd.DataFrame(columns=['Cluster #'] + dist_functions, data=proj_stats_data)
    return proj_stats, proj_stats_data
