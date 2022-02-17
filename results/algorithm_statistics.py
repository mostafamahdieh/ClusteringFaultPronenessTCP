import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as algorithm_stats
import itertools as it

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
to_version = [21, 128, 60, 101, 22]
#to_version = [13, 50, 33, 50, 14]

data_path = '../aggregate/main_results'
results_path = '../aggregate/main_results'
matplotlib.rcParams.update({'font.size': 14})
pd.set_option('display.max_columns', 1000)

first_fail = pd.read_csv(data_path + "/first_fail_all.csv")
apfd = pd.read_csv(data_path + "/apfd_all.csv")

vals = first_fail
cluster_nums = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
algorithms = ['tot_c0', 'add_c0', 'tot_std_xgb_results_14001115_online_c0999', 'add_std_xgb_results_14001115_online_c0999', 'gclef_tot_gclef_xgb_results_14001115_online', 'gclef_add_gclef_xgb_results_14001115_online', 'eucl_fp0_clus200_at', 'eucl_xgb_results_14001115_online_c0999_clus200', 'eucl_xgb_results_14001115_online_c0999_tt_clus200']
alg_complete_names = ['Total', 'Add', 'Total+FP', 'Add+FP', 'GclefTot', 'GclefAdd', 'CovClustering', 'CovClustering+FP', 'CovClusteringTT+FP']

algorithm_stats = []

for (index, project) in enumerate(projects):
    print("Project ", project)
    vals_proj = vals.where(vals.project == project).where(vals.version <= to_version[index])
    vals_proj_mean = vals_proj.mean()
    plt.close('all')

    stat = [project]
    for algorithm in algorithms:
        stat.append(vals_proj_mean[algorithm])

    print(stat)
    algorithm_stats.append(stat)

algorithm_stats_frame = pd.DataFrame(columns=['Project']+alg_complete_names, data=algorithm_stats)
algorithm_stats_frame.to_csv('%s/AlgorithmStatistics.csv' % results_path)
