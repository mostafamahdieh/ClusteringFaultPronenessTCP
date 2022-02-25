import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as algorithm_stats
import itertools as it


def algorithm_statistics(projects, data_path, results_path, algorithms, alg_complete_names, to_version, cluster_nums, cluster_nums_fp):
    algorithm_stats = []
    first_fail = pd.read_csv(data_path + "/first_fail_all.csv")
    apfd = pd.read_csv(data_path + "/apfd_all.csv")
    vals = first_fail

    for (index, project) in enumerate(projects):
        print("Project ", project)
        vals_proj = vals.where(vals.project == project).where(vals.version <= to_version[index])
        vals_proj_mean = vals_proj.mean()
        plt.close('all')

        stat = [project]
        for algorithm in algorithms:
            if 'clus' in algorithm:
                if 'xgb' in algorithm:
                    algorithm = algorithm.replace('clus', 'clus'+str(cluster_nums_fp[index]))
                else:
                    algorithm = algorithm.replace('clus', 'clus'+str(cluster_nums[index]))

            stat.append(vals_proj_mean[algorithm])

        print(stat)
        algorithm_stats.append(stat)

    algorithm_stats_frame = pd.DataFrame(columns=['Project'] + alg_complete_names, data=algorithm_stats)
    algorithm_stats_frame.to_csv('%s/AlgorithmStatistics.csv' % results_path)
