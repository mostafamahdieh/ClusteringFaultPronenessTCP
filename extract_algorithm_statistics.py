import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as algorithm_stats
import itertools as it
from results.algorithm_statistics import algorithm_statistics

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
to_version = [21, 128, 60, 101, 22]
#to_version = [13, 50, 33, 50, 14]

data_path = './aggregate/main_results'
results_path = './aggregate/main_results'

algorithms = ['tot_c0', 'add_c0', 'tot_std_xgb_results_14001115_online_c0999', 'add_std_xgb_results_14001115_online_c0999',
              'gclef_tot_gclef2_xgb_results_14001115_online', 'gclef_add_gclef2_xgb_results_14001115_online',
              'eucl_fp0_clus100_at', 'eucl_fp0_clus200_at', 'eucl_xgb_results_14001115_online_c0_tt_clus100',
              'eucl_agg2_xgb14001115_online_c0999_tot_clus100', 'eucl_agg2_xgb14001115_online_c0999_max_clus100']
alg_complete_names = ['Total', 'Add', 'Total+FP', 'Add+FP',
                      'Gclef2Tot', 'Gclef2Add',
                      'CovClusteringAT100', 'CovClusteringAT200', 'CovClusteringTT100',
                      'CovClusteringAgg2Tot100+FP', 'CovClusteringAgg2Max+FP100']

algorithm_statistics(projects, data_path, results_path, algorithms, alg_complete_names, to_version)