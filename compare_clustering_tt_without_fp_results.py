import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools as it
from results.compare_clustering_results import compare_clustering_results

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
#to_version = [21, 128, 60, 101, 22]
to_version = [26, 133, 65, 106, 27]

data_path = './aggregate/main_results'
output_path = './aggregate/compare_clustering_tt_without_fp'

cluster_nums = [1]+list(range(50,1001,50))
dist_functions = ['eucl']
dist_complete_names = ['Euclidean', 'Manhattan', 'Cosine']
base_alg_name = '_xgb_results_14001115_online_c0_tt_clus'

compare_clustering_results(projects, to_version, data_path, output_path, cluster_nums, dist_functions, dist_complete_names, base_alg_name, "")