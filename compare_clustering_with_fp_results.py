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
to_version = [21, 128, 60, 101, 22]
#to_version = [13, 50, 33, 50, 14]

data_path = './aggregate/main_results'
output_path = './aggregate/compare_clustering_with_fp'

cluster_nums = list(range(50,501,50))
dist_functions = ['eucl']
dist_complete_names = ['Euclidean', 'Manhattan', 'Cosine']
base_algorithmname = '_xgb_results_14001115_online_c0999_clus'

compare_clustering_results(projects, to_version, data_path, output_path, cluster_nums, dist_functions, dist_complete_names, base_algorithmname, "")