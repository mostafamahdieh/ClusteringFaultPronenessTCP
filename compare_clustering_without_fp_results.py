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
#to_version = [26, 133, 65, 106, 27]
#to_version = [13, 50, 33, 50, 14]
to_version = [26, 133, 65, 106, 27]

data_path = 'aggregate/main_results'
output_path = 'aggregate/compare_clustering_without_fp'

cluster_nums = list(range(50,501,50))
dist_functions = ['eucl', 'manh', 'cosd']
dist_complete_names = ['Euclidean', 'Manhattan', 'Cosine']

base_algorithm_name = '_fp0_clus'

compare_clustering_results(projects, to_version, data_path, output_path, cluster_nums, dist_functions, dist_complete_names, base_algorithm_name, "_at")
