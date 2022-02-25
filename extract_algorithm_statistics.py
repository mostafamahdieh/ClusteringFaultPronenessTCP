from results.algorithm_statistics import algorithm_statistics

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
to_version = [21, 128, 60, 101, 22]
#to_version = [13, 50, 33, 50, 14]

data_path = './aggregate/main_results'
results_path = './aggregate/main_results'

algorithms = ['tot_c0', 'add_c0', 'tot_std_xgb_results_14001115_online_c0999', 'add_std_xgb_results_14001115_online_c0999',
              'gclef_tot_gclef2_xgb_results_14001115_online', 'gclef_add_gclef2_xgb_results_14001115_online', 'max_max_xgb_results_14001115_online_c0999',
              'eucl_fp0_clus_at', 'eucl_agg2_xgb14001115_online_c0999_max_clus']



alg_complete_names = ['Total', 'Add', 'Total+FP', 'Add+FP',
                      'Gclef2Tot', 'Gclef2Add', 'Max+FP',
                      'CovClusteringAT', 'CovClusteringMax+FP']

cluster_nums = [100, 150, 200, 150, 150]
#cluster_nums = [100, 150, 175, 125, 150]
cluster_nums_fp = [175, 150, 150, 150, 75]

algorithm_statistics(projects, data_path, results_path, algorithms, alg_complete_names, to_version, cluster_nums, cluster_nums_fp)