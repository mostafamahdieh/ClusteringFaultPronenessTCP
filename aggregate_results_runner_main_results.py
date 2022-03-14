from results.aggregate_results import aggregate_results

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [26, 133, 65, 106, 26]

filenames = ['std2_c0.csv', 'fp0_eucl_clus50_500.csv', 'fp0_manh_clus50_500.csv', 'fp0_cosd_clus50_500.csv', 'eucl_xgb_results_14001115_online_c0999_150-300.csv', 'std_xgb_results_14001115_online_c0999.csv', 'eucl_xgb_results_14001115_online_c0999_50-100-350_500.csv', 'eucl_xgb_results_14001115_online_c0999_tt_50-500.csv', 'eucl_xgb_results_14001115_online_c0999_tt_300.csv', 'gclef_xgb_results_14001115_online.csv', 'gclef2_xgb_results_14001115_online.csv', 'eucl_agg2_xgb14001115_online_c0999_50_500.csv', 'cosd_agg2_xgb14001115_online_c0999_50_500.csv','eucl_xgb_results_14001115_online_c0_tt_50_1000.csv', 'eucl_agg3_xgb14001115_online_c0999_50_500.csv', 'eucl_agg2_xgb14001115_online_c0999_1_2.csv', 'eucl_agg2_xgb14001115_online_c0999_25_175.csv', 'eucl_agg2_xgb14001115_online_c0999_225_475.csv', 'max_xgb_results_14001115_online_c0999.csv', 'gclef3_xgb_results_14001115_online.csv', 'manh2_agg2_xgb14001115_online_c0999_50_500.csv', 'gclef_random_xgb_results_14001115_online.csv', 'art.csv', 'eucl_fp0_clus_at_25_475.csv', 'art2u.csv']

results_path = '../WTP-data/aggregate/main_results'
aggregate_results(filenames, projects, from_version, to_version, results_path)
