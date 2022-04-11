from results.aggregate_results import aggregate_results

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [26, 133, 65, 106, 26]

filenames = ['std.csv', 'fp0_eucl_clus50_500.csv', 'fp0_manh_clus50_500.csv', 'fp0_cosd_clus50_500.csv', 'eucl_xgb_online.csv', 'eucl_fp0.csv', 'std_xgb_online_results.csv', 'gclef_xgb_online.csv', 'art.csv']

results_path = '../WTP-data/aggregate/main_results'
aggregate_results(filenames, projects, from_version, to_version, results_path)
