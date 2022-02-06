import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import Normalizer
from prioritization.prioritization_clustering import clustering_agg
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
#to_version = [26, 133, 65, 106, 27]
to_version = [26, 133, 65, 106, 26]


for index, project in enumerate(projects):
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_prioritization_clustering(project, version_number, clustering_agg, cosine_distances, 'average', [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                                         'fp0_cosd_average_clus50_500.csv', 'cosd_average_fp0')
        run_prioritization_clustering(project, version_number, clustering_agg, manhattan_distances, 'average', [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                                         'fp0_manh_average_clus50_500.csv', 'manh_average_fp0')
        run_prioritization_clustering(project, version_number, clustering_agg, euclidean_distances, 'average', [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                                         'fp0_eucl_average_clus50_500.csv', 'eucl_average_fp0')
        print()
