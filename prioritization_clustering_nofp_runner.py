import pandas as pd
from prioritization.prioritization_manager import run_prioritization_clustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import Normalizer
from prioritization.prioritization_clustering import clustering_agg1
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [26, 133, 65, 106, 26]
#to_version = [26, 133, 65, 106, 27]


for index, project in enumerate(projects):
    print("Project " + project)
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_prioritization_clustering(project, version_number, clustering_agg1, 'euclidean', 'ward', list(range(25,201,25))
                                         , 'eucl_ward_at_nofp_20.csv',   ['eucl_ward_at_nofp_'])
#        run_prioritization_clustering(project, version_number, clustering_agg, cosine_distances, 'average', list(range(50,501,50),
#                                         , 'eucl_at_nofp_50_500.csv',   ['eucl_at_nofp_'])
#        run_prioritization_clustering(project, version_number, clustering_agg, manhattan_distances, 'average', list(range(50,501,50),
#                                         , 'eucl_at_nofp_50_500.csv',   ['eucl_at_nofp_'])
        print()
