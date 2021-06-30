import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.cluster import AgglomerativeClustering
import operator
import math

from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import Normalizer

from prioritization import prioritization_std as ps


def clustering_agg11(coverage, dp_unit_prob, cluster_num):
    print("Running agglomerative clustering (cluster_num = %d)..." % (cluster_num))

#    coverage_eps = 0.1
#    coverage_binary = coverage > coverage_eps
    inf = 1.0e10

    coverage_normalized = Normalizer().transform(coverage)
    distance = nan_euclidean_distances(coverage_normalized, coverage_normalized)
#    distance = euclidean_distances(coverage, coverage)
#    similarity = np.matmul(coverage_normalized, np.matrix.transpose(coverage_normalized))
#    distance = 1-similarity

#    dist = DistanceMetric.get_metric(distance_metric)
#    distance = dist.pairwise(coverage_binary, coverage_binary)
    distance = np.nan_to_num(distance, inf)

#    conn_eps = 0.01
#    connectivity = similarity.copy()
#    connectivity[connectivity > conn_eps] = 1
#    connectivity[connectivity <= conn_eps] = 0


    fp_big_threshold = 0.2
    total_weighted_coverage = np.matmul(coverage, dp_unit_prob)
    total_sorted_arg = np.argsort(total_weighted_coverage)
    cluster_subset_maxsize = math.floor(cluster_num)
    total_sorted_arg_des = total_sorted_arg[::-1]

    print("coverage shape: ", np.shape(coverage))
    print("distance shape: ", np.shape(distance))
    print("total_weighted_coverage shape: ", np.shape(total_weighted_coverage))
    print("number of total_weighted_coverage >= ", fp_big_threshold, ": ", np.sum(total_weighted_coverage >= fp_big_threshold))
    print("cluster_subset_maxsize: ", cluster_subset_maxsize)
    print("total_weighted_coverage[total_sorted_des[", cluster_subset_maxsize-1, "]]: ",
          total_weighted_coverage[total_sorted_arg_des[cluster_subset_maxsize-1]])

    cluster_subset_num = min(np.sum(total_weighted_coverage >= fp_big_threshold), cluster_subset_maxsize)
    print("cluster_subset_num: ", cluster_subset_num)

    total_sorted_arg_des_subset = total_sorted_arg_des[:cluster_subset_num]

    for i in total_sorted_arg_des_subset:
        for j in total_sorted_arg_des_subset:
            distance[i, j] = inf

    clustering = AgglomerativeClustering(n_clusters=cluster_num, linkage='average',
                                         affinity='precomputed').fit(distance)
    print("Clustering finished.")
    return clustering


def clustering_agg12(coverage, dp_unit_prob, cluster_num):
    print("Running agglomerative clustering (cluster_num = %d)..." % (cluster_num))

#    coverage_eps = 0.1
#    coverage_binary = coverage > coverage_eps
    inf = 1.0e10

    coverage_normalized = Normalizer().transform(coverage)
#    distance = nan_euclidean_distances(coverage_normalized, coverage_normalized)
#    distance = euclidean_distances(coverage, coverage)
    similarity = np.matmul(coverage_normalized, np.matrix.transpose(coverage_normalized))
    distance = 1-similarity

#    dist = DistanceMetric.get_metric(distance_metric)
#    distance = dist.pairwise(coverage_binary, coverage_binary)
#    distance = np.nan_to_num(distance, inf)

#    conn_eps = 0.01
#    connectivity = similarity.copy()
#    connectivity[connectivity > conn_eps] = 1
#    connectivity[connectivity <= conn_eps] = 0


    fp_big_threshold = 0.2
    total_weighted_coverage = np.matmul(coverage, dp_unit_prob)
    total_sorted_arg = np.argsort(total_weighted_coverage)
    cluster_subset_maxsize = math.floor(cluster_num)
    total_sorted_arg_des = total_sorted_arg[::-1]

    print("coverage shape: ", np.shape(coverage))
    print("distance shape: ", np.shape(distance))
    print("total_weighted_coverage shape: ", np.shape(total_weighted_coverage))
    print("number of total_weighted_coverage >= ", fp_big_threshold, ": ", np.sum(total_weighted_coverage >= fp_big_threshold))
    print("cluster_subset_maxsize: ", cluster_subset_maxsize)
    print("total_weighted_coverage[total_sorted_des[", cluster_subset_maxsize-1, "]]: ",
          total_weighted_coverage[total_sorted_arg_des[cluster_subset_maxsize-1]])

    cluster_subset_num = min(np.sum(total_weighted_coverage >= fp_big_threshold), cluster_subset_maxsize)
    print("cluster_subset_num: ", cluster_subset_num)

    total_sorted_arg_des_subset = total_sorted_arg_des[:cluster_subset_num]

    for i in total_sorted_arg_des_subset:
        for j in total_sorted_arg_des_subset:
            distance[i, j] = inf

    clustering = AgglomerativeClustering(n_clusters=cluster_num, linkage='average',
                                         affinity='precomputed').fit(distance)
    print("Clustering finished.")
    return clustering
def create_clusters(coverage, clustering_method, cluster_num):
    unit_num = coverage.shape[1]

    clustering = clustering_method(coverage, cluster_num)
    total_weighted_coverage = np.matmul(coverage, np.ones((unit_num,)))

    # constructing the clusters
    clusters = [[] for c in range(0, cluster_num)]
    for (index, val) in enumerate(clustering.labels_):
        clusters[val].append((index, total_weighted_coverage[index]))

    return clusters, clustering


def tcp_full_total(clusters, test_num):
    for cluster_ind in range(0, len(clusters)):
        clusters[cluster_ind].sort(key=operator.itemgetter(1), reverse=True)  # sort by second value of tuple

    ranks = []
    selection_round = 0

    while len(ranks) < test_num:
        for cluster_ind in range(0, len(clusters)):
            if selection_round < len(clusters[cluster_ind]):
                ranks.append(clusters[cluster_ind][selection_round][0])
        selection_round = selection_round + 1

    return ranks


def rearrange_tests_total(cluster):
    return sorted(cluster, key=operator.itemgetter(1), reverse=True)  # sort by second value of tuple


def is_remaining_coverage_zero(cluster, testUsed):
    eps = 1e-8
    for (test_ind, total_coverage) in cluster:
        if not testUsed[test_ind] and total_coverage > eps:
            return False
    return True


def rearrange_tests_additional(cluster, coverage, unit_fp):
    eps = 1e-8
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    test_used = [False] * test_num  # none of the tests are used at the beginning
    unit_coverage = np.ones((unit_num,))
    total_weighted_coverage = np.matmul(coverage, unit_fp)
    additional_weighted_coverage = np.array(total_weighted_coverage)

    rearranged_cluster = []
    while len(rearranged_cluster) < len(cluster):
        if is_remaining_coverage_zero(cluster, test_used):
            for (test_ind, total_coverage) in cluster:
                if not test_used[test_ind]:
                    test_used[test_ind] = True
                    rearranged_cluster.append((test_ind, total_coverage))
            break

        best_coverage = -1
        best_test = None

        # finding test with most coverage
        for (test_ind, total_coverage) in cluster:
            if not test_used[test_ind] and additional_weighted_coverage[test_ind] > best_coverage:
                best_test = test_ind
                best_coverage = additional_weighted_coverage[test_ind]

        assert best_coverage != -1, "Didn't find any test (this must not happen)!"
        assert best_coverage > -eps, "Coverage (" + str(best_coverage) + ") must not be negative!"

        if best_coverage > eps:
            new_unit_coverage = np.maximum(unit_coverage - coverage[best_test, :], 0)
            coverage_diff = (unit_coverage - new_unit_coverage)
            additional_weighted_coverage -= np.matmul(coverage, np.multiply(coverage_diff, unit_fp))
            unit_coverage = new_unit_coverage
            test_used[best_test] = True
            rearranged_cluster.append((best_test, total_weighted_coverage[best_test]))
        else:
            additional_weighted_coverage = np.array(total_weighted_coverage)
            unit_coverage = np.ones((unit_num,))

    return rearranged_cluster


def change_clusters_total_coverage(clusters, total_weighted_coverage):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for (test_ind, total_coverage) in cluster:
            new_cluster.append((test_ind, total_weighted_coverage[test_ind]))

        new_clusters.append(new_cluster)

    return new_clusters


# inner_alg: 1 for total, 2 for additional
# outer_alg: 0 for nothing, 1 for total, 2 for additional
def tcp_clustering_inner_outer(clusters, coverage, unit_fp, inner_alg, outer_alg):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    total_weighted_coverage = np.matmul(coverage, unit_fp)
    clusters = change_clusters_total_coverage(clusters, total_weighted_coverage)

    # inner cluster prioritization
    rearranged_clusters = []
    for cluster_ind in range(0, len(clusters)):
        if inner_alg == 'total':
            rearranged_cluster = rearrange_tests_total(clusters[cluster_ind])
        elif inner_alg == 'additional':
            rearranged_cluster = rearrange_tests_additional(clusters[cluster_ind], coverage, unit_fp)
        else:
            raise Exception("Bad value for inner_alg: " + str(inner_alg))

        assert len(clusters[cluster_ind]) == len(rearranged_cluster)
        # print(cluster_ind,": ",rearranged_cluster)
        rearranged_clusters.append(rearranged_cluster)

    total_selected_tests = 0
    ranks = np.zeros((test_num,))
    selection_round = 0

    while total_selected_tests < test_num:
        selected_tests = []
        for cluster_ind in range(0, len(rearranged_clusters)):
            if selection_round < len(rearranged_clusters[cluster_ind]):
                selected_tests.append(rearranged_clusters[cluster_ind][selection_round])

        if outer_alg == 'nothing':
            pass
        elif outer_alg == 'total':
            selected_tests = rearrange_tests_total(selected_tests)
        elif outer_alg == 'additional':
            selected_tests = rearrange_tests_additional(selected_tests, coverage, unit_fp)
        else:
            raise Exception("Bad value for outer_alg: " + str(outer_alg))

        for i in range(0, len(selected_tests)):
            ranks[total_selected_tests + i] = selected_tests[i][0]

        total_selected_tests = total_selected_tests + len(selected_tests)
        selection_round = selection_round + 1

    return ranks


def compute_ordering_index(ordering):
    ordering_index = [-1] * len(ordering)
    for i in range(0, len(ordering)):
        ordering_index[ordering[i]] = i
    return ordering_index


def create_clusters(coverage, dp_unit_prob, clustering_method, cluster_num):
    unit_num = coverage.shape[1]

    clustering = clustering_method(coverage, dp_unit_prob, cluster_num)
    total_weighted_coverage = np.matmul(coverage, np.ones((unit_num,)))

    # constructing the clusters
    clusters = [[] for c in range(0, cluster_num)]
    for (index, val) in enumerate(clustering.labels_):
        clusters[val].append((index, total_weighted_coverage[index]))

    return clusters, clustering
