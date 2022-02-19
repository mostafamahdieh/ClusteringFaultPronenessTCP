import numpy as np
import operator
import math
import copy

from prioritization import prioritization_std as ps
from sklearn.cluster import AgglomerativeClustering


def clustering_agg(coverage, unit_dp, unit_fp, distance_function, linkage_crit, cluster_num, use_fp):
    print("Running agglomerative clustering (cluster_num = %d)..." % (cluster_num))

    distance = distance_function(coverage, coverage)

    #    conn_eps = 0.01
    #    connectivity = similarity.copy()
    #    connectivity[connectivity > conn_eps] = 1
    #    connectivity[connectivity <= conn_eps] = 0

    print("coverage shape: ", np.shape(coverage))
    print("distance shape: ", np.shape(distance))

    if use_fp:
        fp_big_threshold = 10.0
        total_dp_coverage = np.matmul(coverage, unit_dp)
        total_sorted_arg = np.argsort(total_dp_coverage)
        cluster_subset_maxsize = math.floor(cluster_num)
        total_sorted_arg_des = total_sorted_arg[::-1]
        cluster_subset_num = min(np.sum(total_dp_coverage >= fp_big_threshold), cluster_subset_maxsize)
        print("total_dp_coverage shape: ", np.shape(total_dp_coverage))
        print("cluster_subset_num: ", cluster_subset_num)

        print("number of total_dp_coverage >= ", fp_big_threshold, ": ",
              np.sum(total_dp_coverage >= fp_big_threshold))
        print("cluster_subset_maxsize: ", cluster_subset_maxsize)
        print("total_dp_coverage[total_sorted_des[", 0, "]]: ",
              total_dp_coverage[total_sorted_arg_des[0]])
        print("total_dp_coverage[total_sorted_des[", math.floor(cluster_subset_maxsize / 2) - 1, "]]: ",
              total_dp_coverage[total_sorted_arg_des[math.floor(cluster_subset_maxsize / 2) - 1]])
        print("total_dp_coverage[total_sorted_des[", cluster_subset_maxsize - 1, "]]: ",
              total_dp_coverage[total_sorted_arg_des[cluster_subset_maxsize - 1]])

        total_sorted_arg_des_subset = total_sorted_arg_des[:cluster_subset_num]

        inf = 1.0e10
        for i in total_sorted_arg_des_subset:
            for j in total_sorted_arg_des_subset:
                distance[i, j] = inf

    clustering = AgglomerativeClustering(n_clusters=cluster_num, linkage=linkage_crit,
                                         affinity='precomputed').fit(distance)
    print("Clustering finished.")
    return clustering


def clustering_agg1(coverage, unit_dp, unit_fp, distance_function, linkage_crit, cluster_num, use_fp):
    print("Running agglomerative clustering (cluster_num = %d)..." % (cluster_num))
    clustering, model = run_clustering(cluster_num, coverage, distance_function, linkage_crit)
    print("Clustering finished.")
    return clustering, model


def clustering_agg2(coverage, unit_dp, unit_fp, distance_function, linkage_crit, cluster_num, use_fp):
    print("Running agglomerative clustering (cluster_num = %d)..." % (cluster_num))

    if use_fp:
        coverage = np.multiply(coverage, unit_fp)

    distance = distance_function(coverage, coverage)

    clustering, model = run_clustering(cluster_num, coverage, distance_function, linkage_crit)

    print("Clustering finished.")
    return clustering, model


def run_clustering(cluster_num, coverage, distance_function, linkage_crit):
    if type(distance_function) is str:
        model = AgglomerativeClustering(n_clusters=cluster_num, affinity=distance_function,
                                        linkage=linkage_crit)
        clustering = model.fit(coverage)
    else:
        distance = distance_function(coverage, coverage)
        model = AgglomerativeClustering(n_clusters=cluster_num, linkage=linkage_crit,
                                        affinity='precomputed')
        clustering = model.fit(distance)
    return clustering, model


def clustering_agg3(coverage, unit_dp, unit_fp, distance_function, linkage_crit, cluster_num, use_fp):
    print("Running agglomerative clustering (cluster_num = %d)..." % (cluster_num))

    if use_fp:
        unit_num = coverage.shape[1]
        coverage = np.multiply(coverage, 0.01 * np.ones((unit_num,)) + 0.99 * unit_dp)

    clustering, model = run_clustering(cluster_num, coverage, distance_function, linkage_crit)

    print("Clustering finished.")
    return clustering, model


def clustering_agg_nonprecomputed(coverage, unit_dp, unit_fp, distance_function, linkage_crit, cluster_num, use_fp):
    print("Running agglomerative clustering (distance = %s, linkage = %s, cluster_num = %d)..." % (
    distance_function, linkage_crit, cluster_num))

    clustering = AgglomerativeClustering(n_clusters=cluster_num, affinity=distance_function, linkage=linkage_crit).fit(
        coverage)
    print("Clustering finished.")
    return clustering


def rearrange_tests_total(cluster):
    return sorted(cluster, key=operator.itemgetter(1), reverse=True)  # sort by second value of tuple


def rearrange_tests_max(cluster):
    return sorted(cluster, key=operator.itemgetter(2), reverse=True)  # sort by second value of tuple


def is_remaining_coverage_zero(cluster, test_used):
    eps = 1e-8
    for (test_ind, total_coverage, max_coverage) in cluster:
        if not test_used[test_ind] and total_coverage > eps:
            return False
    return True


def rearrange_tests_additional(cluster, coverage, unit_fp):
    eps = 1e-8
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    test_used = [False] * test_num  # none of the tests are used at the beginning
    unit_coverage = np.ones((unit_num,))
    total_fp_coverage = np.matmul(coverage, unit_fp)
    additional_weighted_coverage = np.array(total_fp_coverage)

    rearranged_cluster = []
    while len(rearranged_cluster) < len(cluster):
        if is_remaining_coverage_zero(cluster, test_used):
            for (test_ind, total_coverage, max_coverage) in cluster:
                if not test_used[test_ind]:
                    test_used[test_ind] = True
                    rearranged_cluster.append((test_ind, total_coverage, max_coverage))
            break

        best_coverage = -1
        best_test = None

        # finding test with most coverage
        for (test_ind, total_coverage, max_coverage) in cluster:
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
            rearranged_cluster.append((best_test, total_fp_coverage[best_test]))
        else:
            additional_weighted_coverage = np.array(total_fp_coverage)
            unit_coverage = np.ones((unit_num,))

    return rearranged_cluster


def copy_clusters(clusters):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for point in cluster:
            new_cluster.append(point)

        new_clusters.append(new_cluster)

    return new_clusters


# inner_alg: 1 for total, 2 for additional
# outer_alg: 0 for nothing, 1 for total, 2 for additional
def tcp_clustering_inner_outer(clusters, coverage, unit_fp, inner_alg, outer_alg):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    clusters = copy_clusters(clusters)

    # inner cluster prioritization
    rearranged_clusters = []
    for cluster_ind in range(0, len(clusters)):
        if inner_alg == 'total':
            rearranged_cluster = rearrange_tests_total(clusters[cluster_ind])
        elif inner_alg == 'additional':
            rearranged_cluster = rearrange_tests_additional(clusters[cluster_ind], coverage, unit_fp)
        elif inner_alg == 'max':
            rearranged_cluster = rearrange_tests_max(clusters[cluster_ind])
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
        elif outer_alg == 'max':
            selected_tests = rearrange_tests_max(selected_tests)
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


def create_clusters(coverage, unit_dp, unit_fp, clustering_method, distance_function, linkage, cluster_num,
                    use_fp):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    total_fp_coverage = np.matmul(coverage, unit_fp)
    #    max_dp = np.max(np.multiply(coverage>=0.1, unit_dp), axis=1)
    #    max_dp = np.max(np.multiply(coverage, unit_dp), axis=1)
    max_dp = np.max(np.multiply(coverage >= 0.05, unit_dp), axis=1)
    assert (len(max_dp) == test_num)
    clustering, model = clustering_method(coverage, unit_dp, unit_fp, distance_function, linkage, cluster_num, use_fp)

    # constructing the clusters
    clusters = [[] for c in range(0, cluster_num)]
    for (index, val) in enumerate(clustering.labels_):
        clusters[val].append((index, total_fp_coverage[index], max_dp[index]))

    return clusters, clustering, model
