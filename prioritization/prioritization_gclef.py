import numpy as np
from prioritization import prioritization_clustering as pr_cl


def create_gclef_clusters(coverage, unit_names, units_in_class, sorted_classes_list):
    unit_num = coverage.shape[1]
    total_weighted_coverage = np.matmul(coverage, np.ones((unit_num,)))

    class_num = len(sorted_classes_list)
    unit_class_matrix = np.zeros((unit_num, class_num))

    for (ind, (cl, class_dp_prob)) in enumerate(sorted_classes_list):
        for u in units_in_class[cl]:
            unit_class_matrix[u][ind] = 1

    test_class_coverage = np.matmul(coverage, unit_class_matrix)
    nonzero_coverage = test_class_coverage >= 0.001
    nonzero_tests, nonzero_classes = np.where(nonzero_coverage)
    
    clusters = [[] for c in range(0, class_num)]

    for ind, test in enumerate(nonzero_tests):
        clusters[nonzero_classes[ind]].append((test, total_weighted_coverage[test]))

    zero_rows = np.flatnonzero(nonzero_coverage.sum(axis=1) == 0)
    zero_cluster = []
    for ind, test in enumerate(zero_rows):
        zero_cluster.append((test, total_weighted_coverage[test]))

    if len(zero_cluster) > 0:
        clusters.append(zero_cluster)

    print("len(zero_cluster): ", len(zero_cluster))

    return clusters, zero_cluster


def tcp_gclef_prioritization(clusters, coverage, inner_alg):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    # inner cluster prioritization
    selected_tests = set()
    rearranged_clusters = []
    for cluster_ind in range(0, len(clusters)):
        unique_cluster = []
        for (test_id, tot_weight) in clusters[cluster_ind]:
            if test_id not in selected_tests:
                unique_cluster.append((test_id, tot_weight))
                selected_tests.add(test_id)
        print("rearrenging cluster #", cluster_ind, " with size ", len(unique_cluster))
        if inner_alg == 'total':
            rearranged_cluster = pr_cl.rearrange_tests_total(unique_cluster)
        elif inner_alg == 'additional':
            rearranged_cluster = pr_cl.rearrange_tests_additional(unique_cluster, coverage, np.ones((unit_num,)), 'zero')
        else:
            raise Exception("Bad value for inner_alg: " + str(inner_alg))

        assert len(unique_cluster) == len(rearranged_cluster)
        rearranged_clusters.append(rearranged_cluster)

    print("len(selected_tests: ", len(selected_tests), " test_num: ", test_num)
    assert(len(selected_tests) == test_num)

    ranks = np.zeros((test_num,))

    added_tests = 0
    for cluster in rearranged_clusters:
        for (test_id, tot_weight) in cluster:
            ranks[added_tests] = test_id
            added_tests = added_tests+1

    assert(added_tests == test_num)

    return ranks


