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

    test_class_coverage = np.multiply(coverage, unit_class_matrix)
    nonzero_tests, nonzero_classes = np.where(test_class_coverage > 0.01)

    clusters = [[] for c in range(0, class_num)]

    for ind, test in enumerate(nonzero_tests):
        clusters[nonzero_classes[ind]].append((test, total_weighted_coverage[test]))

    return clusters


def tcp_gclef_prioritization(clusters, coverage, inner_alg):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    # inner cluster prioritization
    rearranged_clusters = []
    for cluster_ind in range(0, len(clusters)):
        if inner_alg == 'total':
            rearranged_cluster = pr_cl.rearrange_tests_total(clusters[cluster_ind])
        elif inner_alg == 'additional':
            rearranged_cluster = pr_cl.rearrange_tests_additional(clusters[cluster_ind], coverage, np.ones((unit_num,)))
        else:
            raise Exception("Bad value for inner_alg: " + str(inner_alg))

        assert len(clusters[cluster_ind]) == len(rearranged_cluster)
        # print(cluster_ind,": ",rearranged_cluster)
        rearranged_clusters.append(rearranged_cluster)

    ranks = np.zeros((test_num,))
    selected_tests = set()

    for cluster in rearranged_clusters:
        for (test_id, tot_weight) in cluster:
            if test_id not in selected_tests:
                ranks[len(selected_tests)] = test_id
                selected_tests.add(test_id)

    return ranks


