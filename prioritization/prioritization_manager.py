import numpy as np
from prioritization import prioritization_std as ps,  prioritization_core as pc, prioritization_clustering as pr_cl, prioritization_gclef as pr_gclef


def extract_bug_prediction_for_units_version(bug_prediction_data, version, unit_names, unit_num):
    unit_dp_prob = np.zeros((unit_num,))
    for u in range(0, unit_num):
        unit_class = str(unit_names[u])[2:].split('#')[0]
        unit_class = unit_class.strip()
        b_all = bug_prediction_data[bug_prediction_data.LongName == unit_class]
        b = b_all[b_all.version == version]
        if not b.empty:
            unit_dp_prob[u] = b.xgb_score.max()
    return unit_dp_prob


def generate_weighted_unit_fp(c_dp, dp_unit_prob, unit_num):
    return (1 - c_dp) * np.ones((unit_num,)) + c_dp * dp_unit_prob


def alg_to_char(alg_type):
    return alg_type[0]


def run_standard2_prioritization(bug_prediction_data, project, version_number, c_dp_values, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)
    unit_num = coverage.shape[1]

    dp_unit_prob = extract_bug_prediction_for_units_version(bug_prediction_data, version_number, unit_names, unit_num)

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    #    f.write(
    #        "additional_first_fail,additional_apfd,total_first_fail,total_apfd,additional_dp_first_fail,additional_dp_apfd,"
    #        "total_dp_first_fail,total_dp_apfd")

    f.write("alg,first_fail,apfd\n")

    for ind, c_dp in enumerate(c_dp_values):
        print("* Running for c_dp: ", c_dp)
        unit_fp = generate_weighted_unit_fp(c_dp, dp_unit_prob, unit_num)

        additional_ordering = \
            ps.additional_prioritization_std(coverage, unit_fp, 'decrease')
        additional_apfd = pc.rank_evaluation_apfd(additional_ordering, failed_tests_ids)
        print("additional_apfd: ", additional_apfd)

        additional_first_fail = \
            pc.rank_evaluation_first_fail(additional_ordering, failed_tests_ids)
        print("additional_first_fail: ", additional_first_fail)

        result_line = "add_%s,%f,%f" % (alg_prefix[ind], additional_first_fail, additional_apfd)
        f.write(result_line + "\n")

        total_prioritization_ordering = ps.total_prioritization(coverage, unit_fp)

        total_prioritization_apfd = pc.rank_evaluation_apfd(total_prioritization_ordering, failed_tests_ids)
        print("total_prioritization_apfd: ", total_prioritization_apfd)

        total_prioritization_first_fail = pc.rank_evaluation_first_fail(total_prioritization_ordering, failed_tests_ids)
        print("total_prioritization_first_fail: ", total_prioritization_first_fail)

        result_line = "tot_%s,%f,%f" % (alg_prefix[ind], total_prioritization_first_fail, total_prioritization_apfd)
        f.write(result_line + "\n")

    print()
    f.close()


def is_remaining_coverage_zero(cluster, test_used):
    eps = 1e-8
    for (test_ind, total_coverage) in cluster:
        if not test_used[test_ind] and total_coverage > eps:
            return False
    return True


def extract_classes_in_data(unit_names, unit_num):
    class_of_units = []
    units_in_class = dict()

    for u in range(0, unit_num):
        unit_class = str(unit_names[u])[2:].split('#')[0]
        unit_class = unit_class.strip()
        class_of_units.append(unit_class)
        if unit_class not in units_in_class:
            units_in_class[unit_class] = []
        units_in_class[unit_class].append(u)

    return class_of_units, units_in_class


def extract_bug_prediction_for_classes(bug_prediction_data, version, unit_names, class_of_units):
    class_dp_prob = dict()

    for unit_class in class_of_units:
        if not unit_class in class_dp_prob:
            b_all = bug_prediction_data[bug_prediction_data.LongName == unit_class]
            b = b_all[b_all.version == version]
            if not b.empty:
                class_dp_prob[unit_class] = b.xgb_score.max()

    return class_dp_prob


def run_gclef_prioritization(bug_prediction_data, project, version_number, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)
    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    unit_num = coverage.shape[1]
    class_of_units, units_in_class = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, version_number, unit_names, class_of_units)

    # sort classes in decreasing order of fault-proneness
    sorted_classes_list = sorted(class_dp_prob.items(), key=lambda item: item[1], reverse=True)

    # create clusters of test cases where each cluster covers a class. clusters might have redundant test cases
    clusters = pr_gclef.create_gclef_clusters(coverage, unit_names, units_in_class, sorted_classes_list)

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    additional_ordering = pr_gclef.tcp_gclef_prioritization(clusters, coverage, 'additional')
    additional_apfd = pc.rank_evaluation_apfd(additional_ordering, failed_tests_ids)
    additional_first_fail = pc.rank_evaluation_first_fail(additional_ordering, failed_tests_ids)
    result_line = "gclef_add_%s,%f,%f" % (alg_prefix, additional_first_fail, additional_apfd)
    f.write(result_line + "\n")

    total_prioritization_ordering = pr_gclef.tcp_gclef_prioritization(clusters, coverage, 'total')
    total_prioritization_apfd = pc.rank_evaluation_apfd(total_prioritization_ordering, failed_tests_ids)
    total_prioritization_first_fail = pc.rank_evaluation_first_fail(total_prioritization_ordering, failed_tests_ids)
    result_line = "gclef_tot_%s,%f,%f" % (alg_prefix, total_prioritization_first_fail, total_prioritization_apfd)
    f.write(result_line + "\n")

    print()
    f.close()


def run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_method, distance_function, cluster_nums, c_dp_values, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)
    unit_num = coverage.shape[1]

    dp_unit_prob = extract_bug_prediction_for_units_version(bug_prediction_data, version_number, unit_names, unit_num)

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    for ind, c_dp in enumerate(c_dp_values):
        for cluster_num in cluster_nums:
            unit_fp = generate_weighted_unit_fp(c_dp, dp_unit_prob, unit_num)

            if c_dp == 0:
                clusters, clustering = pr_cl.create_clusters(coverage, np.zeros(unit_num), clustering_method, distance_function, cluster_num)
            else:
                clusters, clustering = pr_cl.create_clusters(coverage, dp_unit_prob, clustering_method, distance_function, cluster_num)

            for inner_alg in ['additional']:
                for outer_alg in ['total']:
                    print("Running tcp_clustering_inner_outer_fp for inner_alg: ", inner_alg, " outer_alg: ", outer_alg, " c_dp: ", c_dp)
                    ranks = pr_cl.tcp_clustering_inner_outer(clusters, coverage, unit_fp, inner_alg, outer_alg)
                    first_fail = pc.rank_evaluation_first_fail(ranks, failed_tests_ids)
                    apfd = pc.rank_evaluation_apfd(ranks, failed_tests_ids)
                    print("first_fail: ", first_fail, " apfd: ", apfd)

                    result_line = "%s_clus%d_%s%s,%f,%f" % (alg_prefix[ind], cluster_num, alg_to_char(inner_alg), alg_to_char(outer_alg), first_fail, apfd)

                    f.write(result_line + "\n")

    print()
    f.close()

