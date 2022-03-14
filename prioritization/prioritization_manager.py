from random import seed

import numpy as np
import pandas
from prioritization import prioritization_std as ps, prioritization_core as pc, prioritization_clustering as pr_cl, \
    prioritization_gclef as pr_gclef
from prioritization.prioritization_art import art_tcp, art_tcp_fast


def generate_weighted_unit_fp(c_dp, dp_unit_prob, unit_num):
    return (1 - c_dp) * np.ones((unit_num,)) + c_dp * dp_unit_prob


alg_shortname_dict = dict({"max": "max", "total": "tot", "additional": "add"})


def alg_shortname(alg_type):
    return alg_shortname_dict[alg_type]


def run_standard2_prioritization(project, version_number, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    #    f.write(
    #        "additional_first_fail,additional_apfd,total_first_fail,total_apfd,additional_dp_first_fail,additional_dp_apfd,"
    #        "total_dp_first_fail,total_dp_apfd")

    f.write("alg,first_fail,apfd\n")

    unit_fp = np.ones((unit_num,))

    additional_ordering = \
        ps.additional_prioritization_std(coverage, unit_fp, 'decrease')
    additional_apfd = pc.rank_evaluation_apfd(additional_ordering, failed_tests_ids)
    print("additional_apfd: ", additional_apfd)

    additional_first_fail = \
        pc.rank_evaluation_first_fail(additional_ordering, failed_tests_ids)
    print("additional_first_fail: ", additional_first_fail)

    result_line = "add_%s,%f,%f" % (alg_prefix, additional_first_fail, additional_apfd)
    f.write(result_line + "\n")

    total_prioritization_ordering = ps.total_prioritization(coverage, unit_fp)

    total_prioritization_apfd = pc.rank_evaluation_apfd(total_prioritization_ordering, failed_tests_ids)
    print("total_prioritization_apfd: ", total_prioritization_apfd)

    total_prioritization_first_fail = pc.rank_evaluation_first_fail(total_prioritization_ordering, failed_tests_ids)
    print("total_prioritization_first_fail: ", total_prioritization_first_fail)

    result_line = "tot_%s,%f,%f" % (alg_prefix, total_prioritization_first_fail, total_prioritization_apfd)
    f.write(result_line + "\n")

    print()
    f.close()


def run_art_prioritization(project, version_number, filename, cand_set_function, distance_function, alg_shortname, random_sample_num, mode):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    art_apfds = []
    art_first_fails = []

    distance = distance_function(coverage, coverage)

    for sample in range(random_sample_num):
        seed(sample*10)

        if mode == 'fast':
            art_ordering = art_tcp_fast(coverage, cand_set_function)
        elif mode == 'simple':
            art_ordering = art_tcp(coverage, distance_function, cand_set_function)
        elif mode == 'precomputed':
            art_ordering = art_tcp_distance(distance, distance_function, cand_set_function)

        art_apfd = pc.rank_evaluation_apfd(art_ordering, failed_tests_ids)
        art_apfds.append(art_apfd)

        art_first_fail = pc.rank_evaluation_first_fail(art_ordering, failed_tests_ids)
        art_first_fails.append(art_first_fail)

        print("art_apfd: ", art_apfd)
        print("art_first_fail: ", art_first_fail)


    art_first_fails_arr = np.array(art_first_fails)
    art_apfds_arr = np.array(art_apfds)

    result_line = alg_shortname+"_mean,%f,%f" % (np.mean(art_first_fails_arr), np.mean(art_apfds_arr))
    print(result_line)
    f.write(result_line + "\n")

    result_line = alg_shortname+"_std,%f,%f" % (np.std(art_first_fails_arr), np.std(art_apfds_arr))
    print(result_line)
    f.write(result_line + "\n")

    print()
    f.close()


def run_standard2_prioritization_fp(bug_prediction_data, score_label, project, version_number, c_dp_values, filename,
                                    alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, score_label, version_number, classes)

    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    dp_unit_prob = extract_bug_prediction_for_units_version(bug_prediction_data, score_label, class_of_units,
                                                            class_dp_prob)

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

def run_max_prioritization_fp(bug_prediction_data, score_label, project, version_number, c_dp, filename,
                                    alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, score_label, version_number, classes)

    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    dp_unit_prob = extract_bug_prediction_for_units_version(bug_prediction_data, score_label, class_of_units,
                                                            class_dp_prob)

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    print("* Running for c_dp: ", c_dp)
    unit_fp = generate_weighted_unit_fp(c_dp, dp_unit_prob, unit_num)

    max_ordering = ps.max_prioritization_std(coverage, unit_fp)
    max_apfd = pc.rank_evaluation_apfd(max_ordering, failed_tests_ids)
    print("max_apfd: ", max_apfd)

    max_first_fail = pc.rank_evaluation_first_fail(max_ordering, failed_tests_ids)
    print("max_first_fail: ", max_first_fail)

    result_line = "max_%s,%f,%f" % (alg_prefix, max_first_fail, max_apfd)
    f.write(result_line + "\n")

    print()
    f.close()


def is_remaining_coverage_zero(cluster, test_used):
    eps = 1e-8
    for (test_ind, total_coverage, max_coverage) in cluster:
        if not test_used[test_ind] and total_coverage > eps:
            return False
    return True


def extract_classes_in_data(unit_names, unit_num):
    class_of_units = []
    units_in_class = dict()
    classes = set()

    for u in range(0, unit_num):
        unit_class = str(unit_names[u])[2:].split('#')[0]
        unit_class = unit_class.strip()
        class_of_units.append(unit_class)
        if unit_class not in classes:
            units_in_class[unit_class] = []
            classes.add(unit_class)

        units_in_class[unit_class].append(u)

    for u in range(0, unit_num):
        unit_class = class_of_units[u]

        subclass = unit_class
        for retry in range(2):
            subclass = subclass.rsplit('.', 1)[0]
            if subclass in classes:
                units_in_class[subclass].append(u)

    assert (len(class_of_units) == unit_num)

    return class_of_units, units_in_class, classes


def extract_bug_prediction_for_classes(bug_prediction_data, score_label, version, classes):
    class_dp_prob = dict()

    # pandas.set_option('display.max_rows', 10000)
    # pandas.set_option('display.max_columns', 1000)

    b_version = bug_prediction_data[bug_prediction_data.version == version]
    # print(b_version)

    for code_class in classes:
        if not code_class in class_dp_prob:
            original_code_class = code_class
            for retry in range(3, 0, -1):
                b = b_version[b_version.LongName == code_class]
                if not b.empty:
                    class_dp_prob[original_code_class] = b[score_label].max()
                    break
                #                else:
                #                    print(code_class, " not found")
                if retry > 1:
                    code_class = code_class.rsplit('.', 1)[0]
    #                    print("retry. searching for: ", code_class)

    return class_dp_prob


def extract_bug_prediction_for_units_version(bug_prediction_data, score_label, class_of_units, class_dp_prob):
    unit_num = len(class_of_units)
    unit_dp_prob = np.zeros((unit_num,))
    for u in range(0, unit_num):
        unit_class = class_of_units[u]
        if unit_class in class_dp_prob:
            unit_dp_prob[u] = class_dp_prob[unit_class]

    return unit_dp_prob


def run_gclef_prioritization(bug_prediction_data, score_label, project, version_number, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)
    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, score_label, version_number, classes)

    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    assert (len(units_in_class.items()) == class_num)
    assert (len(class_of_units) == unit_num)
    print("len(class_dp_prob.items()): ", len(class_dp_prob.items()))
    #    assert(len(class_dp_prob.items()) == class_num)

    # sort classes in decreasing order of fault-proneness
    dp_classes = class_dp_prob.items()
    sorted_classes_list = sorted(dp_classes, key=lambda item: item[1], reverse=True)

    # create clusters of test cases where each cluster covers a class. clusters might have redundant test cases
    clusters, zero_cluster = pr_gclef.create_gclef_clusters(coverage, unit_names, units_in_class, sorted_classes_list)

    #    print("zero_cluster tests:")
    for (test_id, test_total_coverage, max_coverage) in zero_cluster:
        #        if (test_total_coverage >= 0.01)
        # print("test ", test_names[test_id], " (", test_id ,") with total coverage ", test_total_coverage)
        test_covers_unknown_classes = False
        if test_total_coverage >= 1:
            for u in np.flatnonzero(coverage[test_id] > 0):
                #        print(unit_names[u], " -> ", coverage[test_id][u], ", is in classes: ",  class_of_units[u], class_of_units[u] in dp_classes)
                if not class_of_units[u] in dp_classes:
                    test_covers_unknown_classes = True
        if not test_covers_unknown_classes:
            assert (test_total_coverage < 1)

    additional_ordering = pr_gclef.tcp_gclef_prioritization(clusters, coverage, 'additional')
    additional_apfd = pc.rank_evaluation_apfd(additional_ordering, failed_tests_ids)
    additional_first_fail = pc.rank_evaluation_first_fail(additional_ordering, failed_tests_ids)

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    result_line = "%s_add,%f,%f" % (alg_prefix, additional_first_fail, additional_apfd)
    f.write(result_line + "\n")
    print("additional_first_fail: ", additional_first_fail, " additional_apfd: ", additional_apfd)

    total_prioritization_ordering = pr_gclef.tcp_gclef_prioritization(clusters, coverage, 'total')
    total_prioritization_apfd = pc.rank_evaluation_apfd(total_prioritization_ordering, failed_tests_ids)
    total_prioritization_first_fail = pc.rank_evaluation_first_fail(total_prioritization_ordering, failed_tests_ids)
    result_line = "%s_tot,%f,%f" % (alg_prefix, total_prioritization_first_fail, total_prioritization_apfd)
    f.write(result_line + "\n")
    print("total_first_fail: ", total_prioritization_first_fail, " total_apfd: ", total_prioritization_apfd)

    print()
    f.close()


def run_gclef_random_prioritization(bug_prediction_data, score_label, project, version_number, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)
    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, score_label, version_number, classes)

    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    assert (len(units_in_class.items()) == class_num)
    assert (len(class_of_units) == unit_num)
    print("len(class_dp_prob.items()): ", len(class_dp_prob.items()))
    #    assert(len(class_dp_prob.items()) == class_num)

    # sort classes in decreasing order of fault-proneness
    dp_classes = class_dp_prob.items()
    sorted_classes_list = sorted(dp_classes, key=lambda item: item[1], reverse=True)

    # create clusters of test cases where each cluster covers a class. clusters might have redundant test cases
    clusters, zero_cluster = pr_gclef.create_gclef_clusters(coverage, unit_names, units_in_class, sorted_classes_list)

    #    print("zero_cluster tests:")
    for (test_id, test_total_coverage, max_coverage) in zero_cluster:
        #        if (test_total_coverage >= 0.01)
        # print("test ", test_names[test_id], " (", test_id ,") with total coverage ", test_total_coverage)
        test_covers_unknown_classes = False
        if test_total_coverage >= 1:
            for u in np.flatnonzero(coverage[test_id] > 0):
                #        print(unit_names[u], " -> ", coverage[test_id][u], ", is in classes: ",  class_of_units[u], class_of_units[u] in dp_classes)
                if not class_of_units[u] in dp_classes:
                    test_covers_unknown_classes = True
        if not test_covers_unknown_classes:
            assert (test_total_coverage < 1)

    random_ordering = pr_gclef.tcp_gclef_prioritization(clusters, coverage, 'random')
    random_apfd = pc.rank_evaluation_apfd(random_ordering, failed_tests_ids)
    random_first_fail = pc.rank_evaluation_first_fail(random_ordering, failed_tests_ids)

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    result_line = "%s_add,%f,%f" % (alg_prefix, random_first_fail, random_apfd)
    f.write(result_line + "\n")
    print("random_first_fail: ", random_first_fail, " random_apfd: ", random_apfd)

    print()
    f.close()


def cluster_sizes(clusters):
    cluster_sizes = []
    for cluster in clusters:
        cluster_sizes.append(len(cluster))

    cluster_sizes.sort(reverse=True)
    return cluster_sizes


def run_prioritization_clustering_fp(bug_prediction_data, score_label, project, version_number, clustering_method,
                                     distance_function, linkage, inner_prioritizations,
                                     cluster_nums, c_dp_values, filename, alg_prefix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, score_label, version_number, classes)

    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    unit_dp = extract_bug_prediction_for_units_version(bug_prediction_data, score_label, class_of_units, class_dp_prob)

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    for ind, c_dp in enumerate(c_dp_values):
        unit_fp = generate_weighted_unit_fp(c_dp, unit_dp, unit_num)
        print("c_dp: ", c_dp)

        if c_dp != 0:
            print('adjusting coverage...')
            coverage1 = np.multiply(coverage, unit_fp)
        else:
            coverage1 = coverage

        if not type(distance_function) is str:
            distance = distance_function(coverage1, coverage1)
            print('distance computed.')
        else:
            distance = distance_function

        for cluster_num in cluster_nums:

            clusters, clustering, labels = pr_cl.create_clusters(coverage, unit_dp, unit_fp, clustering_method,
                                                         distance, linkage, cluster_num, c_dp != 0)

            print("cluster_sizes: ", cluster_sizes(clusters))


            for inner_prioritization in inner_prioritizations:
                print("Running tcp_clustering_inner_outer_fp for c_dp: ", c_dp, " inner alg: ", inner_prioritization)
                if inner_prioritization == 'max':
                    outer_prioritization = 'max'
                else:
                    outer_prioritization = 'total'
                ranks = pr_cl.tcp_clustering_inner_outer(clusters, coverage, unit_fp, inner_prioritization,
                                                         outer_prioritization)
                first_fail = pc.rank_evaluation_first_fail(ranks, failed_tests_ids)
                apfd = pc.rank_evaluation_apfd(ranks, failed_tests_ids)
                print("first_fail: ", first_fail, " apfd: ", apfd)
                result_line = "%s_%s_clus%d,%f,%f" % (
                alg_prefix[ind], alg_shortname(inner_prioritization), cluster_num, first_fail, apfd)
                f.write(result_line + "\n")

    print()
    f.close()


def run_prioritization_clustering(project, version_number, clustering_method, distance_function, linkage, cluster_nums,
                                  filename, alg_prefix, alg_postfix):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)
    failed_tests_ids = pc.read_failed_tests(data_path, test_names)
    unit_num = coverage.shape[1]

    if np.size(failed_tests_ids) == 0:
        print("No Tests found in coverage values, skipping version")
        return

    f = open('%s/%s' % (data_path, filename), "w+")
    f.write("alg,first_fail,apfd\n")

    if not type(distance_function) is str:
        distance1 = distance_function(coverage, coverage)
        print('distance computed.')
    else:
        distance1 = distance_function

    unit_fp = np.ones(unit_num)
    for cluster_num in cluster_nums:
        clusters, clustering, labels = pr_cl.create_clusters(coverage, unit_fp, unit_fp, clustering_method,
                                                     distance1, linkage, cluster_num, False)
        
        print("cluster_sizes: ", cluster_sizes(clusters))

        ranks = pr_cl.tcp_clustering_inner_outer(clusters, coverage, unit_fp, 'additional', 'total')
        first_fail = pc.rank_evaluation_first_fail(ranks, failed_tests_ids)
        apfd = pc.rank_evaluation_apfd(ranks, failed_tests_ids)
        print("first_fail: ", first_fail, " apfd: ", apfd)

        result_line = "%s_clus%d%s,%f,%f" % (alg_prefix, cluster_num, alg_postfix, first_fail, apfd)

        f.write(result_line + "\n")

    print()
    f.close()
