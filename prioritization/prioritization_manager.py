import numpy as np
from prioritization import prioritization_std as ps,  prioritization_core as pc, prioritization_clustering as pr_cl


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


def run_prioritization_clustering_fp(bug_prediction_data, project, version_number, clustering_method, cluster_num, c_dp_values, filename, alg_prefix):
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

    for distance_metric in ['jaccard', 'matching', 'dice', 'kulsinski', 'rogerstanimoto', 'russellrao', 'sokalmichener',
                            'sokalsneath']:

        clusters, clustering = pr_cl.create_clusters(coverage, dp_unit_prob, clustering_method, distance_metric, cluster_num)

        for ind, c_dp in enumerate(c_dp_values):
            unit_fp = generate_weighted_unit_fp(c_dp, dp_unit_prob, unit_num)
            for inner_alg in ['total']:
                for outer_alg in ['total']:
                    print("Running tcp_clustering_inner_outer_fp for inner_alg: ", inner_alg, " outer_alg: ", outer_alg, " c_dp: ", c_dp)
                    ranks = pr_cl.tcp_clustering_inner_outer(clusters, coverage, unit_fp, inner_alg, outer_alg)
                    first_fail = pc.rank_evaluation_first_fail(ranks, failed_tests_ids)
                    apfd = pc.rank_evaluation_apfd(ranks, failed_tests_ids)
                    print("first_fail: ", first_fail, " apfd: ", apfd)

                    result_line = "%s_%s_%s%s,%f,%f" % (alg_prefix[ind], distance_metric, alg_to_char(inner_alg), alg_to_char(outer_alg), first_fail, apfd)

                    f.write(result_line + "\n")

    print()
    f.close()

