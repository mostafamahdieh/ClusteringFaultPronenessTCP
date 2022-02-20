import numpy as np
import pandas
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from prioritization import prioritization_std as ps, prioritization_core as pc, prioritization_clustering as pr_cl, \
    prioritization_gclef as pr_gclef
from prioritization.prioritization_manager import extract_classes_in_data, extract_bug_prediction_for_classes, \
    extract_bug_prediction_for_units_version, generate_weighted_unit_fp


def compute_clustering_metrics(bug_prediction_data, score_label, project, version_number, clustering_method,
                               distance_function, linkage,
                               cluster_nums, c_dp_values):
    data_path = "../WTP-data/%s/%d" % (project, version_number)

    coverage, test_names, unit_names = pc.read_coverage_data(data_path)

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]
    assert (test_num == len(test_names))
    assert (unit_num == len(unit_names))

    class_of_units, units_in_class, classes = extract_classes_in_data(unit_names, unit_num)
    class_dp_prob = extract_bug_prediction_for_classes(bug_prediction_data, score_label, version_number, classes)

    class_num = len(classes);

    print("test_num: ", test_num, " unit_num: ", unit_num, " class_num: ", class_num)

    unit_dp = extract_bug_prediction_for_units_version(bug_prediction_data, score_label, class_of_units, class_dp_prob)

    metrics_all = []
    for cluster_num in cluster_nums:
        print("cluster_num: ", cluster_num)

        metrics = [cluster_num]
        for c_dp in c_dp_values:
            print("c_dp: ", c_dp)
            unit_fp = generate_weighted_unit_fp(c_dp, unit_dp, unit_num)

            clusters, clustering, model = pr_cl.create_clusters(coverage, unit_dp, unit_fp, clustering_method,
                                                         distance_function, linkage, cluster_num, c_dp != 0)
            labels = clustering.labels_
            #yhat_2 = model.fit_predict(coverage)
            # retrieve unique clusters

            score_AGclustering_s = silhouette_score(coverage, labels, metric='euclidean')
            score_AGclustering_c = calinski_harabasz_score(coverage, labels)
            score_AGclustering_d = 0 # davies_bouldin_score(coverage, yhat_2)

            print('Silhouette, Calinski Harabasz, Davies Bouldin Score: %.4f %.4f %.4f' % (score_AGclustering_s, score_AGclustering_c, score_AGclustering_d))
            metrics = metrics + [score_AGclustering_s, score_AGclustering_c, score_AGclustering_c]
        metrics_all.append(metrics)

    return metrics_all
