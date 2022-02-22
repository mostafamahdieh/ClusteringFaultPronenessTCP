import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from prioritization import prioritization_std as ps, prioritization_core as pc, prioritization_clustering as pr_cl, \
    prioritization_gclef as pr_gclef
from prioritization.prioritization_manager import extract_classes_in_data, extract_bug_prediction_for_classes, \
    extract_bug_prediction_for_units_version, generate_weighted_unit_fp
from results.compare_clustering_results import get_project_tcp_results


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

    if not type(distance_function) is str:
        distance1 = distance_function(coverage, coverage)
        print('distance computed.')
    else:
        distance1 = distance_function

    for cluster_num in cluster_nums:
        print("cluster_num: ", cluster_num)

        metrics = [cluster_num]
        for c_dp in c_dp_values:
            print("c_dp: ", c_dp)
            unit_fp = generate_weighted_unit_fp(c_dp, unit_dp, unit_num)

            clusters, clustering, model = pr_cl.create_clusters(coverage, unit_dp, unit_fp, clustering_method,
                                                         distance1, linkage, cluster_num, c_dp != 0)
            labels = clustering.labels_
            # retrieve unique clusters

            score_AGclustering_s = silhouette_score(coverage, labels, metric='euclidean')
            score_AGclustering_c = calinski_harabasz_score(coverage, labels)
            score_AGclustering_d = davies_bouldin_score(coverage, labels)

            print('Silhouette, Calinski Harabasz, Davies Bouldin Score: %.4f %.4f %.4f' % (score_AGclustering_s, score_AGclustering_c, score_AGclustering_d))
            metrics = metrics + [score_AGclustering_s, score_AGclustering_c, score_AGclustering_d]
        metrics_all.append(metrics)

    return metrics_all

def plot_clustering_metrics(data_path, project, version_number, base_alg_name, after_name, metrics, matric_labels, cluster_nums_filter, tcp_results_cluster_nums, dist_functions, dist_complete_names, tcp_to_version, is_fp, output_name):
    first_fail = pd.read_csv(data_path + "/first_fail_all.csv")
    apfd = pd.read_csv(data_path + "/apfd_all.csv")
    vals = first_fail
    proj_metrics = pd.read_csv('clustering_metrics/1/%s_%d.csv' % (project, version_number))

#    print(proj_metrics.shape)
#    print(len(cluster_nums))

    proj_metrics = proj_metrics[proj_metrics["cluster_num"].isin(cluster_nums_filter)]
    cluster_nums = proj_metrics["cluster_num"]

    for index, metric in enumerate(metrics):
        fig1 = plt.figure()
        plt.plot(cluster_nums, proj_metrics[metric], '--', label=metric)
        plt.plot(cluster_nums, proj_metrics[metric + '_fp'], '--', label=metric + '_fp')
        plt.xticks([2] + list(range(50, 501, 50)))
        plt.legend()

        fig1.savefig('clustering_metrics/plots/%s_%s.png' % (project, metric))

        fig2, axs = plt.subplots(2, 1, sharex=True)
        # Remove horizontal space between axes
        fig2.subplots_adjust(hspace=0)

        if is_fp:
            axs[0].plot(cluster_nums, proj_metrics[metric + '_fp'], '-.', label=matric_labels[index])
        else:
            axs[0].plot(cluster_nums, proj_metrics[metric], '--', label=matric_labels[index])
        axs[0].set_xticks([2] + list(range(50, 501, 50)))
        axs[0].legend()
        # plt.legend(title='Distance function')
        # plt.ylabel('APFD (%)')
        axs[1].set_ylabel('First fail (%)')
        axs[1].grid(axis='x', color='0.95')
        plt.suptitle(project)

        proj_stats, proj_stats_data = get_project_tcp_results(base_alg_name, after_name, tcp_results_cluster_nums, dist_functions, project,
                                tcp_to_version, vals)

        for (ind_dist, dist) in enumerate(dist_functions):
            axs[1].plot(proj_stats["Cluster #"], proj_stats[dist], 'o--', label=dist_complete_names[ind_dist])

        fig2.savefig('clustering_metrics/plots/clus_hop5/%s_%s_%s.png' % (output_name, project, metric))

        plt.close('all')
