import pandas as pd
from matplotlib import pyplot as plt

cluster_nums = [2]+list(range(5,501,5))
projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [1, 1, 1, 1, 1]


for index, project in enumerate(projects):
    for version_number in range(from_version[index], to_version[index] + 1):
        print("Project: ", project, " Version: ", version_number)

        proj_metrics = pd.read_csv('clustering_metrics/server/%s_%d.csv' % (project, version_number))
        cluster_nums = proj_metrics["cluster_num"]

        metrics = ['silhouette', 'calinski', 'davies']

        for metric in metrics:
            fig1 = plt.figure()
            plt.plot(cluster_nums, proj_metrics[metric], '-', label=metric)
            plt.plot(cluster_nums, proj_metrics[metric + '_fp'], '-', label=metric + '_fp')
            plt.legend()

            fig1.savefig('clustering_metrics/plots/%s_%s.png' % (project,metric))
            plt.close('all')