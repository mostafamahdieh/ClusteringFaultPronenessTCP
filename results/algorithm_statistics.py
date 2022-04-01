import pandas as pd
from pandas import Categorical
import numpy as np
from numpy import std, mean, sqrt
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools as it


def algorithm_statistics(projects, data_path, results_path, algorithms, alg_complete_names, to_version, cluster_nums, cluster_nums_fp, proposed_alg_nofp, proposed_alg_with_fp):
    algorithm_stats = []
    proposed_alg_nofp_wil_stats = []
    proposed_alg_with_fp_wil_stats = []
    first_fail = pd.read_csv(data_path + "/first_fail_all.csv")
    apfd = pd.read_csv(data_path + "/apfd_all.csv")
    vals = first_fail

    vals_agg_nofp = pd.DataFrame(columns=[proposed_alg_nofp])
    vals_agg_with_fp = pd.DataFrame(columns=[proposed_alg_with_fp])
    vals_agg_cmp = pd.DataFrame(columns=algorithms)

    for (index, project) in enumerate(projects):
        print("Project ", project)
        vals_proj = vals[vals.version <= to_version[index]]
        vals_proj = vals_proj[vals_proj.project == project]
        vals_proj_mean = vals_proj.mean()
        plt.close('all')

        stat = [project]
        stat_nofp_wil = [project]
        stat_with_fp_wil = [project]

        proposed_alg_nofp0 = proposed_alg_nofp.replace('clus', 'clus'+str(cluster_nums[index]))
        proposed_alg_with_fp0 = proposed_alg_with_fp.replace('clus', 'clus'+str(cluster_nums_fp[index]))

        print("proposed_alg_nofp0: ", proposed_alg_nofp0)
        print("proposed_alg_with_fp0: ", proposed_alg_with_fp0)

#        print('vals_proj[proposed_alg_nofp0]:', vals_proj[proposed_alg_nofp0])
#        print('vals_proj[proposed_alg_with_fp0]:', vals_proj[proposed_alg_with_fp0])

        vals_cmp = pd.DataFrame()

        vals_nofp = pd.DataFrame()
        vals_with_fp = pd.DataFrame()
        vals_all = pd.DataFrame()

        for alg_index, algorithm in enumerate(algorithms):
            if 'clus' in algorithm:
                if 'xgb' in algorithm:
                    algorithm0 = algorithm.replace('clus', 'clus'+str(cluster_nums_fp[index]))
                else:
                    algorithm0 = algorithm.replace('clus', 'clus'+str(cluster_nums[index]))
            else:
                algorithm0 = algorithm

            stat.append(vals_proj_mean[algorithm0])

            vals_cmp[algorithm] = vals_proj[algorithm0]

            if 'xgb' in algorithm:
                vals_with_fp[alg_complete_names[alg_index]] = vals_proj[algorithm0]
            else:
                vals_nofp[alg_complete_names[alg_index]] = vals_proj[algorithm0]

            vals_all[alg_complete_names[alg_index]] = vals_proj[algorithm0]

            if proposed_alg_nofp0 != algorithm0:
                stat_nofp_wil.append(stats.wilcoxon(vals_proj[proposed_alg_nofp0], vals_proj[algorithm0]).pvalue)
            else:
                stat_nofp_wil.append('---')

            #            print('vals_proj['+algorithm+']):', vals_proj[algorithm])

            if proposed_alg_with_fp0 != algorithm0:
                stat_with_fp_wil.append(stats.wilcoxon(vals_proj[proposed_alg_with_fp0], vals_proj[algorithm0]).pvalue)
            else:
                stat_with_fp_wil.append('---')

        #print("min(to_version): ", min(to_version))
        vals_agg_cmp = vals_agg_cmp.append(vals_cmp.head(min(to_version)+1))

#        print(stat)
        algorithm_stats.append(stat)
        proposed_alg_nofp_wil_stats.append(stat_nofp_wil)
        proposed_alg_with_fp_wil_stats.append(stat_with_fp_wil)

        boxplot_algs(project, results_path, vals_nofp, 'nofp')
        boxplot_algs(project, results_path, vals_with_fp, 'with_fp')
        boxplot_algs(project, results_path, vals_all, 'all')

    algorithm_stats_frame = pd.DataFrame(columns=['Project'] + alg_complete_names, data=algorithm_stats)
    algorithm_stats_frame.to_csv('%s/AlgorithmStatistics.csv' % results_path)

    stat_nofp_wil = ['all']
    stat_with_fp_wil = ['all']

    for algorithm in algorithms:
        if algorithm != proposed_alg_nofp:
            stat_nofp_wil.append(stats.wilcoxon(vals_agg_cmp[proposed_alg_nofp], vals_agg_cmp[algorithm]).pvalue)
        else:
            stat_nofp_wil.append('---')

        if algorithm != proposed_alg_with_fp:
            stat_with_fp_wil.append(stats.wilcoxon(vals_agg_cmp[proposed_alg_with_fp], vals_agg_cmp[algorithm]).pvalue)
        else:
            stat_with_fp_wil.append('---')

    proposed_alg_nofp_wil_stats.append(stat_nofp_wil)
    proposed_alg_with_fp_wil_stats.append(stat_with_fp_wil)

    proposed_alg_nofp_wil_stats_frame = pd.DataFrame(columns=['Project'] + alg_complete_names, data=proposed_alg_nofp_wil_stats)
    proposed_alg_nofp_wil_stats_frame.to_csv('%s/ProposedAlgorithm_NoFp_Wilcoxon.csv' % results_path)

    proposed_alg_with_fp_wil_stats_frame = pd.DataFrame(columns=['Project'] + alg_complete_names, data=proposed_alg_with_fp_wil_stats)
    proposed_alg_with_fp_wil_stats_frame.to_csv('%s/ProposedAlgorithm_WithFp_Wilcoxon.csv' % results_path)


def boxplot_algs(project, results_path, alg_vals, type):
    plt.close('all')
    plot1 = alg_vals.boxplot()
    plot1.set_ylabel('First Fail (%)')
    plot1.set_ylim(0, 100)
    fig1 = plot1.get_figure()
    fig1.autofmt_xdate(rotation=32)
    # fig1.savefig('%s/first_fail/%s.first_fail.boxplot.png' % (results_path, project), bbox_inches='tight')
    fig1.savefig('%s/%s/%s_%s_boxplot.png' % (results_path, type, project, type))
