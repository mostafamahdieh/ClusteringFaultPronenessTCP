import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import uniform
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import ParameterSampler

from utils import get_best_threshold, evaluate_model_prediction, subsample_tarin_data, \
    drop_non_numerical, get_labels, make_dataframes, add_is_changed_column

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

XGB_RESULT_DIR = 'xgb_results'
if not os.path.exists(XGB_RESULT_DIR):
    os.mkdir(XGB_RESULT_DIR)


def run_single_xgboost_fit(index, parameter_sample, x_train, y_train, x_test, y_test):
    clf = xgb.XGBRegressor(**parameter_sample)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    y_train_pred = clf.predict(x_train)

    thr = get_best_threshold(y_test, y_pred)
    y_label = y_pred > thr

    test_score = matthews_corrcoef(y_test, y_label)
    y_train_label = y_train_pred > thr
    train_score = matthews_corrcoef(y_train, y_train_label)
    other_scores = evaluate_model_prediction(y_test, y_pred, thr, print_results=False)

    # print(f"iter: {index}, score: {test_score}, threshold: {thr}, params: {parameter_sample}")
    return {'index': index, 'params': parameter_sample, 'threshold': thr,
            'test_score': test_score, 'train_score': train_score, 'other_scoring_results': other_scores}


def perform_cv(x_train, y_train, x_test, y_test):
    parameters = {'verbosity': [0],
                  'booster': ['gbtree'],
                  'nthread': [2],
                  'objective': ['reg:logistic'],
                  'learning_rate': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1],
                  'max_depth': [2, 3, 4, 5, 8, 10, 16, 20],
                  'subsample': [0.9, 1],
                  'colsample_bytree': uniform(0.5, 0.5),
                  'n_estimators': [5, 10, 20, 40, 80, 100, 150, 200, 300, 400],
                  'lambda': [0, 0.01, 0.1, 1, 2, 5],
                  'gamma': [0, 1],
                  'alpha': [0, 0.01, 0.1, 1, 2, 5],
                  'seed': [1337]}

    param_sampler = ParameterSampler(param_distributions=parameters, random_state=1337, n_iter=2000)
    with Pool(4) as pool:
        results = pool.starmap(run_single_xgboost_fit, [(index, parameter_sample, x_train, y_train, x_test, y_test)
                                                        for index, parameter_sample in enumerate(param_sampler)])
    results_df = pd.DataFrame(results)
    results_df['robustness'] = results_df.params.apply(lambda x: x['n_estimators'] / x['max_depth'])
    results_df = results_df.sort_values(by=['test_score', 'robustness'], ascending=[False, False])

    best_model = results_df.iloc[0]
    print(results_df.test_score.mean())
    print(best_model.other_scoring_results)
    print(len(results_df[results_df.test_score == best_model.test_score]))
    print(len(results_df[(results_df.test_score == best_model.test_score) &
                         (results_df.train_score == best_model.train_score)]))

    best_thr = best_model.threshold
    return best_model, results_df, best_thr


def determine_best_params_and_threshold_cv(dataframe, cache_file='xgb_cv_params_thresholds_cache.pkl',
                                           logs_file='xgb_cv_results.pkl'):
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        print('reading cv params and thresholds from cache file')
        with open(cache_path, 'rb') as cache_fd:
            params_dict, thresholds_dict = pickle.load(cache_fd)
        return params_dict, thresholds_dict

    print('determining the params via cross-validation hyper parameter tuning and its thresholds')
    logs_path = os.path.join(CACHE_DIR, logs_file)
    # if os.path.exists(logs_path):
    #    with open(logs_path, 'rb') as logs_fd:
    #     results_dict = pickle.load(logs_fd)
    #     return results_dict

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()
    thresholds_dict = {}
    params_dict = {}
    results_dict = {}
    for project in projects:
        # Perform cross-validation per project to find the best hyper parameters
        # Last 5 versions of each project is used as validation data and other versions as train data
        # First 5 versions of each project is also used for testing purposes
        print(project)
        last_k_versions = dataframe[dataframe.project == project].version.sort_values().unique()[-5:]
        cv_train_indices = (sampled_dataframe.project != project) & (sampled_dataframe.version > 5)
        cv_test_indices = ((dataframe.project == project) & (dataframe.version.isin(last_k_versions))) | \
                          ((dataframe.project != project) & (dataframe.version <= 5))

        cv_x_train, cv_x_test = drop_non_numerical(sampled_dataframe[cv_train_indices]), drop_non_numerical(
            dataframe[cv_test_indices])
        cv_y_train, cv_y_test = get_labels(sampled_dataframe[cv_train_indices]), get_labels(dataframe[cv_test_indices])

        best_model, results_df, best_thr = perform_cv(cv_x_train, cv_y_train, cv_x_test, cv_y_test)

        params_dict[project] = best_model.params
        thresholds_dict[project] = best_thr
        results_dict[project] = results_df

    with open(logs_path, 'wb') as logs_file_fd:
        pickle.dump(results_dict, logs_file_fd)

    with open(cache_path, 'wb') as cache_fd:
        pickle.dump((params_dict, thresholds_dict), cache_fd)

    return params_dict, thresholds_dict


# TODO: integrate this with the non cached version
def determine_best_params_and_threshold_cv_cache(logs_file='xgb_cv_results.pkl'):
    print('determining the params via cross-validation hyper parameter tuning and its thresholds')
    logs_path = os.path.join(CACHE_DIR, logs_file)
    if os.path.exists(logs_path):
       with open(logs_path, 'rb') as logs_fd:
        results_dict = pickle.load(logs_fd)
    else:
        raise RuntimeError(f"cv logs file not present")

    projects = results_dict.keys()
    thresholds_dict = {}
    params_dict = {}
    for project in projects:
        results = results_dict[project]
        best_models = results[results.test_score == results.iloc[0].test_score]
        best_models['robustness'] = best_models.params.apply(lambda x: x['n_estimators']/x['max_depth'])
        best_model = best_models.sort_values(by=['robustness'], ascending=[False]).iloc[0]
        thresholds_dict[project] = best_model.threshold
        params_dict[project] = best_model.params

    return params_dict, thresholds_dict


def train_model_offline_learning(dataframe):
    # Determine best params and thresholds using cross-validation
    params_dict, thresholds_dict = determine_best_params_and_threshold_cv(dataframe)
    # params_dict, thresholds_dict = determine_best_params_and_threshold_cv_cache()

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()
    result_df = dataframe.copy()
    result_df['xgb_score_offline'] = 0
    result_df['xgb_label_offline'] = False
    result_df['xgb_threshold_offline'] = 0
    scores_list = []
    for project in projects:
        print(project)
        # Train with all of other project data-points and test with this project
        # just to check how better the other method is
        not_last_k_versions = dataframe[dataframe.project == project].version.sort_values().unique()[:-5]
        train_indices = (sampled_dataframe.project != project)
        test_indices = (dataframe.project == project) & (dataframe.version.isin(not_last_k_versions))

        x_train, x_test = drop_non_numerical(sampled_dataframe[train_indices]), drop_non_numerical(
            dataframe[test_indices])
        y_train, y_test = get_labels(sampled_dataframe[train_indices]), get_labels(dataframe[test_indices])

        clf = xgb.XGBRegressor(**params_dict[project])
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        scores = evaluate_model_prediction(y_test, y_pred, thresholds_dict[project])
        scores['project'] = project
        scores['type'] = 'xgb-offline'
        scores['threshold'] = thresholds_dict[project]
        scores_list.append(scores)

        # Re run inference on all of the project data
        result_indices = (dataframe.project == project)
        x_result = drop_non_numerical(dataframe[result_indices])
        y_pred = clf.predict(x_result)
        y_label = y_pred > thresholds_dict[project]

        result_df.loc[result_indices, 'xgb_score_offline'] = y_pred
        result_df.loc[result_indices, 'xgb_label_offline'] = y_label
        result_df.loc[result_indices, 'xgb_threshold_offline'] = thresholds_dict[project]
        result_df[result_indices].copy().reset_index().to_csv(os.path.join(XGB_RESULT_DIR, "xgb-" + project + ".csv"))

    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(os.path.join(XGB_RESULT_DIR, "xgb-offline-scores.csv"))
    result_df.to_csv(os.path.join(XGB_RESULT_DIR, "xgb-offline-results.csv"))
    return result_df


def train_model_online_learning(dataframe):
    # Determine best params and thresholds using cross-validation
    params_dict, thresholds_dict = determine_best_params_and_threshold_cv(dataframe)
    # params_dict, thresholds_dict = determine_best_params_and_threshold_cv_cache()

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()

    result_df = dataframe.copy()
    result_df['xgb_score_online'] = 0
    result_df['xgb_label_online'] = False
    result_df['xgb_threshold_online'] = 0
    scores_list = []
    for project in projects:
        versions = np.array(dataframe[dataframe.project == project].version.sort_values().unique())
        print(project)
        for version in versions:
            # print(f"{project}:#{version}")
            train_indices = (sampled_dataframe.project != project) | (
                    (sampled_dataframe.project == project) & (sampled_dataframe.version > version))
            test_indices = (dataframe.project == project) & (dataframe.version == version)

            x_train, x_test = drop_non_numerical(sampled_dataframe[train_indices]), drop_non_numerical(
                dataframe[test_indices])
            y_train, y_test = get_labels(sampled_dataframe[train_indices]), get_labels(dataframe[test_indices])

            clf = xgb.XGBRegressor(**params_dict[project])
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            y_label = y_pred > thresholds_dict[project]
            result_df.loc[test_indices, 'xgb_score_online'] = y_pred
            result_df.loc[test_indices, 'xgb_label_online'] = y_label
            result_df.loc[test_indices, 'xgb_threshold_online'] = thresholds_dict[project]

        not_last_k_versions = result_df[result_df.project == project].version.sort_values().unique()[:-5]
        test_indices = (result_df.project == project) & (result_df.version.isin(not_last_k_versions))
        y_test = get_labels(result_df[test_indices])
        y_pred = result_df[test_indices].xgb_score_online

        scores = evaluate_model_prediction(y_test, y_pred, thresholds_dict[project])
        scores['project'] = project
        scores['type'] = 'xgb-online'
        scores['threshold'] = thresholds_dict[project]
        scores_list.append(scores)

        result_indices = result_df.project == project
        result_df[result_indices].copy().reset_index() \
            .to_csv(os.path.join(XGB_RESULT_DIR, "xgb-online-" + project + ".csv"))

    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(os.path.join(XGB_RESULT_DIR, "xgb-online-scores.csv"))
    result_df.to_csv(os.path.join(XGB_RESULT_DIR, "xgb-online-results.csv"))
    return result_df


if __name__ == '__main__':
    df = make_dataframes('features_data')
    df = add_is_changed_column(df)
    train_model_offline_learning(df)
    train_model_online_learning(df)
