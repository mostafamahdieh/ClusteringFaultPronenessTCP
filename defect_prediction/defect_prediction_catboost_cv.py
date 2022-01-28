import os

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef

from utils import get_best_threshold, evaluate_model_prediction, subsample_tarin_data, \
    drop_non_numerical, get_labels, make_dataframes, add_is_changed_column

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

CB_RESULT_DIR = 'cb_results'
if not os.path.exists(CB_RESULT_DIR):
    os.mkdir(CB_RESULT_DIR)


def run_single_catboost_fit(x_train, y_train, x_test, y_test):
    clf = CatBoostClassifier(verbose=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)[:, 1]

    thr = get_best_threshold(y_test, y_pred)
    y_label = y_pred > thr
    score = matthews_corrcoef(y_test, y_label)
    other_scores = evaluate_model_prediction(y_test, y_label, print_results=False)

    print(score, other_scores)

    return y_pred, y_label, thr


def determine_best_params_and_threshold_cv(dataframe):
    print('determining the params via cross-validation hyper parameter tuning and its thresholds')
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
        train_indices = (sampled_dataframe.project != project) & (sampled_dataframe.version > 5)
        test_indices = ((dataframe.project == project) & (dataframe.version.isin(last_k_versions))) | \
                       ((dataframe.project != project) & (dataframe.version <= 5))

        cv_x_train, cv_x_test = drop_non_numerical(sampled_dataframe[train_indices]), drop_non_numerical(
            dataframe[test_indices])
        cv_y_train, cv_y_test = get_labels(sampled_dataframe[train_indices]), get_labels(dataframe[test_indices])

        _, _, best_thr = run_single_catboost_fit(cv_x_train, cv_y_train, cv_x_test, cv_y_test)

        thresholds_dict[project] = best_thr
    return thresholds_dict


def train_model_offline_learning(dataframe):
    # Determine best params and thresholds using cross-validation
    thresholds_dict = determine_best_params_and_threshold_cv(dataframe)

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()
    result_df = dataframe.copy()
    result_df['cb_score_offline'] = 0
    result_df['cb_label_offline'] = False
    result_df['cb_threshold_offline'] = 0
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

        clf = CatBoostClassifier(verbose=0)
        clf.fit(x_train, y_train)

        y_pred = clf.predict_proba(x_test)[:, 1]
        y_label = y_pred > thresholds_dict[project]
        evaluate_model_prediction(y_test, y_label)

        # Re run inference on all of the project data
        result_indices = (dataframe.project == project)
        x_result = drop_non_numerical(dataframe[result_indices])
        y_pred = clf.predict_proba(x_result)[:, 1]
        y_label = y_pred > thresholds_dict[project]

        result_df.loc[result_indices, 'cb_score_offline'] = y_pred
        result_df.loc[result_indices, 'cb_label_offline'] = y_label
        result_df.loc[result_indices, 'cb_threshold_offline'] = thresholds_dict[project]
        result_df[result_indices].copy().reset_index().to_csv(os.path.join(CB_RESULT_DIR, "cb-" + project + ".csv"))

    result_df.to_csv(os.path.join(CB_RESULT_DIR, "cb-results.csv"))
    return result_df


def train_model_online_learning(dataframe):
    # Determine best params and thresholds using cross-validation
    thresholds_dict = determine_best_params_and_threshold_cv(dataframe)

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()

    result_df = dataframe.copy()
    result_df['cb_score_online'] = 0
    result_df['cb_label_online'] = False
    result_df['cb_threshold_online'] = 0
    for project in projects:
        versions = np.array(dataframe[dataframe.project == project].version.sort_values().unique())
        print(project)

        last_k_versions = dataframe[dataframe.project == project].version.sort_values().unique()[-5:]
        train_indices = (sampled_dataframe.project != project) | (sampled_dataframe.project == project)
        validation_indices = (dataframe.project == project) & (dataframe.version.isin(last_k_versions))

        x_train, x_val = drop_non_numerical(sampled_dataframe[train_indices]), drop_non_numerical(
            dataframe[validation_indices])
        y_train, y_val = get_labels(sampled_dataframe[train_indices]), get_labels(dataframe[validation_indices])

        y_pred, _, _ = run_single_catboost_fit(x_train, y_train, x_val, y_val)
        y_label = y_pred > thresholds_dict[project]
        result_df.loc[validation_indices, 'cb_score_online'] = y_pred
        result_df.loc[validation_indices, 'cb_label_online'] = y_label
        result_df.loc[validation_indices, 'cb_threshold_online'] = thresholds_dict[project]

        for version in versions[:-5]:
            # print(f"{project}:#{version}")
            train_indices = (sampled_dataframe.project != project) | (
                    (sampled_dataframe.project == project) & (sampled_dataframe.version > version))
            test_indices = (dataframe.project == project) & (dataframe.version == version)

            x_train, x_test = drop_non_numerical(sampled_dataframe[train_indices]), drop_non_numerical(
                dataframe[test_indices])
            y_train, y_test = get_labels(sampled_dataframe[train_indices]), get_labels(dataframe[test_indices])

            clf = CatBoostClassifier(verbose=0)
            clf.fit(x_train, y_train)

            y_pred = clf.predict_proba(x_test)[:, 1]
            y_label = y_pred > thresholds_dict[project]

            result_df.loc[test_indices, 'cb_score_online'] = y_pred
            result_df.loc[test_indices, 'cb_label_online'] = y_label
            result_df.loc[test_indices, 'cb_threshold_online'] = thresholds_dict[project]

        not_last_k_versions = result_df[result_df.project == project].version.sort_values().unique()[:-5]
        test_indices = (result_df.project == project) & (result_df.version.isin(not_last_k_versions))
        y_test = get_labels(result_df[test_indices])
        y_label = result_df[test_indices].xgb_label_online
        evaluate_model_prediction(y_test, y_label)

        result_indices = result_df.project == project
        result_df[result_indices].copy().reset_index() \
            .to_csv(os.path.join(CB_RESULT_DIR, "cb-online-" + project + ".csv"))

    result_df.to_csv(os.path.join(CB_RESULT_DIR, "cb-online-results.csv"))
    return result_df


if __name__ == '__main__':
    df = make_dataframes('features_data')
    df = add_is_changed_column(df)
    train_model_offline_learning(df)
    train_model_online_learning(df)
