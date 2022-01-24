import os
import pickle

import pandas as pd
import numpy as np
from scipy.stats import uniform, gamma
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, make_scorer, f1_score, accuracy_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, ParameterSampler


def matthews_corrcoef_wrapper(y_true, y_pred):
    return matthews_corrcoef(y_true.tolist(), (y_pred > 0.5).tolist())


CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def evaluate_model_prediction(y_true, y_pred):
    mcc_score = matthews_corrcoef(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    f_score_weighted = f1_score(y_true, y_pred, average='weighted')
    precision, recall = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"MCC_score: {mcc_score}")
    print(f"f_score: {f_score}")
    print(f"f_score_weighted: {f_score_weighted}")
    print(f"precision: {precision}, recall: {recall}")
    print(f"accuracy: {accuracy}")
    print(f"num positive test samples: {len(y_true[y_true == 1])}, "
          f"num negative test samples: {len(y_true[y_true == 0])}")


def drop_non_numerical(df):
    return df.drop(
        columns=["before_bugs", "after_bugs", "is_buggy",
                 "LongName", "Name", "Path", "project", "version"]
    ).values


def get_labels(df):
    return df.is_buggy > 0


def get_best_threshold(y_test, y_pred):
    max_score = 0
    best_thr = 0
    for thr in np.arange(1e-4, 1, 1e-2):
        y_label = y_pred > thr
        score = matthews_corrcoef(y_test, y_label)
        if score > max_score:
            max_score = score
            best_thr = thr
    return best_thr


def subsample_tarin_data(dataframe, negative_sample_frac=0.05):
    sampled_df = pd.concat([dataframe[dataframe.is_buggy == 0].sample(frac=negative_sample_frac),
                            dataframe[dataframe.is_buggy != 0]], ignore_index=True)
    print(f"Total number of samples: {len(dataframe)}\n"
          f"Sampled dataset size: {len(sampled_df)}\n"
          f"Original negative sample size: {len(dataframe[dataframe.is_buggy == 0])}\n"
          f"Original positive sample size: {len(dataframe[dataframe.is_buggy != 0])}\n"
          f"Sampled negative sample size: {len(sampled_df[sampled_df.is_buggy == 0])}\n"
          f"Sampled positive sample size {len(sampled_df[sampled_df.is_buggy != 0])}")
    return sampled_df


def make_dataframes(data_directory, cache_file='merged_dataframes.csv'):
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    all_dfs = []
    subdirs = [x[0] for x in os.walk(data_directory) if "Features.csv" in x[2]]

    for directory in sorted(subdirs):
        csv_file = os.path.join(directory, "Features.csv")
        df = pd.read_csv(csv_file)
        df['version'] = float(directory.split("/")[-1])
        df['project'] = directory.split("/")[-2]
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs)
    merged_df.to_csv(cache_path)
    return merged_df


def add_is_changed_column(dataframe, cache_file='features_dataframe.csv'):
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    new_dfs = []
    groups = dataframe.groupby(["Name", "Path", "project"])

    for name_path, df in groups:
        dataframe_sorted = df.sort_values(by=["version"]).reset_index(drop=True)
        dataframe_sorted_dropped = dataframe_sorted.drop(
            columns=["before_bugs", "after_bugs", "is_buggy", "LongName", "Name", "Path", "project",
                     "version"]).reset_index(drop=True)
        if len(dataframe_sorted) == 0:
            print("Assertion failed")
            break
        else:
            feature_diff = (dataframe_sorted_dropped - dataframe_sorted_dropped.shift(-1, fill_value=0))
            feature_diff.iloc[-1, :] = 0

        feature_diff['is_changed'] = np.sum(np.abs(feature_diff.values), axis=1) > 0
        feature_diff = feature_diff.rename(columns={x: x + "_diff" for x in feature_diff.columns})
        result = pd.concat([dataframe_sorted, feature_diff], axis=1)
        new_dfs.append(result)

    feature_df = pd.concat(new_dfs)
    feature_df.to_csv(cache_path)
    return feature_df


def perform_cv_(x_train, y_train, x_test, y_test):
    parameters = {'verbosity': [0],
                  'booster': ['gbtree'],
                  'nthread': [2],
                  'objective': ['reg:logistic'],
                  'learning_rate': gamma(1.0),
                  'max_depth': [2, 3, 4, 5, 8, 10, 16, 20],
                  'subsample': [1],
                  'colsample_bytree': uniform(0.1, 0.90),
                  'n_estimators': [5, 10, 20, 40, 80, 100, 160],
                  'lambda': gamma(2.0),
                  'gamma': uniform(0, 1),
                  'alpha': gamma(2.0),
                  'seed': [1337]}

    clf = RandomizedSearchCV(estimator=xgb.XGBRegressor(), param_distributions=parameters,
                             n_iter=200, n_jobs=8, cv=StratifiedKFold(n_splits=2, random_state=1337, shuffle=True),
                             scoring=make_scorer(matthews_corrcoef_wrapper, greater_is_better=True,
                                                 needs_threshold=False, needs_proba=False),
                             verbose=2)
    clf.fit(x_train, y_train)
    print(clf.cv_results_['mean_test_score'])

    y_pred = clf.predict(x_test)
    print(evaluate_model_prediction(y_test, y_pred > 0.5))

    best_thr = get_best_threshold(y_test, y_pred)
    return clf, best_thr


def perform_cv(x_train, y_train, x_test, y_test):
    parameters = {'verbosity': [0],
                  'booster': ['gbtree'],
                  'nthread': [2],
                  'objective': ['reg:logistic'],
                  'learning_rate': gamma(1.0),
                  'max_depth': [2, 3, 4, 5, 8, 10, 16, 20],
                  'subsample': [1],
                  'colsample_bytree': uniform(0.1, 0.90),
                  'n_estimators': [5, 10, 20, 40, 80, 100, 160],
                  'lambda': gamma(2.0),
                  'gamma': uniform(0, 1),
                  'alpha': gamma(2.0),
                  'seed': [1337]}

    #param_list = ParameterSampler(param_distributions=parameters, n_iter=400, random_state=1337)

    clf = RandomizedSearchCV(estimator=xgb.XGBRegressor(), param_distributions=parameters,
                             n_iter=200, n_jobs=8, cv=StratifiedKFold(n_splits=2, random_state=1337, shuffle=True),
                             scoring=make_scorer(matthews_corrcoef_wrapper, greater_is_better=True,
                                                 needs_threshold=False, needs_proba=False),
                             verbose=2)
    clf.fit(x_train, y_train)
    print(clf.cv_results_['mean_test_score'])

    y_pred = clf.predict(x_test)
    print(evaluate_model_prediction(y_test, y_pred > 0.5))

    best_thr = get_best_threshold(y_test, y_pred)
    return clf, best_thr


def determine_best_params_and_threshold_cv(dataframe, cache_file='cv_params_thresholds_cache.pkl',
                                           logs_file='cv_results.pkl'):
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        params_dict, thresholds_dic = pickle.load(cache_path)
        return params_dict, thresholds_dic

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()
    thresholds_dict = {}
    params_dict = {}
    models_dict = {}
    for project in projects:
        # Perform cross-validation per project to find the best hyper parameters
        # Last 5 versions of each project is used as validation data and other versions as train data
        # First 5 versions of each project is also used for testing purposes
        print(project)
        last_k_versions = dataframe[project].version.sort_values().unique()[:-5]
        cv_train_indices = (sampled_dataframe.project != project) & (sampled_dataframe.version > 10)
        cv_test_indices = ((dataframe.project != project) & (dataframe.version <= 5)) | \
                          ((dataframe.project == project) & (dataframe.version >= last_k_versions))

        cv_x_train, cv_x_test = drop_non_numerical(sampled_dataframe[cv_train_indices]), drop_non_numerical(
            dataframe[cv_test_indices])
        cv_y_train, cv_y_test = get_labels(sampled_dataframe[cv_train_indices]), get_labels(dataframe[cv_test_indices])

        clf, thr = perform_cv(cv_x_train, cv_y_train, cv_x_test, cv_y_test)
        params_dict[project] = clf.best_params_
        thresholds_dict[project] = thr
        models_dict[project] = clf

    with open(cache_path, 'wb') as cache_fd:
        pickle.dump((params_dict, thresholds_dict), cache_fd)

    with open(logs_file, 'wb') as logs_file_fd:
        pickle.dump(models_dict, logs_file_fd)

    return params_dict, thresholds_dict


def train_model_offline_learning(dataframe):
    # Determine best params and thresholds using cross-validation
    params_dict, thresholds_dict = determine_best_params_and_threshold_cv(dataframe)

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()
    dataframe['xgb_score_offline'] = 0
    for project in projects:
        # Train with all of other project data-points and test with this project
        # just to check how better the other method is
        train_indices = (sampled_dataframe.project != project)
        test_indices = (dataframe.project == project)

        x_train, x_test = drop_non_numerical(sampled_dataframe[train_indices]), drop_non_numerical(
            dataframe[test_indices])
        y_train, y_test = get_labels(sampled_dataframe[train_indices]), get_labels(dataframe[test_indices])

        clf = xgb.XGBRegressor(params_dict[project])
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        y_label = y_pred > thresholds_dict[project]
        evaluate_model_prediction(y_test, y_label)
        dataframe[test_indices]['xgb_score_offline'] = y_pred


def train_model_online_learning(dataframe):
    # Determine best params and thresholds using cross-validation
    params_dict, thresholds_dict = determine_best_params_and_threshold_cv(dataframe)

    sampled_dataframe = subsample_tarin_data(dataframe, 0.05)

    projects = dataframe.project.unique()
    dataframe['xgb_score_online'] = 0
    for project in projects:
        versions = np.array(dataframe[dataframe.project == project].version.sort_values().unique())
        for index in range(1, len(versions) + 1):
            train_indices = (dataframe.project != project) | (
                    (dataframe.project == project) & (dataframe.version > index))
            test_indices = (dataframe.project == project) & (dataframe.version == index)

            x_train, x_test = drop_non_numerical(dataframe[train_indices]), drop_non_numerical(dataframe[test_indices])
            y_train, y_test = get_labels(dataframe[train_indices]), get_labels(dataframe[test_indices])

            clf = xgb.XGBRegressor(params_dict[project])
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)

            dataframe[test_indices]['xgb_score_online'] = y_pred

        # Evaluate final results of the online learning process
        test_indices = (dataframe.project == project)

        y_test = get_labels(dataframe[test_indices])
        y_pred = dataframe[test_indices]['xgb_score_online']
        y_label = y_pred > thresholds_dict[project]

        evaluate_model_prediction(y_test, y_label)


if __name__ == '__main__':
    df = make_dataframes('features_data')
    df = add_is_changed_column(df)
    train_model_offline_learning(df)
