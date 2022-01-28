import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score


def matthews_corrcoef_wrapper(y_true, y_pred):
    return matthews_corrcoef(y_true.tolist(), (y_pred > 0.5).tolist())


CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def evaluate_model_prediction(y_true, y_pred, print_results=True):
    mcc_score = matthews_corrcoef(y_true, y_pred)
    f_score = f1_score(y_true, y_pred)
    f_score_weighted = f1_score(y_true, y_pred, average='weighted')
    precision, recall = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    if print_results:
        print(f"MCC_score: {mcc_score}")
        print(f"f_score: {f_score}")
        print(f"f_score_weighted: {f_score_weighted}")
        print(f"precision: {precision}, recall: {recall}")
        print(f"accuracy: {accuracy}")
        print(f"num positive test samples: {len(y_true[y_true == 1])}, "
              f"num negative test samples: {len(y_true[y_true == 0])}")
    return mcc_score, f_score, f_score_weighted, precision, recall, accuracy


def drop_non_numerical(df):
    return df.drop(
        columns=["before_bugs", "after_bugs", "is_buggy",
                 "LongName", "Name", "Path", "project", "version"]
    ).values


def get_labels(df):
    return df.is_buggy > 0


def get_best_threshold(y_test, y_pred):
    max_score = -1
    best_thr = 0
    for thr in np.arange(1e-4, 1, 1e-2):
        y_label = y_pred > thr
        score = matthews_corrcoef(y_test, y_label)
        if score > max_score:
            max_score = score
            best_thr = thr
    return best_thr


def subsample_tarin_data(dataframe, negative_sample_frac=0.05):
    sampled_df = pd.concat([dataframe[dataframe.is_buggy == 0].sample(frac=negative_sample_frac, random_state=1337),
                            dataframe[dataframe.is_buggy != 0]], ignore_index=True)
    print(len(dataframe.columns), len(sampled_df.columns))
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
        print("reading merged dataframe from cached file")
        return pd.read_csv(cache_path)

    print("generating merged dataframe")
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
        print("reading features dataframe from cached file")
        return pd.read_csv(cache_path)

    print("generating features dataframe")
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
