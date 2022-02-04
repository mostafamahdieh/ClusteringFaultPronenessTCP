import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


def matthews_corrcoef_wrapper(y_true, y_pred):
    return matthews_corrcoef(y_true.tolist(), (y_pred > 0.5).tolist())


CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def evaluate_model_prediction(y_true, y_pred, threshold, print_results=True):
    y_label = y_pred > threshold
    mcc_score = matthews_corrcoef(y_true, y_label)
    f_score = f1_score(y_true, y_label)
    auc = roc_auc_score(y_true, y_pred)
    precision, recall = precision_score(y_true, y_label), recall_score(y_true, y_label)
    accuracy = accuracy_score(y_true, y_label)

    if print_results:
        print(f"MCC_score: {mcc_score}")
        print(f"f_score: {f_score}")
        print(f"auc: {auc}")
        print(f"precision: {precision}, recall: {recall}")
        print(f"accuracy: {accuracy}")
        print(f"num positive test samples: {len(y_true[y_true == 1])}, "
              f"num negative test samples: {len(y_true[y_true == 0])}")
    return {"MCC": mcc_score, "f1": f_score, "auc": auc, "precision": precision, "recall": recall, "accuracy": accuracy}


def drop_non_numerical(df):
    return df.drop(
        columns=["before_bugs", "after_bugs", "is_buggy",
                 "LongName", "Name", "Path", "project", "version"]
    ).values


def get_labels(df):
    return df.is_buggy > 0


def get_best_threshold(y_test, y_pred):
    max_score = -1
    fisrt_best_thr = 0
    last_best_thr = 0
    for thr in np.unique(y_pred[y_test]):
        y_label = y_pred > thr-1e-6
        score = matthews_corrcoef(y_test, y_label)
        if score > max_score:
            max_score = score
            fisrt_best_thr = thr
            last_best_thr = thr
        elif math.fabs(max_score-score) < 1e-6:
            # same value
            last_best_thr = thr
        else:
            pass

    return (fisrt_best_thr+last_best_thr)/2.0  # use the threshold with the highest margin


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
        df = pd.read_csv(cache_path)
        print(len(df))
        return df

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
    print(len(merged_df))
    return merged_df


def add_is_changed_column(dataframe, cache_file='features_dataframe.csv'):
    cache_path = os.path.join(CACHE_DIR, cache_file)
    if os.path.exists(cache_path):
        print("reading features dataframe from cached file")
        df = pd.read_csv(cache_path)
        print(len(df))
        return df

    print("generating features dataframe")
    new_dfs = []
    groups = dataframe.groupby(["Name", "Path", "project"])

    for name_path, df in groups:
        dataframe_sorted = df.sort_values(by=["version"]).reset_index(drop=True)
        dataframe_sorted_dropped = dataframe_sorted.drop(
            columns=["before_bugs", "after_bugs", "is_buggy", "LongName", "Name", "Path", "project",
                     "version"]).reset_index(drop=True)
        feature_diff = dataframe_sorted_dropped.diff(-1).fillna(0)

        feature_diff = feature_diff.rename(columns={x: str(x) + "_diff" for x in feature_diff.columns})
        feature_diff['is_changed'] = np.sum(np.abs(feature_diff.values), axis=1) > 0
        result = pd.concat([dataframe_sorted, feature_diff], axis=1)
        new_dfs.append(result)

    feature_df = pd.concat(new_dfs, sort=True)
    feature_df.to_csv(cache_path)
    print(len(feature_df))
    return feature_df
