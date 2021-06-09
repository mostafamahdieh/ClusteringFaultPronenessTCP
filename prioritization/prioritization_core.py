import numpy as np
import pandas as pd
import os.path
import zipfile
import h5py


def rank_evaluation_apfd(ranks, failed_ids):
    ranks_arr = np.array(ranks)
    rank_sum = 0
    for f in failed_ids:
        ind = np.where(ranks_arr == f)
        assert (np.size(ind) == 1)
        rank_sum += (ind[0] + 1)

    m = np.size(failed_ids)
    n = np.size(ranks_arr)
    apfd = 100 * (1 - rank_sum / (m * n) + 1 / (2 * n))
    return apfd


def rank_evaluation_first_fail(ranks, failed_ids):
    #   print("ranks: ", ranks)
    ranks_arr = np.array(ranks)
    min_val = -1
    for f in failed_ids:
        #      print("failed_ids",failed_ids)
        ind = np.where(ranks_arr == f)
        #       print("ind: ", ind)
        assert (np.size(ind) == 1)
        if min_val == -1:
            min_val = ind[0] + 1
        else:
            min_val = min(min_val, ind[0] + 1)

    assert (min_val != -1)
    n = np.size(ranks_arr)
    return min_val / float(n)


def find_row_index(data, value):
    for i in range(0, data.shape[0]):
        if str(data[i]) == value:
            return i
    return -1


# noinspection DuplicatedCode
def read_coverage_data(dataPath):
    h5_file_address = '%s/TestCoverage.h5' % dataPath

    zip_file_name = '%s/TestCoverage.zip' % (dataPath)

    if not os.path.isfile(h5_file_address) and os.path.isfile(zip_file_name):
        print("Unzipping ", zip_file_name)
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(dataPath)

    h5 = h5py.File(h5_file_address, 'r')

    data_size = h5["data"].shape[0]
    test_names = h5["columnTestNamesArray"]
    unit_names = h5["rowMethodNamesArray"]
    unit_num = unit_names.shape[0]
    test_num = test_names.shape[0]
    read_seq = range(0, data_size, 1000)
    coverage = np.zeros(shape=(test_num, unit_num))

    print("Loading coverage from " + h5_file_address + "...")
    for i in range(0, len(read_seq)):
        floor = read_seq[i]
        if i < len(read_seq) - 1:
            top = read_seq[i + 1]
        else:
            top = data_size - 1

        d = h5["data"][floor:top]
        r = h5["row"][floor:top]
        c = h5["column"][floor:top]
        for j in range(floor, top):
            coverage[r[j - floor]][c[j - floor]] = d[j - floor]

    return coverage, test_names, unit_names


def read_failed_tests(data_path, test_names):
    failed_tests_file = ("%s/FailedTests.txt" % data_path)

    with open(failed_tests_file) as f:
        failed_tests = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    failed_tests = [x.strip() for x in failed_tests]

    failed_tests_ids = list()

    for failedTest in failed_tests:
        failed_test_index = find_row_index(test_names, "b'" + failedTest + "'")
        if failed_test_index == -1:
            print("Test %s not found in coverage test names" % failedTest)
        else:
            failed_tests_ids.append(failed_test_index)

    #    print("failedTestsIds: ", failedTestsIds)
    #    print("unit_num: ", unit_num)
    #    print("testNum: ", testNum)

    return failed_tests_ids


def read_bug_prediction_data(data_path):
    bug_prediction_data = pd.read_csv('%s/nn_bugprediction.csv' % data_path, delimiter=',')
    # bugpred = pd.read_csv('%s/bugpred.csv' % dataPath, delimiter=',')
    return bug_prediction_data
