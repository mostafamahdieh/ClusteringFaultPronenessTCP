import numpy as np

eps = 1.0e-8


def total_prioritization(coverage, unit_prob):
    total_weighted_coverage = np.matmul(coverage, unit_prob)
    return np.flip(np.argsort(total_weighted_coverage))


def add_all_remaining_tests(tcp_ordering, test_num, test_used):
    for test_ind in range(0, test_num):
        if not test_used[test_ind]:
            test_used[test_ind] = True
            tcp_ordering.append(test_ind)


def additional_prioritization(coverage, unit_prob):
    test_num = coverage.shape[0]

    additional_sum_ranks = np.zeros((test_num,))
    test_used = [False] * test_num  # none of the tests are used at the beginning

    total_weighted_coverage = np.matmul(coverage, unit_prob)
    additional_weighted_coverage = np.array(total_weighted_coverage)
    #    print("additional_weighted_coverage: ", additional_weighted_coverage, "\n")

    # additional_weighted_coverage = np.zeros((testNum, )) # the weighted coverage of each of the test cases
    # for u in range(0,unit_num):
    #    additional_weighted_coverage = unitProb[u]*coverage[:,u]

    for rank in range(0, test_num):
        #        print(sprintf("additional (%d/%d)", rank, testNum))
        best_additional_weighted_coverage = -1
        best_total_weighted_coverage = -1
        best_test = None

        for candidate_test in range(0, test_num):
            if test_used[candidate_test]:
                continue

            if (additional_weighted_coverage[candidate_test] > best_additional_weighted_coverage + eps or
                    (abs(additional_weighted_coverage[candidate_test] - best_additional_weighted_coverage) <= eps
                     and total_weighted_coverage[candidate_test] > best_total_weighted_coverage)):
                best_test = candidate_test
                best_additional_weighted_coverage = additional_weighted_coverage[candidate_test]
                best_total_weighted_coverage = total_weighted_coverage[candidate_test]

        #        print("bestTest: ", bestTest)
        #        print("best_additional_weighted_coverage: ", best_additional_weighted_coverage)
        #        print("best_total_weighted_coverage: ", best_total_weighted_coverage)
        assert best_test is not None
        additional_sum_ranks[rank] = best_test
        test_used[best_test] = True

        new_unit_prob = np.maximum(unit_prob - coverage[best_test, :], 0)

        #     another way of zeroing the values
        #     unitProb[additional_weighted_coverage>0]=0
        if np.sum(unit_prob - new_unit_prob) > eps:
            # ignore changing additional_weighted_coverage if unit probs have not changed
            additional_weighted_coverage -= np.matmul(coverage, (unit_prob - new_unit_prob))

        unit_prob = new_unit_prob
    #        print("new UnitProb: ", newUnitProb, "\n")
    return additional_sum_ranks


def is_remaining_coverage_zero(test_used, total_coverage):
    for test_ind in range(0, len(total_coverage)):
        if not test_used[test_ind] and total_coverage[test_ind] > eps:
            return False
    return True


def max_prioritization_std(coverage, unit_fp):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    max_dp = np.max(np.multiply(coverage >= 0.05, unit_fp), axis=1)
    assert (len(max_dp) == test_num)

    return np.flip(np.argsort(max_dp))


def additional_prioritization_std(coverage, unit_fp, additional_style):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    test_used = [False] * test_num  # none of the tests are used at the beginning
    unit_coverage = np.ones((unit_num,))

    total_weighted_coverage = np.matmul(coverage, unit_fp)
    additional_weighted_coverage = np.array(total_weighted_coverage)
    #    print("additional_weighted_coverage: ", additional_weighted_coverage, "\n")
    tcp_ordering = []

    while len(tcp_ordering) < test_num:
        if is_remaining_coverage_zero(test_used, total_weighted_coverage):
            # add all remaining test cases to the ordering
            add_all_remaining_tests(tcp_ordering, test_num, test_used)
            break

        best_coverage = -1
        best_test = None

        for candidate_test in range(0, test_num):
            if not test_used[candidate_test] and additional_weighted_coverage[candidate_test] > best_coverage:
                best_test = candidate_test
                best_coverage = additional_weighted_coverage[candidate_test]

        assert best_coverage != -1, "Didn't find any test (this must not happen)!"
        assert best_coverage > -eps, "Coverage (" + str(best_coverage) + ") must not be negative!"

        if best_coverage > eps:
            if additional_style == 'decrease':
                new_unit_coverage = np.maximum(unit_coverage - coverage[best_test, :], 0)
            elif additional_style == 'zero':
                new_unit_coverage = unit_coverage.copy()
                new_unit_coverage[coverage[best_test, :] > eps] = 0
            else:
                raise Exception("Bad value for additional_style: " + additional_style)
            coverage_diff = (unit_coverage - new_unit_coverage)
            additional_weighted_coverage -= np.matmul(coverage, np.multiply(coverage_diff, unit_fp))
            unit_coverage = new_unit_coverage
            test_used[best_test] = True
            tcp_ordering.append(best_test)
        else:
            additional_weighted_coverage = np.array(total_weighted_coverage)
            unit_coverage = np.ones((unit_num,))
    #        print("new UnitProb: ", newUnitProb, "\n")

    print("len(tcp_ordering): ", len(tcp_ordering))
    print("test_num: ", test_num)
    assert len(tcp_ordering) == test_num

    return tcp_ordering


def max_prioritization(coverage, unit_prob):
    test_num = coverage.shape[0]

    max_prob_covered = np.zeros((test_num,))

    for t in range(test_num):
        max_prob_covered[t] = np.max(coverage[t, :] * unit_prob)

    sorted_by_max_indexes = np.flip(np.argsort(max_prob_covered))
    return sorted_by_max_indexes


def max_normalized_prioritization(coverage, unit_prob):
    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    max_prob_covered = np.zeros((test_num,))
    unit_coverage = np.zeros((unit_num,))

    for u in range(unit_num):
        unit_coverage[u] = np.sum(coverage[:, u])

    unit_prob_normalized = unit_prob * 1 / (unit_coverage + 1)

    for t in range(test_num):
        max_prob_covered[t] = np.max(coverage[t, :] * unit_prob_normalized)

    sorted_by_max_indexes = np.flip(np.argsort(max_prob_covered))
    return sorted_by_max_indexes


def normalized_total_prioritization(coverage, unit_prob):
    unit_prob_normalized = unit_prob_coverage_normalized(coverage, unit_prob)
    return total_prioritization(coverage, unit_prob_normalized)


def normalized_additional_prioritization(coverage, unit_prob):
    unit_prob_normalized = unit_prob_coverage_normalized(coverage, unit_prob)
    return additional_prioritization(coverage, unit_prob_normalized)


def unit_prob_coverage_normalized(coverage, unit_prob):
    unit_num = coverage.shape[1]
    unit_coverage = np.zeros((unit_num,))
    for u in range(unit_num):
        unit_coverage[u] = np.sum(coverage[:, u])
    unit_prob_normalized = unit_prob * 1 / (unit_coverage + 1)
    return unit_prob_normalized
