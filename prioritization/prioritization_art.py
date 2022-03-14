import numpy as np
from random import random, randrange, sample


def art_create_candidate_set(coverage, remaining_tests):
    remaining_tests = remaining_tests.copy()
    eps = 1.0e-8

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    selected_tests = set()

    unit_coverage = np.ones((unit_num,))
    unit = np.ones((unit_num,))

    total_weighted_coverage = np.matmul(coverage, unit)
    additional_weighted_coverage = np.array(total_weighted_coverage)
    #    print("additional_weighted_coverage: ", additional_weighted_coverage, "\n")
    candidate_set = set()

    while len(remaining_tests) > 0:
        test = sample(remaining_tests, 1)[0]
        test_coverage = additional_weighted_coverage[test]

        if len(candidate_set) == 0 or test_coverage > eps:
            new_unit_coverage = np.maximum(unit_coverage - coverage[test, :], 0)
            coverage_diff = (unit_coverage - new_unit_coverage)
            additional_weighted_coverage -= np.matmul(coverage, np.multiply(coverage_diff, unit))
            unit_coverage = new_unit_coverage
            candidate_set.add(test)
            remaining_tests.remove(test)
        else:
            break

    return candidate_set


def art_create_candidate_set2(coverage, remaining_tests):
    sample_size = 50
    if len(remaining_tests) <= sample_size:
        return remaining_tests.copy()
    else:
        return sample(remaining_tests, 50)


def art_tcp(coverage, distance_function, cand_set_function):
    eps = 1.0e-8

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    remaining_tests = set(list(range(0, test_num)))

    first_test = randrange(test_num)
    remaining_tests.remove(first_test)
    added_set_cov = coverage[[first_test]]
    prioritized = list([first_test])

    while len(remaining_tests) > 0:
        candidate_set = cand_set_function(coverage, remaining_tests)

        while len(candidate_set) > 0:
            candidate_set_list = list(candidate_set)
            if len(candidate_set) == 1:
                best_test = candidate_set_list[0]
            else:
                candidate_set_matrix = coverage[candidate_set_list]
                dist = distance_function(added_set_cov, candidate_set_matrix)

                assert(dist.shape[0] == len(prioritized))
                assert(dist.shape[1] == len(candidate_set))

                dist_min = dist.min(axis=0)
                assert(dist_min.shape[0] == len(candidate_set))

                best_test = candidate_set_list[np.argmax(dist_min)]

            assert(best_test in remaining_tests)
            assert(best_test in candidate_set)

            prioritized.append(best_test)
            candidate_set.remove(best_test)
            remaining_tests.remove(best_test)

            added_set_cov = np.append(added_set_cov, coverage[[best_test]], axis=0)
        #print('len(prioritized): ', len(prioritized))

    assert(len(prioritized) == test_num)
    return prioritized


def std_tuple(x, y):
    return (min(x, y), max(x, y))


def art_tcp_cache(coverage, distance_function, cand_set_function):
    eps = 1.0e-8
    inf = 1.0e100

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    remaining_tests = set(list(range(0, test_num)))

    first_test = randrange(test_num)
    remaining_tests.remove(first_test)
    prioritized = list([first_test])

    dist_dict = dict()
    while len(remaining_tests) > 0:
        candidate_set = cand_set_function(coverage, remaining_tests)

        while len(candidate_set) > 0:
            candidate_set_list = list(candidate_set)
            if len(candidate_set) == 1:
                best_candidate_test = candidate_set_list[0]
            else:
                # add dists to dist_dict
                # candidate_set_matrix = coverage[candidate_set_list]
                # euclidean_distances(added_set_cov, candidate_set_matrix)

                # dist_min = dist.min(axis=0)
                # assert(dist_min.shape[0] == len(candidate_set))
                # best_test = candidate_set_list[np.argmax(dist_min)]
                best_candidate_dist = -1
                best_candidate_test = None

                last_prioritized = prioritized[len(prioritized)-1]
                new_dists = distance_function(coverage[[last_prioritized]], coverage[candidate_set_list])
                for (ind, c) in enumerate(candidate_set_list):
                    dist_dict[std_tuple(last_prioritized, c)] = new_dists[0][ind]

                for candidate_test in candidate_set_list:
                    candidate_dist = inf
                    for src in prioritized:

                        if not (src, candidate_test) in dist_dict:
                            d = distance_function(coverage[[src]], coverage[[candidate_test]])[0][0]
                            dist_dict[std_tuple(src, candidate_test)] = d
                        else:
                            d = dist_dict[std_tuple(src, candidate_test)]

                        candidate_dist = min(candidate_dist, d)

                        if candidate_dist <= best_candidate_dist:
                            break

                    if candidate_dist > best_candidate_dist:
                        best_candidate_test = candidate_test
                        best_candidate_dist = candidate_dist

            # print("best_candidate_test: ", best_candidate_test)

            assert(best_candidate_test in remaining_tests)
            assert(best_candidate_test in candidate_set)

            prioritized.append(best_candidate_test)
            candidate_set.remove(best_candidate_test)
            remaining_tests.remove(best_candidate_test)

        #print('len(prioritized): ', len(prioritized))

    assert(len(prioritized) == test_num)
    return prioritized

def art_tcp_distance(coverage, distance, cand_set_function):
    eps = 1.0e-8

    test_num = coverage.shape[0]
    unit_num = coverage.shape[1]

    remaining_tests = set(list(range(0, test_num)))

    first_test = randrange(test_num)
    remaining_tests.remove(first_test)
    added_set_distance = distance[[first_test]]
    prioritized = list([first_test])

    while len(remaining_tests) > 0:
        candidate_set = cand_set_function(coverage, remaining_tests)

        while len(candidate_set) > 0:
            candidate_set_list = list(candidate_set)
            if len(candidate_set) == 1:
                best_test = candidate_set_list[0]
            else:
                #candidate_set_matrix = coverage[candidate_set_list]
                #dist = distance_function(added_set_cov, candidate_set_matrix)

                dist_min = added_set_distance[:, candidate_set_list].min(axis=0)
                assert(dist_min.shape[0] == len(candidate_set))

                best_test = candidate_set_list[np.argmax(dist_min)]

            assert(best_test in remaining_tests)
            assert(best_test in candidate_set)

            prioritized.append(best_test)
            candidate_set.remove(best_test)
            remaining_tests.remove(best_test)

            added_set_distance = np.append(added_set_distance, distance[[best_test]], axis=0)
        print('len(prioritized): ', len(prioritized))

    assert(len(prioritized) == test_num)
    return prioritized
