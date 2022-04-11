from sklearn.metrics import euclidean_distances

from prioritization.prioritization_art import art_create_candidate_set
from prioritization.prioritization_manager import run_standard2_prioritization, run_art_prioritization

projects = ['Closure']
from_version = [1, 1, 1, 1, 1]
to_version = [1]

for index, project in enumerate(projects):
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_art_prioritization(project, version_number, 'art_rnd.csv', art_create_candidate_set, euclidean_distances, 'art_rnd', 1, 'precomputed')
        print()
