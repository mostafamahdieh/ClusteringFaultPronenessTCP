import pandas as pd
from prioritization.prioritization_manager import run_standard2_prioritization, run_art_prioritization

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
from_version = [1, 1, 1, 1, 1]
to_version = [26, 133, 65, 106, 26]

for index, project in enumerate(projects):
    for version_number in range(from_version[index], to_version[index] + 1):
        print("* Version %d" % version_number)
        run_art_prioritization(project, version_number, 'art.csv')
        print()
