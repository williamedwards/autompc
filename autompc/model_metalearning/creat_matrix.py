import argparse
from collections import defaultdict
import json
import os
import pickle

import pandas as pd
import sklearn.externals.joblib

from meta_utils import meta_data

# parser = argparse.ArgumentParser()
# parser.add_argument('--working-directory', type=str, required=True)
# parser.add_argument('--save-to', type=str, required=True)
# args = parser.parse_args()
# working_directory = args.working_directory
working_directory = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/temp.json'
save_to = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning'

def read_runs_for_task_id(task_id):
    results = defaultdict(dict)
    try:
        task_id = int(task_id)
    except ValueError as e:
        return None
    task_dir = os.path.join(working_directory, str(task_id))
    for run_file in os.listdir(task_dir):
        if run_file.endswith('.lock'):
            continue
        filename = run_file.split('.')[0]
        strategy, config_id = filename.split('_')
        with open(os.path.join(task_dir, run_file)) as fh:
            result = json.load(fh)
        results[strategy][config_id] = result
        #results_by_strategy[strategy][task_id][config_id] = result
    return results

# Set up tasks directories 
task_directories = list(reversed(sorted(os.listdir(working_directory))))
print('Found %d directories' % len(task_directories))
expected_tasks = automl_metadata_task_ids
task_directories = [task for task in task_directories if int(task) in expected_tasks]
print('Left with %d directories, expected %d' % (len(task_directories), len(expected_tasks)))

rvals = sklearn.externals.joblib.Parallel(
    n_jobs=10, verbose=20, backend='multiprocessing', batch_size=1,
)(
    sklearn.externals.joblib.delayed(read_runs_for_task_id)(task_id)
    for task_id in task_directories
)

for i, task_id in enumerate(task_directories):
    results = rvals[i]
    if results is None:
        continue
    else:
        for strategy in results:
            results_by_strategy[strategy][task_id] = results[strategy]

output_results_dictionary = {}
for strategy in results_by_strategy:
    if strategy not in output_results_dictionary:
        output_results_dictionary[strategy] = {}
    for task_id in results_by_strategy[strategy]:
        r = pd.DataFrame(results_by_strategy[strategy][task_id])
        r = r.transpose()
        r['configuration_id'] = r['configuration_id']#.astype(int)
        r.set_index('configuration_id', inplace=True)
        output_results_dictionary[strategy][int(task_id)] = r

if len(output_results_dictionary) != 1:
    raise ValueError("Found more than one strategy")

output_results_dictionary = output_results_dictionary[strategy]

output_file_name = os.path.join(args.save_to, 'matrix.pkl')
print("Dumping to ", output_file_name)
with open(output_file_name, 'wb') as fh:
    pickle.dump(output_results_dictionary, fh)
