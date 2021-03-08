#
from src.run_experiment import run
import os
current_dir = os.getcwd()

# todo result_path arguments from input

#data_path = os.path.join(current_dir, 'data')
data_path = '/home/floregol/scratch/data'
result_path = os.path.join(current_dir, 'results')
run_configuration = {'run_mode': 'hp_tuning',  # test_mode, multi_trials, hp_tuning
                     'cpu_mp': False, 'cores': 4, 'num_trials': 10, 'model': 'FCFlow'}
data_configuration = {'dataset_name': 'elec', 'horizon': 12, 'history': 12}

run(data_path, result_path, run_configuration, data_configuration)
