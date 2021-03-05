
from src.run_experiment import run
import os
import sys
import time
from src.files_utils import check_path
# check on which machine we are and set appropriate path
machine = os.uname()[1]
print('We are on', machine)
if machine == 'flo-XPS-13-9360':

    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data')
    result_path = os.path.join(current_dir, 'results')
    log_path = os.path.join(current_dir, 'log')
    log_file = os.path.join(log_path, 'log.txt')
else:
    data_path = '/home/floregol/scratch/data'
    result_path = '/home/floregol/scratch/results/'
    log_path = '/home/floregol/scratch/log/'
    log_file = os.path.join(log_path, 'log.txt')

check_path(result_path)
check_path(log_path)


#sys.stdout = open(log_file, 'w')


run_configuration = {'run_mode': 'hp_tuning',  # test_mode, multi_trials, hp_tuning
                     'cpu_mp': False, 'cores': 4, 'num_trials': 10, 'model': 'LSTMFlow'}

data_configuration = {'dataset_name': 'synthetic', 'horizon': 4, 'history': 4}

run(data_path, result_path, run_configuration, data_configuration)

sys.stdout.close()
