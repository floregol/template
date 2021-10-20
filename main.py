
from src.run_experiment import run
import os
import sys
import time
from src.files_utils import check_path

import argparse

YOUR_MACHINE = 'flo-XPS-13-9360'
DESKTOP = 'floregol-XPS-8930'
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--f', action='store_true',
                    help='A boolean switch tou store log in files, no stdout')
args = parser.parse_args()
stdout_to_file = args.f
# Check where we are and set up paths/folders
machine = os.uname()[1]
print('We are on', machine)

if machine in [YOUR_MACHINE,DESKTOP ]: # set up local machine path here
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data')
    result_path = os.path.join(current_dir, 'results')
    log_path = os.path.join(current_dir, 'log')

else: # set up server path here
    data_path = '/home/floregol/scratch/data'
    result_path = '/home/floregol/scratch/results/'
    log_path = '/home/floregol/scratch/log/'
check_path(result_path, verbose=True, path_description='Result path')
check_path(log_path,  verbose=True, path_description='Log path')

if stdout_to_file:
    log_file = os.path.join(log_path, str(time.time())+'_log.txt')
    print('Stdout in', log_file)
    sys.stdout = open(log_file, 'w')


run_configuration = {'run_mode': 'multi_trials',  # test_mode, multi_trials, hp_tuning
                    'num_trials': 20}

run_trial_fun =  lambda trial, x : 0
list_arg_trial_fun = [{}]
#list_arg_trial_fun = [{'diffusion_lambda':-3, 'diff':True},{'diffusion_lambda':0.5482,'diff':True}]


run(result_path, run_configuration, run_trial_fun, list_arg_trial_fun)
if stdout_to_file:
    sys.stdout.close()
