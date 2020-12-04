
from code.run_experiment import run
import os
current_dir = os.getcwd()

# todo result_path arguments from input
test_run = False # check that all storing is properly set up
cpu_mp = True
cores = 4
num_trials = 10
data_path = os.path.join(current_dir,'data')
result_path = os.path.join(current_dir,'results')

run(cpu_mp, cores, num_trials, data_path, result_path)


