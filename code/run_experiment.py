import time
from code.data_loader import load_data
from code. store_final_results

from itertools import starmap

    



def run(cpu_mp, cores, num_trials, data_path, result_path, run_configuration, data_configuration):
    # get data
    print('load data')
    data = load_data(data_path, data_configuration)
    print('Starting time : ')
    trials_results = []
    if cpu_mp:
        # build trial_args

        for trial in trials:
            # check memory increased
            m = starmap(run_trial, num)
            print (m)
            #Output:<itertools.starmap object at 0x0326F028>
            # Converting map object to list using list()
            num2 = list(m)
            trials_results = 
    else:
        for trial in trials:
            # check memory 
            run_trial(trial)
        # gpu stuff
            trials_results.append() 

    # store final results.
    store_final_results(result_path, trials_results)

def run_trial(trial, x,y):
    time.sleep(10)
    return x * y