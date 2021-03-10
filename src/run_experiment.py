import time
from src.dataset.data_loader import load_data
from src.result_storing import store_final_results, store_hp_results
from src.run_hp_tuning import run_hp_tuning
from src.run_trials import run_trial
from src.evaluation.hp_eval import eval_hp_results
from src.evaluation.eval import eval_results
from itertools import starmap


def run(data_path, result_path, run_configuration, data_configuration):
    # get data
    print('load data')
    start_data_time = time.time()
    data = load_data(data_path, data_configuration)
    print('Loading data time : ', "{:10.4f}".format(
        (time.time()-start_data_time)), 'ms')
    trials_results = []
    if run_configuration['run_mode'] == 'hp_tuning':
        dict_to_store = run_hp_tuning(
            data, data_configuration, run_configuration['model'])
        store_hp_results(result_path, dict_to_store)
        eval_hp_results(result_path)
    else:
        if run_configuration['cpu_mp']:  # TODO add mp
            # build trial_args

            for trial in range(run_configuration['num_trials']):
                # check memory increased
                # m = starmap(run_trial, num)
                print(m)

        else:
            for trial in range(run_configuration['num_trials']):
                # check memory
                trial_results = run_trial(
                    trial, data, data_configuration, run_configuration, result_path)
                store_intermediate_results(trial_results) 
                trials_results.append(trial_results)

        # store final results to files.
        store_final_results(result_path, trials_results)
        eval_results(result_path)
