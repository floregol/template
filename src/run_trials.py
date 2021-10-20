
import numpy as np

def consolidate_trials(list_trial_results):

    all_p_valid = [result_dict['sample_eval']['p valid'] for result_dict in list_trial_results]
    all_p_invalid = [result_dict['sample_eval']['p invalid'] for result_dict in list_trial_results]
    dict_trial = {'mean_p_val':np.mean(all_p_valid), 'mean_p_inval':np.mean(all_p_invalid), 'all_p_val':all_p_valid, 'all_p_inval':all_p_invalid}
    return dict_trial


def multiple_trials(run_trial_fun, num_trials, arg_trial_fun,run_configuration,silence):
    trials_results = []
    for trial in range(num_trials):
            # chseck memory
            trial_results = run_trial(
                trial, run_configuration, run_trial_fun, arg_trial_fun,silence=silence)
        # store_intermediate_results(trial_results)  #TODO
            trials_results.append(trial_results)
            if not silence:
                print(trial_results)
    # store final results to files.
    dict_trial = consolidate_trials(trials_results)
   
    return dict_trial




def run_trial(trial: int, run_configuration: dict, run_trial_fun, arg_trial_fun,silence):
    if not silence:
        print('trial',trial)
    return run_trial_fun(trial=trial,pair_model_params=arg_trial_fun,silence=silence)
