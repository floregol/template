
from src.result_storing import store_final_results
from src.files_utils import store_file
from src.run_trials import multiple_trials
from src.hp_optim import run_hp_tuning


def dict_arg_to_experiment(arg_trial_fun):
    experiment_key = ''
    for key, val in arg_trial_fun.items():
        experiment_key+=str(key) +':'+ str(val)
    return experiment_key

def run(result_path, run_configuration, run_trial_fun, list_arg_trial_fun):
    experiment_dict = {}
    num_trials = run_configuration['num_trials']
    if run_configuration['run_mode'] == 'hp_tuning':
        print('Tuning the parameters')
        _ = run_hp_tuning(run_trial_fun, num_trials, None,run_configuration)
        
    elif run_configuration['run_mode'] == 'multi_trials':
        for arg_trial_fun in list_arg_trial_fun:

        
                dict_trial = multiple_trials(run_trial_fun, num_trials, arg_trial_fun,run_configuration,silence=False)
                experiment_key = dict_arg_to_experiment(arg_trial_fun)
                #store_final_results(result_path, trials_results)
                experiment_dict[experiment_key] = dict_trial
        name_experiment = 'NAME'
        store_file(result_path,name_experiment+'.pkl',experiment_dict)

