from bayes_opt import BayesianOptimization
import numpy as np
from src.run_trials import multiple_trials

def get_metric_to_max(dict_trial):
    return dict_trial['mean_p_val']

def run_hp_tuning(run_trial_fun, num_trials, arg_trial_fun,run_configuration):
    
    pbounds = {'diffusion_lambda': (0,1)}
    def black_box_function(diffusion_lambda):
        pair_model_params={'diffusion_lambda':diffusion_lambda, 'diff':True}
        dict_trial = multiple_trials(run_trial_fun, num_trials, pair_model_params,run_configuration,silence=True)
        return get_metric_to_max(dict_trial)

   
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=4,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=4,
    )

    return optimizer

