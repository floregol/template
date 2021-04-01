from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from src.model.data_helper import *
from src.evaluation.metrics import *
from src.evaluation.plotting import plot_input_output_batch
import numpy as np
import tensorflow as tf
import time
import tensorflow_probability as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


def get_prediction(model, x_test_dict, y_test_dict):
    test_data_iter = IterData(x_test_dict, y_test_dict, 16)
    batch_test_predictions = []
    for input_batch_dict, output_batch_dict in test_data_iter:
        size_batch = output_batch_dict['targets'].shape[0]
        num_time_series = output_batch_dict['targets'].shape[1]
        horizon = output_batch_dict['targets'].shape[2]
        samples_batch = model.get_batch_samples(input_batch_dict)
        batch_test_predictions.append((
            input_batch_dict, output_batch_dict, samples_batch))

    return batch_test_predictions


def eval_results(batch_test_predictions):
    metrics_dict = {'coverage':[]}
    ci = []
    for _, ground, samples in batch_test_predictions:
        #mean_crps, mean_energy = get_metrics(ground, samples)
        ci_store, _,  coverage = get_ci_median(ground, samples)
        metrics_dict['coverage'].append(coverage)
        ci.append(ci_store)
    #print('Average coverage',np.mean(metrics_dict['coverage']))
   # plot_input_output_batch(batch_test_predictions, ci)
    return metrics_dict, ci




def get_ci_median(output_batch_dict, samples_batch):
    pred_result = []
    ground_true = []
    ave_l = [] = []
    
    size_batch = output_batch_dict['targets'].shape[0]
    num_time_series = output_batch_dict['targets'].shape[1]
    horizon = output_batch_dict['targets'].shape[2]
    ci_store = np.zeros((size_batch, horizon, num_time_series, 3))
    batch_result = []
    for b in range(size_batch):
        horizon_result = []
        for h in range(horizon):
            dim_result = []
            for ts in range(num_time_series):
                samples = samples_batch[:, b, ts, h].numpy()
                ci_q = np.quantile(
                    samples, [0.05, 0.5, 0.95])
                ci_store[b, h, ts, :] = ci_q
                ave_l.append(ci_q[2] - ci_q[0])

    
    return ci_store, output_batch_dict['targets'],  np.mean(ave_l)


def get_metrics(output_batch_dict, samples_batch):
    crps = []
    energy = []
    size_batch = output_batch_dict['targets'].shape[0]
    num_time_series = output_batch_dict['targets'].shape[1]
    horizon = output_batch_dict['targets'].shape[2]
    
    for b in range(size_batch):
        for h in range(horizon):
            multivariate_sample = samples_batch[:, b, :, h]
            multivariate_true_val = output_batch_dict['targets'][b, :, h]
            start_energy = time.time()
            energy.append(compute_energy(
                multivariate_sample, multivariate_true_val))
            
            crps.append(compute_crpsum(
                multivariate_sample, multivariate_true_val))

    return np.mean(crps), np.mean(energy)
