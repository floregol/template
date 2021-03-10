from src.model.data_helper import *
from src.evaluation.metrics import *
import numpy as np
import tensorflow as tf
import time
import tensorflow_probability as tfp
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

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

    for _, ground, samples in batch_test_predictions:
        mean_crps, mean_energy = get_metrics(ground, samples)
        ci_store, _,  coverage = get_ci_median(ground, samples)
    print()
    # pred_result, ground_true, coverage = get_ci_median(
    #     IterData(x_test_dict, y_test_dict, 16))
    # print(coverage)

    # print(pred_result)
    # print('xtest,', X_test.shape)
    # truth_index = list(
    #     range(0, data_configuration['history']+ci_size))
    # predict_index = list(
    #     range(data_configuration['history'], data_configuration['history']+ci_size))
    # predict_signal = list(pred_result[0][0:ci_size, 0, 0, :])

    # true_signal = list(X_test[0, 0, :, 0])+list(Y_test[0:ci_size, 0, 0, 0])

    # plot_time_series_ci(true_signal, predict_signal, list_index_ts=[
    #     truth_index, predict_index])

    # test_data_iter = IterData(x_test_dict, y_test_dict, 16)
    # crp_sum = fcflow.get_crps(test_data_iter)
    # print("crp sum : ", crp_sum)

    # pred_result, ground_true = fcflow.get_ci_median(
    #     IterData(x_test_dict, y_test_dict, 16))

    # truth_index = list(
    #     range(0, data_configuration['history']+data_configuration['horizon']))
    # predict_index = list(
    #     range(data_configuration['history'], data_configuration['history']+data_configuration['horizon']))
    # predict_signal = list(pred_result[0][0, :, 0, :])
    # print(predict_signal)
    # true_signal = list(X_test[0, 0, :, 0])+list(Y_test[0, 0, :, 0])

    # plot_time_series_ci(true_signal, predict_signal, list_index_ts=[
    #     truth_index, predict_index])

    # forward pass; from normal -> flow -> end result


def get_samples(data_iter, num_samples=100):
    sampled_prediction = []
    for input_batch_dict, output_batch_dict in data_iter:

        sampled_prediction.append(get_batch_samples(
            batch_data=input_batch_dict, num_samples=num_samples))

    return sampled_prediction


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
        print(b, '/', size_batch)
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
