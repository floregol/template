import numpy as np
import properscoring as ps

import matplotlib.pyplot as plt


def average_crps(multivariate_sample, multivariate_true_val):
    univar_crp = []
    for ts in range(multivariate_sample.shape[0]):
        univariate_true_val = multivariate_true_val[ts]
        univariate_sample = multivariate_sample[ts, :]
        univar_crp.append(ps.crps_ensemble(
            univariate_true_val, univariate_sample))

    return np.mean(univar_crp)


def crpsum(multivariate_sample, multivariate_true_val):
    sample_sum = np.sum(multivariate_sample, axis=0)
    true_val_sum = np.sum(multivariate_true_val)
    return ps.crps_ensemble(true_val_sum, sample_sum) / np.sum(np.abs(multivariate_true_val))


def mc_energy(multivariate_sample, multivariate_true_val):

    diff = multivariate_sample - multivariate_true_val.reshape(-1, 1)
    energy = np.mean(np.linalg.norm(diff, axis=0))  # 1/k sum || x_i -x ||

    diff_var = multivariate_sample[:, :-1] - multivariate_sample[:, 1:]
    # 1/k-1 sum || x_i - x_i+1 ||
    var_correction = np.mean(np.linalg.norm(diff_var, axis=0))

    return var_correction/2 - energy


def evaluate_pred(data, f_hat):
    a_crps = []
    mcrps = []
    crpssum = []
    for i in range(data.shape[1]):
        a_crps.append(average_crps(f_hat(), data[:, i]))
        mcrps.append(mc_energy(f_hat(), data[:, i]))
        crpssum.append(crpsum(f_hat(), data[:, i]))
    dict_metric = {'a_crps': a_crps, 'mcrps': mcrps, 'crpssum': crpssum}
    return dict_metric


def get_gaussian(mean, cov, num_samples):

    return np.random.multivariate_normal(mean, cov, num_samples).T


def experiment(samples_ground_truth, f_hat, f_hat_true):
    dict_metric = evaluate_pred(samples_ground_truth, f_hat)
    dict_metric_true = evaluate_pred(samples_ground_truth, f_hat_true)
    print('Average CRPS of Pred:', np.mean(dict_metric['a_crps']))
    print('Average CRPS of True F:', np.mean(dict_metric_true['a_crps']))

    print('CRPS_SUM of Pred:', np.mean(dict_metric['crpssum']))
    print('CRPS_SUM of True F:', np.mean(dict_metric_true['crpssum']))

    print('MCRPS of Pred:', np.mean(dict_metric['mcrps']))
    print('MCRPS of True F:', np.mean(dict_metric_true['mcrps']))


def variance_metric(num_samples, ds, metric):
    estimated_metric_mean = []
    estimated_metric_var = []
    for d in ds:
        for num_sample in num_samples:
            metric_trial = []
            for _ in range(10):
                data, f_hat = experiment_3(num_samples=num_sample, d=d)
                metric_result = []
                for i in range(data.shape[1]):
                    metric_result.append(metric(f_hat(), data[:, i]))
                metric_trial.append(np.mean(metric_result))
            estimated_metric_mean.append(np.mean(metric_trial))
            estimated_metric_var.append(np.std(metric_trial))
    return estimated_metric_mean, estimated_metric_var


def experiment_1():
    true_mean = [0, 0]
    true_cov = [[1, 0.5], [0.5, 1]]  # diagonal covariance
    samples_ground_truth = get_gaussian(true_mean, true_cov, 1000)

    pred_mean = [0, 0]
    pred_cov = [[1.5, 0], [0, 1.5]]  # diagonal covariance

    def f_hat():
        return get_gaussian(pred_mean, pred_cov, 100)

    def f_hat_true():
        return get_gaussian(true_mean, true_cov, 100)
    return samples_ground_truth, f_hat, f_hat_true


def experiment_2():
    true_mean = [0, 0, 0]
    true_cov = [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]  # diagonal covariance
    samples_ground_truth = get_gaussian(true_mean, true_cov, 10000)

    pred_mean = [0, 0, 0]
    pred_cov = [[1, 0, 0.5], [0, 1, 0], [0.5, 0, 1]]  # diagonal covariance

    def f_hat():
        return get_gaussian(pred_mean, pred_cov, 100)

    def f_hat_true():
        return get_gaussian(true_mean, true_cov, 100)
    return samples_ground_truth, f_hat, f_hat_true


def experiment_3(num_samples, d):
    true_mean = [0. for _ in range(d)]
    true_cov = np.random.random_sample(size=(d, d))
    true_cov = (true_cov + true_cov.T)/2 + d * np.eye(d)
    samples_ground_truth = get_gaussian(true_mean, true_cov, num_samples)

    pred_mean = true_mean
    pred_cov = np.random.random_sample(size=(d, d))
    pred_cov = (pred_cov + pred_cov.T)/2 + d * np.eye(d)

    def f_hat():
        return get_gaussian(pred_mean, pred_cov, 100)
    return samples_ground_truth, f_hat


if __name__ == '__main__':
    print('Experiment 1')
    samples_ground_truth, f_hat, f_hat_true = experiment_1()
    experiment(samples_ground_truth, f_hat, f_hat_true)
    print()
    print('Experiment 2')
    samples_ground_truth, f_hat, f_hat_true = experiment_2()
    experiment(samples_ground_truth, f_hat, f_hat_true)

    print()
    # print('Experiment 3')
    # d = [50]
    # num_N = [10, 50, 100, 500, 1000]
    # _, var_avg = variance_metric(num_N,
    #                          ds=d, metric=average_crps)
    # _, var_crpsum = variance_metric(num_N, ds=d, metric=crpsum)
   
    # _, var_ene = variance_metric(num_N, ds=d, metric=mc_energy)
   
    # plt.plot(num_N,var_avg,label='CRPS')
    # plt.plot(num_N,var_crpsum, label='CRPSsum')
    # plt.plot(num_N,var_ene, label='MCRPS')
    # plt.xlabel('sample size N')
    # plt.ylabel('Variance')
    # plt.legend()
    # plt.savefig('num.pdf')
    # plt.close()
    # d = [2,5,10,50,100]
    # num_N = [100]
    # _, var_avg = variance_metric(num_N,
    #                          ds=d, metric=average_crps)
    # _, var_crpsum = variance_metric(num_N, ds=d, metric=crpsum)
   
    # _, var_ene = variance_metric(num_N, ds=d, metric=mc_energy)
   
    # plt.plot(d,var_avg,label='CRPS')
    # plt.plot(d,var_crpsum, label='CRPSsum')
    # plt.plot(d,var_ene, label='MCRPS')
    # plt.xlabel('R.V. dimension')
    # plt.ylabel('Variance')
    # plt.legend()
    # plt.savefig('dimension.pdf')
