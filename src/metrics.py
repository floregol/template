"""
All time series metrics
"""

import numpy as np
import properscoring as ps
import time


def univariate_crp(multivariate_sample, multivariate_true_val):
    univar_crp = []
    for ts in range(multivariate_sample.shape[1]):
        univariate_true_val = multivariate_true_val[ts]
        univariate_sample = multivariate_sample[:, ts].numpy().tolist()
        univar_crp.append(ps.crps_ensemble(
            univariate_true_val, univariate_sample))

    return np.mean(univar_crp)


def compute_crpsum(multivariate_sample, multivariate_true_val):
    sample_sum = np.sum(multivariate_sample, axis=1)
    true_val_sum = np.sum(multivariate_true_val)
    return ps.crps_ensemble(true_val_sum, sample_sum) / np.sum(np.abs(multivariate_true_val))


def compute_energy(multivariate_sample, multivariate_true_val):
    num_samples = multivariate_sample.shape[0]
   
    squared = (multivariate_sample-multivariate_true_val)**2
    energy = np.mean([np.sum(squared[i])**0.5 for i in range(num_samples)])
    
    every = np.array([multivariate_sample- multivariate_sample[i] for i in range(num_samples)])
    every = every**2
    every = np.reshape(every, (num_samples*num_samples, -1))
    var_correction = np.mean([np.sum(every[i])**0.5 for i in range(num_samples*num_samples)])/2
    return (energy - var_correction) / np.mean(np.abs(multivariate_true_val))


if __name__ == "__main__":
    a = np.array([[3, 3], [4, 4], [5, 5]])
    b = np.array([1, 2])
    compute_energy(a, b)
