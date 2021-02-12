import numpy as np

import matplotlib.pyplot as plt

num_time_series = 5
friends = [[0,1], [2,3]]
num_time_steps = 16

def generate_cov():
    cov = np.eye(num_time_series)
    for f in friends:
        cov[f[0], f[1]] = 2
        cov[f[1], f[0]] = 2
    return cov


mean = [0 for i in range(num_time_series)]
cov = generate_cov()

cov_signal_samples = np.random.multivariate_normal(mean, cov, num_time_steps).T


def plot_time_series(time_series):
    for i, ts in enumerate(time_series):
        plt.plot(ts , label='TS'+str(i))
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_time_series(signal_samples.tolist())