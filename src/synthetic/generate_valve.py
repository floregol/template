import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
shape, scale = 1, 0.05
beta_param = 0.5
mu = 0
var = 0.1


def sample_RV():
    noises = np.random.normal(mu, var, 3)
    x = np.random.gamma(shape, scale, 1)
    V = np.random.beta(beta_param, beta_param, 2)
    return noises, x, V


def get_y0(rv):
    return rv[1]+2


def get_y1_y2(rv, y_0):
    v_1 = rv[2][0]
    v_2 = rv[2][1]
    y_1 = (v_1/(2*(v_1+v_2))) * y_0 + rv[0][0] + 1
    y_2 = (v_2/(2*(v_1+v_2))) * y_0 + rv[0][1] + 1
    return y_1, y_2


def get_y3(rv, y_1, y_2):
    return y_1 + y_2 + rv[0][2] 


def generate_TS(time_serie_length):
    ts_np = np.zeros((4, time_serie_length))

    ts_np[0, 0] = get_y0(sample_RV())
    rv_t1 = sample_RV()
    y_0_t0 = get_y0(rv_t1)
    ts_np[0, 0] = y_0_t0
    y_1, y_2 = get_y1_y2(rv_t1, y_0_t0)
    ts_np[0, 1] = y_0_t0
    ts_np[1, 1] = y_1
    ts_np[2, 1] = y_2
    for t in range(2, time_serie_length):
        rv = sample_RV()
        y_0 = get_y0(rv)
        previous_y_0 = ts_np[0, t-1]
        y_1, y_2 = get_y1_y2(rv, previous_y_0)
        previous_y_1 = ts_np[1, t-1]
        previous_y_2 = ts_np[2, t-1]
        y_3 = get_y3(rv, previous_y_1, previous_y_2)
        ts_np[0, t] = y_0
        ts_np[1, t] = y_1
        ts_np[2, t] = y_2
        ts_np[3, t] = y_3
    return ts_np


if __name__ == '__main__':
    time_serie_length = 20
    TS = generate_TS(time_serie_length=time_serie_length)
    TS = TS.t
    x = list(range(time_serie_length))
    plt.plot(x, TS[:, 0], label=r'$y_1$')
    plt.plot(x[1:], TS[1:, 1], label=r'$y_2$')
    plt.plot(x[1:], TS[1:, 2], label=r'$y_3$')
    plt.plot(x[2:], TS[2:, 3], label=r'$y_4$')
    plt.legend()
    plt.tight_layout()
    plt.show() 
