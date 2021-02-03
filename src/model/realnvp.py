import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU
from tensorflow.keras import Model
tfd = tfp.distributions
tfb = tfp.bijectors


class NN(tf.keras.layers.Layer):
    def __init__(self, input_shape, cond_size, output_shape, n_hidden=[8, 8], activation="relu", name="nn"):
        super(NN, self).__init__(name="nn")
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(
                Dense(hidden, activation=activation, name='dense_{}_1'.format(i)))
            layer_list.append(
                Dense(hidden, activation=activation, name='dense_{}_2'.format(i)))
        self.layer_list = layer_list

        self.log_s_layer = Dense(output_shape, activation="tanh", name='log_s')
        self.t_layer = Dense(output_shape, name='t')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t


class RealNVP(tfb.Bijector):
    def __init__(
        self,
        input_shape,
        cond_size,
        n_hidden=[64, 64],
        # this bijector do vector wise quantities.
        forward_min_event_ndims=1,
        validate_args: bool = False,
        name="real_nvp",
    ):

        super(RealNVP, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        self.input_shape = input_shape
        self.lower_split = int(input_shape[-1] // 2)
        self.upper_split = input_shape[-1] - self.lower_split
        self.cond_size = cond_size
        nn_layer = NN(input_shape=self.lower_split, cond_sizecond_size, output_shape=self.upper_split, n_hidden=n_hidden)
        nn_input_shape = input_shape.copy()
        nn_input_shape[-1] = self.lower_split + self.cond_size #concat(x|h)#
        x = tf.keras.Input(nn_input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name=name+"_nn")

    def _forward(self, x):
        x_a, x_b = tf.split(x, [self.upper_split, self.lower_split], axis=-1)
        y_b = x_b
        log_s, t = self.nn(x_b)
        s = tf.exp(log_s)
        y_a = s * x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, [self.upper_split, self.lower_split], axis=-1)
        print(y_b)

        x_b = y_b

        log_s, t = self.nn(y_b)
        s = tf.exp(log_s)
        x_a = (y_a - t) / s
        print(x_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        _, x_b = tf.split(x, [self.upper_split, self.lower_split], axis=-1)
        log_s, t = self.nn(x_b)
        return log_s
