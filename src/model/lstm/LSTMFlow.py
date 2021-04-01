from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import os
import tensorflow as tf
import numpy as np
from src.model.hyperparameter import LSTMParameters, FlowParameters
import tensorflow_probability as tfp
import properscoring as ps
from tensorflow.keras.layers import Input, Dense, LSTM
import time
from src.evaluation.metrics import compute_crpsum, compute_energy



tfd = tfp.distributions
tfb = tfp.bijectors


def Conditional_Coupling(horizon, pdf_dim, cond_dim, coupling_layers, hidden_dim):

    cond_flow_input = Input(
        shape=(horizon, pdf_dim+cond_dim), name='conditional_flow_input')

    reg = 0.01
    t_layer = Dense(
        hidden_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(cond_flow_input)

    s_layer = Dense(
        hidden_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(cond_flow_input)

    for l in range(coupling_layers-2):
        t_layer = Dense(
            hidden_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
        )(t_layer)

        s_layer = Dense(
            hidden_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(reg)
        )(s_layer)
    t_layer_last = Dense(
        pdf_dim, activation="linear", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(t_layer)

    s_layer_last = Dense(
        pdf_dim, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(reg)
    )(s_layer)
    return tf.keras.Model(inputs=cond_flow_input, outputs=[s_layer_last, t_layer_last])


class MultiVariateLSTM:
    def __init__(self, hyperparams: LSTMParameters, name: str, num_ts: int, history_length: int, horizon: int):
        super(MultiVariateLSTM, self).__init__()

        self.hyperparams = hyperparams
        self.name = name
        self.num_ts = num_ts
        self.history_length = history_length
        self.input_size = self.num_ts+1
        # inputs: A 3D tensor with shape [batch, timesteps, feature].

        self.lstm_layers = []

        for i in range(hyperparams.num_layer-1):
            self.lstm_layers.append(
                LSTM(units=hyperparams.hidden_units, dropout=hyperparams.dropout, return_sequences=True))
        self.lstm_layers.append(
            LSTM(units=hyperparams.hidden_units, dropout=hyperparams.dropout))
        self.output_dense = Dense(self.num_ts*self.history_length)

        inputs, outputs = self.get_model()
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self.inputs = inputs
        self.outputs = outputs
        self.model = model

    def get_model(self):

        history_in = tf.keras.layers.Input(
            shape=(self.num_ts, self.history_length), name='history')
        time_of_day_in = tf.keras.layers.Input(
            shape=(self.num_ts, self.history_length), name='time_of_day')
        node_id_in = tf.keras.layers.Input(
            shape=(self.num_ts, 1), dtype=tf.uint16, name='node_id')
        time = tf.reshape(
            time_of_day_in[:, 0, :], shape=[-1, 1, self.history_length])
        lstm_input = tf.concat([history_in, time], axis=1)
        lstm_input = tf.reshape(
            lstm_input, shape=[-1, self.history_length, self.input_size])
        lstm_output = self.lstm_layers[0](lstm_input)

        for nbg in self.lstm_layers[1:]:
            lstm_output = nbg(lstm_output)
        forecast = self.output_dense(lstm_output)

        forecast = tf.where(tf.math.is_nan(forecast),
                            tf.zeros_like(forecast), forecast)
        forecast = tf.reshape(
            forecast, shape=[-1, self.num_ts, self.history_length])
        inputs = {'history': history_in, 'node_id': node_id_in,
                  'time_of_day': time_of_day_in}
        outputs = {'targets': forecast}

        return inputs, outputs


class LSTMFlow(tf.keras.Model):

    def __init__(self, lstm_hyperparams: LSTMParameters, flow_hyperparams: FlowParameters, model_name: str, num_ts: int, history_length: int, horizon: int):
        super(LSTMFlow, self).__init__()
        self.flow_hyperparams = flow_hyperparams
        self.lstm_hyperparams = lstm_hyperparams
        self.model_name = model_name
        self.pdf_dim = num_ts
        self.cond_dim = num_ts
        self.num_ts = num_ts
        self.horizon = horizon
        self.history_length = history_length
        self.num_coupling_layers = flow_hyperparams.num_flow_coupling

        # Split the random variables in two, alternating to be set constant between each coupling flows.
        lower_d_var = int(self.pdf_dim/2)
        upper_D_var = self.pdf_dim - lower_d_var

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(name='distribution',
                                                                     loc=[0.0 for i in range(self.pdf_dim)], scale_diag=[1.0 for i in range(self.pdf_dim)]
                                                                     )
        zeros = [0 for i in range(lower_d_var)]
        ones = [1 for i in range(upper_D_var)]
        self.masks = np.array(
            [zeros+ones, ones+zeros] * (self.num_coupling_layers // 2), dtype="float32"
        )

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.layers_list = [Conditional_Coupling(horizon=horizon, pdf_dim=self.pdf_dim, cond_dim=self.cond_dim, coupling_layers=flow_hyperparams.coupling_layers, hidden_dim=flow_hyperparams.hidden_coupling)
                            for i in range(self.num_coupling_layers)]
        self.lstm = MultiVariateLSTM(hyperparams=lstm_hyperparams,
                                     name="lstm_model", num_ts=num_ts, history_length=history_length, horizon=horizon).model

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, data, training=True):
        cond_forecast = self.lstm(data[0])['targets']
        target_x = data[1]['targets']
        # put pdf dim as last shape
        cond_forecast = tf.transpose(cond_forecast, perm=[0, 2, 1])
        target_x = tf.transpose(target_x, perm=[0, 2, 1])
        log_det_inv = 0
        for i in range(self.num_coupling_layers)[::-1]:

            x_masked = tf.math.multiply(target_x, self.masks[i])
            reversed_mask = 1 - self.masks[i]

            x_maked_cond = tf.concat([target_x, cond_forecast], axis=2)

            s, t = self.layers_list[i](x_maked_cond)
            s *= reversed_mask
            t *= reversed_mask

            target_x = (
                reversed_mask
                * (target_x * tf.exp(-1 * s) - 1 * t * tf.exp(-1 * s))
                + x_masked
            )

            log_det_inv += -1 * tf.reduce_sum(s, [2])

        return target_x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x, y):
        y, logdet = self([x, y])
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, dict_data):
        x, y = dict_data
        with tf.GradientTape() as tape:

            loss = self.log_loss(x, y)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, dict_data):
        x, y = dict_data
        loss = self.log_loss(x, y)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def get_batch_samples(self, batch_data, num_samples=100):
        cond_forecast = self.lstm(batch_data)['targets']
        cond_forecast = tf.transpose(cond_forecast, perm=[0, 2, 1])
        shape = cond_forecast.shape

        cond_forecast = tf.tile(cond_forecast, [num_samples, 1, 1])

        x = tfd.Sample(self.distribution,
                       sample_shape=(
                           cond_forecast.shape[0], self.horizon)).sample()

        for i in range(self.num_coupling_layers):

            x_masked = tf.math.multiply(x, self.masks[i])

            reversed_mask = 1 - self.masks[i]
            x_maked_cond = tf.concat([x, cond_forecast], axis=2)
            s, t = self.layers_list[i](x_maked_cond)
            s *= reversed_mask
            t *= reversed_mask
            x = (
                reversed_mask
                * (x * tf.exp(s) + t)
                + x_masked
            )
        x = tf.reshape(x, shape=(num_samples, -1, shape[1], shape[2]))
        x = tf.transpose(x, perm=[0, 1, 3, 2])
        return x
