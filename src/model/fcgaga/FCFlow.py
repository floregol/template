from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
import os
from src.model.hyperparameter import FCParameters, FCFlowParameters
from src.model.fcgaga.FCGAGA import FcGaga
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense
from src.evaluation.metrics import compute_crpsum, compute_energy
import time
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

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


class FcFlow(tf.keras.Model):

    def __init__(self, fc_hyperparams: FCParameters, flow_hyperparams: FCFlowParameters, model_name: str, num_ts: int, history_length: int, horizon: int):
        super(FcFlow, self).__init__()
        self.pdf_dim = num_ts
        self.cond_dim = num_ts
        self.num_ts = num_ts
        self.horizon = horizon
        self.num_coupling_layers = flow_hyperparams.num_flow_coupling

        # Split the random variables in two, alternating to be set constant between each coupling flows.
        lower_d_var = int(self.pdf_dim/2)
        upper_D_var = self.pdf_dim - lower_d_var

        # Distribution of the latent space.
        self.distribution = tfp.distributions.MultivariateNormalDiag(
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
        self.fcgaga = FcGaga(hyperparams=fc_hyperparams,
                             name="fcgaga_model", num_ts=num_ts, history_length=history_length, horizon=horizon).model

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    # data is a list of dict of inputs and outputs : [{nodeid, time, time features},{targets}]
    def call(self, data, training=True):
        cond_forecast = self.fcgaga(data[0])['targets']
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

    # forward pass; from normal -> flow -> end result
    def get_samples(self, data_iter, num_samples=100):
        sampled_prediction = []
        for input_batch_dict, output_batch_dict in data_iter:

            sampled_prediction.append(self.get_batch_samples(
                batch_data=input_batch_dict, num_samples=num_samples))

        return sampled_prediction

    def get_ci_median(self, data_iter, num_samples=100):
        pred_result = []
        ground_true = []
        ave_l = []
        for input_batch_dict, output_batch_dict in data_iter:
            samples_batch = self.get_batch_samples(input_batch_dict)
            size_batch = output_batch_dict['targets'].shape[0]
            num_time_series = output_batch_dict['targets'].shape[1]
            horizon = output_batch_dict['targets'].shape[2]
            samples_batch = self.get_batch_samples(input_batch_dict)
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

            pred_result.append(ci_store)
            ground_true.append(output_batch_dict['targets'])
        return pred_result, ground_true, np.mean(ave_l)

    def get_batch_samples(self, batch_data, num_samples=100):
        cond_forecast = self.fcgaga(batch_data)['targets']
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

    def get_crps(self, data_iter):
        crps = []
        energy = []
        for input_batch_dict, output_batch_dict in data_iter:
            size_batch = output_batch_dict['targets'].shape[0]
            num_time_series = output_batch_dict['targets'].shape[1]
            horizon = output_batch_dict['targets'].shape[2]
            samples_batch = self.get_batch_samples(input_batch_dict)
            for b in range(size_batch):
                for h in range(horizon):

                    multivariate_sample = samples_batch[:, b, :, h]
                    multivariate_true_val = output_batch_dict['targets'][b, :, h]
                    
                    energy.append(compute_energy(
                        multivariate_sample, multivariate_true_val))
                    

                    crps.append(compute_crpsum(
                        multivariate_sample, multivariate_true_val))

        return np.mean(crps), np.mean(energy)