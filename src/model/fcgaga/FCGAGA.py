import tensorflow as tf
import numpy as np
import os
from src.model.hyperparameter import FCParameters


class FcGaga:
    def __init__(self, hyperparams: FCParameters, name: str, num_ts: int, history_length: int, horizon: int):
        super(FcGaga, self).__init__()
        

        self.hyperparams = hyperparams
        self.name = name
        self.num_ts = num_ts
        self.history_length = history_length
        self.input_size = self.history_length + \
            self.hyperparams.ts_id_dim + self.num_ts*self.history_length
        self.fcgaga_layers = []
        for i in range(hyperparams.num_stacks):
            self.fcgaga_layers.append(FcGagaLayer(hyperparams=hyperparams,
                                                  input_size=self.input_size,
                                                  horizon_output_size=horizon,
                                                  history_length=self.history_length,
                                                  num_ts=self.num_ts,
                                                  name=f"fcgaga_{i}")
                                      )

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

        _, forecast = self.fcgaga_layers[0](
            history_in=history_in, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
        for nbg in self.fcgaga_layers[1:]:
            _, forecast_graph = nbg(
                history_in=forecast, node_id_in=node_id_in, time_of_day_in=time_of_day_in)
            forecast = forecast + forecast_graph
            
        forecast = forecast / self.hyperparams.num_stacks
        forecast = tf.where(tf.math.is_nan(forecast),
                            tf.zeros_like(forecast), forecast)
       
        
        inputs = {'history': history_in, 'node_id': node_id_in,
                  'time_of_day': time_of_day_in}
        outputs = {'targets': forecast}
        return inputs, outputs


class FcGagaLayer(tf.keras.layers.Layer):
    def __init__(self, hyperparams: FCParameters, input_size: int, horizon_output_size: int, history_length: int, num_ts: int, **kw):
        super(FcGagaLayer, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.num_ts = num_ts
        self.input_size = input_size
        self.horizon_output_size = horizon_output_size
        self.blocks = []
        for i in range(self.hyperparams.blocks):
            self.blocks.append(FcBlock(hyperparams=hyperparams,
                                       input_size=self.input_size,
                                       output_size=self.horizon_output_size,
                                       name=f"block_{i}"))

        self.node_id_em = tf.keras.layers.Embedding(input_dim=self.num_ts,
                                                    output_dim=self.hyperparams.ts_id_dim,
                                                    embeddings_initializer='uniform',
                                                    input_length=self.num_ts, name="dept_id_em",
                                                    embeddings_regularizer=tf.keras.regularizers.l2(hyperparams.weight_decay))

        self.time_gate1 = tf.keras.layers.Dense(hyperparams.hidden_units,
                                                activation=tf.keras.activations.relu,
                                                name=f"time_gate1")
        self.time_gate2 = tf.keras.layers.Dense(self.horizon_output_size,
                                                activation=None,
                                                name=f"time_gate2")
        self.time_gate3 = tf.keras.layers.Dense(history_length,
                                                activation=None,
                                                name=f"time_gate3")

    def call(self, history_in, node_id_in, time_of_day_in):
        
        node_id = self.node_id_em(node_id_in)

        node_embeddings = tf.squeeze(node_id[0, :, :])
        node_id = tf.squeeze(node_id, axis=-2)

        time_gate = self.time_gate1(
            tf.concat([node_id, time_of_day_in], axis=-1))
        time_gate_forward = self.time_gate2(time_gate)
        time_gate_backward = self.time_gate3(time_gate)

        history_in = history_in / (1.0 + time_gate_backward)

        node_embeddings_dp = tf.tensordot(
            node_embeddings,  tf.transpose(node_embeddings, perm=[1, 0]), axes=1)
        node_embeddings_dp = tf.math.exp(
            self.hyperparams.epsilon*node_embeddings_dp)
        node_embeddings_dp = node_embeddings_dp[tf.newaxis, :, :, tf.newaxis]

        level = tf.reduce_max(history_in, axis=-1, keepdims=True)

        history = tf.math.divide_no_nan(history_in, level)
        # Add history of all other nodes
        shape = history_in.get_shape().as_list()
        all_node_history = tf.tile(history_in[:, tf.newaxis, :, :], multiples=[
                                   1, self.num_ts, 1, 1])

        all_node_history = all_node_history * node_embeddings_dp
        all_node_history = tf.reshape(
            all_node_history, shape=[-1, self.num_ts, self.num_ts*shape[2]])
        all_node_history = tf.math.divide_no_nan(
            all_node_history - level, level)
        all_node_history = tf.where(
            all_node_history > 0, all_node_history, 0.0)
        history = tf.concat([history, all_node_history], axis=-1)
        # Add node ID
        history = tf.concat([history, node_id], axis=-1)

        backcast, forecast_out = self.blocks[0](history)
        for i in range(1, self.hyperparams.blocks):
            backcast, forecast_block = self.blocks[i](backcast)
            forecast_out = forecast_out + forecast_block
        forecast_out = forecast_out[:, :, :self.horizon_output_size]
        forecast = forecast_out * level

        forecast = (1.0 + time_gate_forward) * forecast

        return backcast, forecast


class FcBlock(tf.keras.layers.Layer):
    def __init__(self, hyperparams: FCParameters, input_size: int, output_size: int, **kw):
        super(FcBlock, self).__init__(**kw)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.fc_layers = []
        for i in range(hyperparams.block_layers):
            self.fc_layers.append(
                tf.keras.layers.Dense(hyperparams.hidden_units,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.keras.regularizers.l2(
                                          hyperparams.weight_decay),
                                      name=f"fc_{i}")
            )
        self.forecast = tf.keras.layers.Dense(
            self.output_size, activation=None, name="forecast")
        self.backcast = tf.keras.layers.Dense(
            self.input_size, activation=None, name="backcast")

    def call(self, inputs, training=False):
        h = self.fc_layers[0](inputs)
        for i in range(1, self.hyperparams.block_layers):
            h = self.fc_layers[i](h)
        backcast = tf.keras.activations.relu(inputs - self.backcast(h))
        return backcast, self.forecast(h)
