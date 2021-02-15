from src.model.data_helper import *
from src.plotting import plot_time_series, plot_time_series_pdf, plot_time_series_ci
from src.model.hyperparameter import *
import numpy as np
from src.model.lstm.LSTMFlow import LSTMFlow
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

def run_lstmflow(X_train, Y_train, X_valid, Y_valid, X_test,
                 Y_test, num_time_series, data_configuration, lstm_hyperparams, flow_hyperparams, verbose=True):

    lstmflow = LSTMFlow(lstm_hyperparams=lstm_hyperparams, flow_hyperparams=flow_hyperparams,  model_name="lstm_model",
                        num_ts=num_time_series, history_length=data_configuration['history'], horizon=data_configuration['horizon'])

    x_train_dict, y_train_dict, x_val_dict, y_val_dict, x_test_dict, y_test_dict = construct_input_output_keras_dict(X_train, Y_train, X_valid, Y_valid, X_test,
                                                                                                                     Y_test, num_time_series)

    lstmflow.compile(optimizer=tf.keras.optimizers.Adam())

    fit_output = lstmflow.fit(x=x_train_dict, y=y_train_dict, batch_size=lstm_hyperparams.batch_size,
                              validation_data=(x_val_dict, y_val_dict),
                              epochs=lstm_hyperparams.epochs,
                              steps_per_epoch=lstm_hyperparams.steps_per_epoch,
                              verbose=verbose)
   # lstmflow.summary()

    return lstmflow, x_test_dict, y_test_dict
