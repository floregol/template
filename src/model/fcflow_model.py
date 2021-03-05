
from src.model.fcgaga.FCGAGA import FcGaga
from src.model.fcgaga.FCFlow import FcFlow
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
from src.evaluation.plotting import plot_time_series, plot_time_series_pdf, plot_time_series_ci
from src.model.data_helper import *


def run_fcflow(X_train, Y_train, X_valid, Y_valid, X_test,
               Y_test, num_time_series, data_configuration, fc_hyperparams, flow_hyperparams, verbose=True):

    fcflow = FcFlow(fc_hyperparams=fc_hyperparams, flow_hyperparams=flow_hyperparams,
                    model_name="fcgaga_model", num_ts=num_time_series, history_length=data_configuration['history'], horizon=data_configuration['horizon'])

    x_train_dict, y_train_dict, x_val_dict, y_val_dict, x_test_dict, y_test_dict = construct_input_output_keras_dict(X_train, Y_train, X_valid, Y_valid, X_test,
                                                                                                                     Y_test, num_time_series)

    fcflow.compile(optimizer=tf.keras.optimizers.Adam())

    fit_output = fcflow.fit(x=x_train_dict, y=y_train_dict, batch_size=fc_hyperparams.batch_size,
                            validation_data=(x_val_dict, y_val_dict),
                            epochs=fc_hyperparams.epochs,
                            steps_per_epoch=fc_hyperparams.steps_per_epoch,
                            verbose=verbose)
    
    return fcflow, x_test_dict, y_test_dict


def run_fcgaga(X_train, Y_train, X_valid, Y_valid, X_test,
               Y_test, num_time_series, data_configuration, fc_hyperparams, verbose=True):

    fc_model = FcGaga(hyperparams=fc_hyperparams,
                      name="fcgaga_model", num_ts=num_time_series, history_length=data_configuration['history'], horizon=data_configuration['horizon'])


    x_train_dict, y_train_dict, x_val_dict, y_val_dict, x_test_dict, y_test_dict = construct_input_output_keras_dict(X_train, Y_train, X_valid, Y_valid, X_test,
                                                                                                                     Y_test, num_time_series)
    fc_model.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE))
    fit_output = fc_model.model.fit(x=x_train_dict, y=y_train_dict, batch_size=fc_hyperparams.batch_size,
                                    validation_data=(x_val_dict, y_val_dict),
                                    epochs=fc_hyperparams.epochs,
                                    steps_per_epoch=fc_hyperparams.steps_per_epoch,
                                    verbose=verbose)

    prediction = fc_model.model.predict(x=x_test_dict)['targets']
    truth_index = list(
        range(0, data_configuration['history']+data_configuration['horizon']))
    predict_index = list(
        range(data_configuration['history'], data_configuration['history']+data_configuration['horizon']))
    predict_signal = list(prediction[0, 0, :])
    true_signal = list(X_test[0, 0, :, 0])+list(Y_test[0, 0, :, 0])

    plot_time_series(list_ts=[true_signal, predict_signal], list_index_ts=[
                     truth_index, predict_index])

    predict_signal = list(prediction[0, 3, :])
    true_signal = list(X_test[0, 3, :, 0])+list(Y_test[0, 3, :, 0])

    plot_time_series(list_ts=[true_signal, predict_signal], list_index_ts=[
                     truth_index, predict_index])

    predict_signal = list(prediction[0, 43, :])
    true_signal = list(X_test[0, 43, :, 0])+list(Y_test[0, 43, :, 0])

    plot_time_series(list_ts=[true_signal, predict_signal], list_index_ts=[
                     truth_index, predict_index])
