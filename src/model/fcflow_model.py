
from src.model.fcgaga.FCGAGA import FcGaga
from src.model.fcgaga.FCFlow import FcFlow
import tensorflow as tf
import numpy as np
#from src.plotting import plot_time_series, plot_time_series_pdf

# fcgaga takes node id as an input feature


def construct_node_id_feature(num_time_series, num_train, num_val, num_test):
    node_id = np.array([i for i in range(num_time_series)]).reshape(-1, 1)
    node_id_train = np.repeat(
        node_id, num_train, axis=1).T
    node_id_val = np.repeat(
        node_id, num_val, axis=1).T
    node_id_test = np.repeat(
        node_id, num_test, axis=1).T
    return node_id_train, node_id_val, node_id_test


def construct_input_output_keras_dict(X_train, Y_train, X_valid, Y_valid, X_test,
                                      Y_test, num_time_series):
    node_id_train, node_id_val, node_id_test = construct_node_id_feature(
        num_time_series, X_train.shape[0], X_valid.shape[0], X_test.shape[0])
    x_train_dict = {'history': X_train[:, :, :, 0],
                    'time_of_day': X_train[:, :, :, 1], 'node_id': node_id_train}
    y_train_dict = {'targets': Y_train[:, :, :, 0]}
    x_val_dict = {'history': X_valid[:, :, :, 0],
                  'time_of_day': X_valid[:, :, :, 1], 'node_id': node_id_val}
    y_val_dict = {'targets': Y_valid[:, :, :, 0]}
    x_test_dict = {'history': X_test[:, :, :, 0],
                   'time_of_day': X_test[:, :, :, 1], 'node_id': node_id_test}
    y_test_dict = {'targets': Y_test[:, :, :, 0]}
    return x_train_dict, y_train_dict, x_val_dict, y_val_dict, x_test_dict, y_test_dict


class IterData():
    def __init__(self, input_dict, output_dict, batch_size):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.batch_size = batch_size
        self.end_index = input_dict['history'].shape[0]
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        start_batch = self.counter
        end_batch = start_batch+self.batch_size
        if end_batch > self.end_index:
            raise StopIteration
        self.counter = end_batch
        input_batch_dict = {'history': self.input_dict['history'][start_batch:end_batch, :, :],
                      'time_of_day': self.input_dict['time_of_day'][start_batch:end_batch, :, :], 'node_id': self.input_dict['node_id'][start_batch:end_batch]}
        output_batch_dict = {'targets': self.output_dict['targets'][start_batch:end_batch, :, :]}
        return input_batch_dict, output_batch_dict


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

    

    test_data_iter = IterData(x_test_dict, y_test_dict, 16)
    crp_sum = fcflow.get_crps(test_data_iter)
    print(crp_sum)
    print(crp_sum/num_time_series)
    pred_result = fcflow.get_ci_median(IterData(x_test_dict, y_test_dict, 16))
    print(len(pred_result))
    

def run_fcgaga(X_train, Y_train, X_valid, Y_valid, X_test,
               Y_test, num_time_series, data_configuration, fc_hyperparams, verbose=True):

    fc_model = FcGaga(hyperparams=fc_hyperparams,
                      name="fcgaga_model", num_ts=num_time_series, history_length=data_configuration['history'], horizon=data_configuration['horizon'])

    print(fc_model.model.summary())

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
