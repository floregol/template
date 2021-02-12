import time
from src.data_loader import load_data
from src.result_storing import store_final_results
from itertools import starmap
from src.model.lstm_model import run_lstm
from src.model.fcflow_model import run_fcflow, run_fcgaga
from src.model.lstmflow_model import run_lstmflow
from src.model.hyperparameter import *
from src.plotting import visualize_dataset
from src.model.data_helper import *
from src.plotting import plot_time_series, plot_time_series_pdf, plot_time_series_ci

general_hyperparams = GeneralParameters(
    epochs=45,
    steps_per_epoch=100,
    batch_size=64,
    init_learning_rate=1e-3)

fc_hyperparams = FCParameters(epochs=general_hyperparams.epochs,
                              steps_per_epoch=general_hyperparams.steps_per_epoch,
                              block_layers=2,
                              hidden_units=8,
                              blocks=2,
                              init_learning_rate=general_hyperparams.init_learning_rate,
                              batch_size=general_hyperparams.batch_size,
                              weight_decay=1e-5,
                              ts_id_dim=2,
                              num_stacks=1,
                              epsilon=10)
flow_hyperparams = FCFlowParameters(num_flow_coupling=6,
                                    hidden_coupling=16,
                                    coupling_layers=4)

lstm_hyperparams = LSTMParameters(epochs=general_hyperparams.epochs,
                                  steps_per_epoch=general_hyperparams.steps_per_epoch,
                                  num_layer=2,
                                  init_learning_rate=general_hyperparams.init_learning_rate,
                                  batch_size=general_hyperparams.batch_size,
                                  hidden_units=40,
                                  dropout=0.1
                                  )


def run(data_path, result_path, run_configuration, data_configuration):
    # get data
    print('load data')
    start_data_time = time.time()
    data = load_data(data_path, data_configuration)
    print('Loading data time : ', "%{:10.4f}".format(
        (time.time()-start_data_time)/1000), 'sec')
    trials_results = []
    if run_configuration['cpu_mp']:
        # build trial_args

        for trial in range(run_configuration['num_trials']):
            # check memory increased
            # m = starmap(run_trial, num)
            print(m)

    else:
        for trial in range(run_configuration['num_trials']):
            # check memory
            trial_results = run_trial(
                trial, data, data_configuration, run_configuration['model'])

            trials_results.append(trial_results)

    # store final results.
    store_final_results(result_path, trials_results)


def run_trial(trial: int, data, data_configuration: dict, model_name: str):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train = data.get_train_val_test_split()
    num_time_series = X_train.shape[1]
   # visualize_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    # data preparation
    X_train = data.scale_ts(X_train, ts_scaler_computed_from_train)
    Y_train = data.scale_ts(Y_train, ts_scaler_computed_from_train)
    X_valid = data.scale_ts(X_valid, ts_scaler_computed_from_train)
    Y_valid = data.scale_ts(Y_valid, ts_scaler_computed_from_train)
    X_test = data.scale_ts(X_test, ts_scaler_computed_from_train)
    Y_test = data.scale_ts(Y_test, ts_scaler_computed_from_train)
   # visualize_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    if model_name == 'LSTM':
        run_lstm(X_train, Y_train, X_valid, Y_valid, X_test,
                 Y_test, num_time_series, data_configuration, lstm_hyperparams)
    elif model_name == 'FC':

        run_fcgaga(X_train, Y_train, X_valid, Y_valid, X_test,
                   Y_test, num_time_series, data_configuration, fc_hyperparams)
    elif model_name == 'FCFlow':

        tf_model, x_test_dict, y_test_dict = run_fcflow(X_train, Y_train, X_valid, Y_valid, X_test,
                                                        Y_test, num_time_series, data_configuration, fc_hyperparams, flow_hyperparams)
    elif model_name == 'LSTMFlow':
        tf_model, x_test_dict, y_test_dict = run_lstmflow(X_train, Y_train, X_valid, Y_valid, X_test,
                                                          Y_test, num_time_series, data_configuration, lstm_hyperparams, flow_hyperparams)
    else:
        print('no model bb')

    test_data_iter = IterData(x_test_dict, y_test_dict, 16)
    crp_sum, energy = tf_model.get_crps(test_data_iter)
    print("crp sum : ", crp_sum)
    print("energy : ", energy)
    # ci_size = 30
    pred_result, ground_true, coverage = tf_model.get_ci_median(
        IterData(x_test_dict, y_test_dict, 16))
    print(coverage)
    # print(pred_result)
    # print('xtest,', X_test.shape)
    # truth_index = list(
    #     range(0, data_configuration['history']+ci_size))
    # predict_index = list(
    #     range(data_configuration['history'], data_configuration['history']+ci_size))
    # predict_signal = list(pred_result[0][0:ci_size, 0, 0, :])

    # true_signal = list(X_test[0, 0, :, 0])+list(Y_test[0:ci_size, 0, 0, 0])

    # plot_time_series_ci(true_signal, predict_signal, list_index_ts=[
    #     truth_index, predict_index])

    # exit()
# test_data_iter = IterData(x_test_dict, y_test_dict, 16)
#     crp_sum = fcflow.get_crps(test_data_iter)
#     print("crp sum : ", crp_sum)

#     pred_result, ground_true = fcflow.get_ci_median(
#         IterData(x_test_dict, y_test_dict, 16))

#     truth_index = list(
#         range(0, data_configuration['history']+data_configuration['horizon']))
#     predict_index = list(
#         range(data_configuration['history'], data_configuration['history']+data_configuration['horizon']))
#     predict_signal = list(pred_result[0][0, :, 0, :])
#     print(predict_signal)
#     true_signal = list(X_test[0, 0, :, 0])+list(Y_test[0, 0, :, 0])

#     plot_time_series_ci(true_signal, predict_signal, list_index_ts=[
#         truth_index, predict_index])
