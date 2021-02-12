import time
from src.data_loader import load_data
from src.result_storing import store_final_results
from itertools import starmap
from src.model.lstm_model import run_lstm
from src.model.fcflow_model import run_fcflow, run_fcgaga
from src.model.gruflow_model import run_gruflow
from src.model.hyperparameter import FCParameters, FCFlowParameters
#from src.plotting import visualize_dataset

fc_hyperparams = FCParameters(epochs=2,
                              steps_per_epoch=3,
                              block_layers=2,
                              hidden_units=4,
                              blocks=2,
                              init_learning_rate=1e-3,
                              batch_size=8,
                              weight_decay=1e-5,
                              ts_id_dim=2,
                              num_stacks=2,
                              epsilon=10)
flow_hyperparams = FCFlowParameters(num_flow_coupling=6,
                                    hidden_coupling=16,
                                    coupling_layers=4)


def run(data_path, result_path, run_configuration, data_configuration):
    # get data
    print('load data')
    start_data_time = time.time()
    data = load_data(data_path, data_configuration)
    print('Loading data time : ', "%.2f".format(
        (time.time()-start_data_time)/1000), 'sec')
    trials_results = []
    hyperparams = {'epochs': 40, 'steps_per_epoch': 100, 'batch_size': 32}
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

        run_fcflow(X_train, Y_train, X_valid, Y_valid, X_test,
                   Y_test, num_time_series, data_configuration, fc_hyperparams, flow_hyperparams)
    elif model_name == 'GRUNF':
        run_gruflow(X_train, Y_train, X_valid, Y_valid, X_test,
                    Y_test, num_time_series, data_configuration, gru_hyperparams)
    else:
        print('no model bb')

    exit()
