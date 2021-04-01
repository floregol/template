from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow import keras
from src.evaluation.eval import eval_results, get_prediction
from src.model.data_helper import *
from src.model.hyperparameter import *
from src.model.lstmflow_model import run_lstmflow
from src.model.lstm.LSTMFlow import LSTMFlow
from src.model.fcflow_model import run_fcflow, run_fcgaga
from src.model.lstm_model import run_lstm


def run_trial(trial: int, data, data_configuration: dict, run_configuration: dict, result_path: str):
    model_name = run_configuration['model']
    run_mode = run_configuration['run_mode']
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train = data.get_train_val_test_split()
    num_time_series = X_train.shape[1]
   # visualize_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    # data preparation
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = data.scale_all_ts(
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train)

   # visualize_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    hyperparam = get_default_hyperparam(data.name, run_mode)
    if model_name == 'LSTM':
        run_lstm(X_train, Y_train, X_valid, Y_valid, X_test,
                 Y_test, num_time_series, data_configuration, hyperparam.lstm)
    elif model_name == 'FC':
        run_fcgaga(X_train, Y_train, X_valid, Y_valid, X_test,
                   Y_test, num_time_series, data_configuration, hyperparam.fc)
    elif model_name == 'FCFlow':
        tf_model, x_test_dict, y_test_dict = run_fcflow(X_train, Y_train, X_valid, Y_valid, X_test,
                                                        Y_test, num_time_series, data_configuration, hyperparam.fc, hyperparam.flow)
    elif model_name == 'LSTMFlow':
        tf_model, x_test_dict, y_test_dict = run_lstmflow(X_train, Y_train, X_valid, Y_valid, X_test,
                                                          Y_test, num_time_series, data_configuration, hyperparam.lstm, hyperparam.flow)
    else:
        print('no model bb')

    prediction_test = get_prediction(tf_model, x_test_dict, y_test_dict)
    # store 
    
    result_store = eval_results(prediction_test)
    return result_store
