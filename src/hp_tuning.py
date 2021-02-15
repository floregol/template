from bayes_opt import BayesianOptimization
import numpy as np
from src.model.lstmflow_model import run_lstmflow
from src.model.hyperparameter import *
from src.model.data_helper import *
general_hyperparams = GeneralParameters(
    epochs=45,
    steps_per_epoch=100,
    batch_size=64,
    init_learning_rate=1e-3)


def run_hp_tuning(data, data_configuration: dict, model_name: str):
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

    def black_box_function(num_flow_coupling, hidden_coupling, coupling_layers, init_learning_rate, num_layer, hidden_units, dropout):
        num_flow_coupling = int(2*num_flow_coupling)
        hidden_coupling = int(hidden_coupling)
        coupling_layers = int(coupling_layers)
        num_layer = int(num_layer)
        hidden_units = int(hidden_units)
        flow_hyperparams = FCFlowParameters(num_flow_coupling=num_flow_coupling,
                                            hidden_coupling=hidden_coupling,
                                            coupling_layers=coupling_layers)

        lstm_hyperparams = LSTMParameters(epochs=general_hyperparams.epochs,
                                          steps_per_epoch=general_hyperparams.steps_per_epoch,
                                          num_layer=num_layer,
                                          init_learning_rate=init_learning_rate,
                                          batch_size=general_hyperparams.batch_size,
                                          hidden_units=hidden_units,
                                          dropout=dropout
                                          )
        tf_model, x_test_dict, y_test_dict = run_lstmflow(X_train, Y_train, X_valid, Y_valid, X_test,
                                                          Y_test, num_time_series, data_configuration, lstm_hyperparams, flow_hyperparams, verbose=False)

        test_data_iter = IterData(x_test_dict, y_test_dict, 16, verbose=False)
        crp_sum, energy = tf_model.get_crps(test_data_iter)
        return crp_sum

    pbounds = {'num_flow_coupling': (1, 5), 'hidden_coupling': (4, 64), 'coupling_layers': (
        2, 10), 'init_learning_rate': (1e-5, 1e-2), 'num_layer': (1, 3), 'hidden_units': (8, 64), 'dropout': (0.1, 0.7)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    s = optimizer.maximize(
        init_points=5,
        n_iter=10,
    )

    print(s)
