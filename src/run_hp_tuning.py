from bayes_opt import BayesianOptimization
import numpy as np
from src.model.lstmflow_model import run_lstmflow
from src.model.hyperparameter import *
from src.model.data_helper import *
from src.evaluation.eval import eval_results, get_prediction

# transform continuous to float value
def adjust_hp_param(num_flow_coupling, coupling_layers, init_learning_rate, num_layer, hidden_units, dropout):
    num_flow_coupling = 2*int(num_flow_coupling)
    hidden_coupling = 4
    coupling_layers = int(coupling_layers)
    num_layer = int(num_layer)
    hidden_units = int(hidden_units)
    return num_flow_coupling, coupling_layers, init_learning_rate, num_layer, hidden_units, dropout, hidden_coupling


def run_hp_tuning(data, data_configuration: dict, model_name: str):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train = data.get_train_val_test_split()
    num_time_series = X_train.shape[1]
    # data preparation
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = data.scale_all_ts(
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train)

    general_hyperparams = get_default_hyperparam(data.name).general

    def black_box_function(num_flow_coupling, coupling_layers, init_learning_rate, num_layer, hidden_units, dropout):
        num_flow_coupling, coupling_layers, init_learning_rate, num_layer, hidden_units, dropout, hidden_coupling = adjust_hp_param(
            num_flow_coupling, coupling_layers, init_learning_rate, num_layer, hidden_units, dropout)

        print('\t\t\t\t ', coupling_layers, '|{:10.2f}'.format(dropout), '|', hidden_units, '\t    |{:10.4f}'.format(
              init_learning_rate), '| ', num_flow_coupling, ' \t    | ', num_layer)
        flow_hyperparams = FlowParameters(num_flow_coupling=num_flow_coupling,
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

        prediction_test = get_prediction(tf_model, x_test_dict, y_test_dict)
    
        metrics_dict, _ = eval_results(prediction_test)
        return -np.mean(metrics_dict['coverage'])

    pbounds = {'num_flow_coupling': (1, 5),  'coupling_layers': (
        2, 10), 'init_learning_rate': (1e-5, 1e-1), 'num_layer': (1, 3), 'hidden_units': (8, 64), 'dropout': (0.1, 0.7)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )

    s = optimizer.maximize(
        init_points=3,
        n_iter=5,
    )
    dict_to_store = {'max_config': optimizer.max, 'all_config': optimizer.res}
    return dict_to_store
