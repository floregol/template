
from typing import NamedTuple


class GeneralParameters(NamedTuple):
    epochs: int
    steps_per_epoch: int
    batch_size: int
    init_learning_rate: float


class FCParameters(NamedTuple):
    epochs: int
    steps_per_epoch: int
    block_layers: int
    hidden_units: int
    blocks: int
    init_learning_rate: float
    batch_size: int
    weight_decay: float
    ts_id_dim: int
    num_stacks: int
    epsilon: float


class FlowParameters(NamedTuple):
    num_flow_coupling: int
    hidden_coupling: int
    coupling_layers: int


class LSTMParameters(NamedTuple):
    epochs: int
    steps_per_epoch: int
    init_learning_rate: float
    batch_size: int
    num_layer: int
    hidden_units: int
    dropout: float


class AllParameters(NamedTuple):
    general: GeneralParameters
    flow: FlowParameters
    lstm: LSTMParameters
    fc: FCParameters


def get_default_hyperparam(dataset_name, run_mode='multi_trials'):
    epochs = 45
    steps_per_epoch = 100
    batch_size = 4
    init_learning_rate = 5e-3
    # FcFlow
    num_flow_coupling = 2
    hidden_coupling = 4
    coupling_layers = 4
    # LSTM
    num_layer = 2
    hidden_units = 12
    dropout = 0.5
    # FC
    block_layers = 2
    blocks = 2
    weight_decay = 1e-3
    ts_id_dim = 4
    num_stacks = 2
    epsilon = 1e-6
    if run_mode == 'test_mode':
        epochs = 5
    if dataset_name == 'synthetic':
        num_layer = 1
        init_learning_rate = 0.03
        coupling_layers = 5
        hidden_units = 8
    elif dataset_name == 'elec':
        pass

    general_hyperparams = GeneralParameters(
        epochs,
        steps_per_epoch,
        batch_size,
        init_learning_rate)
    flow_hyperparams = FlowParameters(num_flow_coupling,
                                      hidden_coupling,
                                      coupling_layers)

    lstm_hyperparams = LSTMParameters(epochs,
                                      steps_per_epoch,
                                      init_learning_rate,
                                      batch_size,
                                      num_layer,
                                      hidden_units,
                                      dropout
                                      )
    fc_hyperparams = FCParameters(epochs,
                                  steps_per_epoch,
                                  block_layers,
                                  hidden_units,
                                  blocks,
                                  init_learning_rate,
                                  batch_size,
                                  weight_decay,
                                  ts_id_dim,
                                  num_stacks,
                                  epsilon
                                  )
    return AllParameters(general_hyperparams, flow_hyperparams, lstm_hyperparams, fc_hyperparams)
