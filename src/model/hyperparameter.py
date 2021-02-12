
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


class FCFlowParameters(NamedTuple):
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
