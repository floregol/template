
from typing import NamedTuple

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

