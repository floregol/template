import numpy as np


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
    def __init__(self, input_dict, output_dict, batch_size, verbose=True):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.batch_size = batch_size
        self.end_index = input_dict['history'].shape[0]
        self.counter = 0
        self.verbose = verbose

    def __iter__(self):
        return self

    def __next__(self):
        start_batch = self.counter
        end_batch = start_batch+self.batch_size
        if end_batch > self.end_index:
            raise StopIteration
        self.counter = end_batch
        if self.verbose:
            print(end_batch, '/', self.end_index)
        input_batch_dict = {'history': self.input_dict['history'][start_batch:end_batch, :, :],
                            'time_of_day': self.input_dict['time_of_day'][start_batch:end_batch, :, :], 'node_id': self.input_dict['node_id'][start_batch:end_batch]}
        output_batch_dict = {
            'targets': self.output_dict['targets'][start_batch:end_batch, :, :]}
        return input_batch_dict, output_batch_dict
