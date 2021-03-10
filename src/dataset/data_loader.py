import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import pickle as pkl
from src.dataset.electricity import ElectricityDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.evaluation.plotting import plot_time_series
from src.synthetic.generate_valve import generate_TS
from src.files_utils import if_not_load_generate_and_store


def load_elec_data(dataset_path: str):
    elec_dataset = ElectricityDataset.load()
    metadata = {'ids': elec_dataset.ids, 'dates': elec_dataset.dates}
    data_raw = elec_dataset.values
    return metadata, data_raw  # [num_ts, num_timesteps]
# Todo move to class


def load_synthetic_data(dataset_path: str, num_timesteps: int = 2000):
    data_raw = generate_TS(time_serie_length=num_timesteps)
    num_ts = data_raw.shape[0]
    metadata = {'ids': list(range(num_ts)),
                'dates': list(range(num_timesteps))}
    return metadata, data_raw  # [num_ts, num_timesteps]

# Load from files if already generated, else need to generate and store.


class Dataset():
    def __init__(self, dataset_path, name, load_dataset):
        self.dataset_path = dataset_path
        self.name = name
        self.load_dataset = load_dataset
        self.partition_ratio = {'train': 0.6, 'test': 0.3, 'val': 0.1}
        data_dict = if_not_load_generate_and_store(
            dataset_path, name+'_data.pkl', self.generate_dataset)
        self.metadata = data_dict['metadata']
        self.np_data = data_dict['np_data']
        # reduce data, only take 200 first time steps

        self.np_data = self.np_data[:, 0:1000]

        XY_dict = if_not_load_generate_and_store(
            dataset_path, name+'_XY.pkl', self.generate_X_Y)
        self.X = XY_dict['X']
        self.Y = XY_dict['Y']

        self.num_input = self.X.shape[0]

    def generate_dataset(self):
        self.metadata, self.np_data = self.load_dataset(self.dataset_path)
        data_dict_store = {'metadata': self.metadata,
                           'np_data': self.np_data, 'name': self.name}
        return data_dict_store

    def get_train_val_test_split(self, shuffle=True, seed=0):
        # self.X, self.y = [num_example, dim], [num_example, labels]
        if shuffle:
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.X, self.Y, test_size=self.partition_ratio['test'], random_state=seed)
            valid_ratio_of_train = self.partition_ratio['valid'] * \
                self.num_input/X_train.shape[0]
            X_train, X_valid, Y_train, Y_valid = train_test_split(
                X_train, Y_train, test_size=valid_ratio_of_train, random_state=seed)
        else:
            end_train_index = int(self.partition_ratio['train']*self.num_input)
            end_val_index = int(
                self.partition_ratio['val']*self.num_input) + end_train_index

            # self.X[0:end_train_index, :,:,:,....]
            X_train = self.X[0:end_train_index, Ellipsis]
            Y_train = self.Y[0:end_train_index, Ellipsis]
            X_valid = self.X[end_train_index:end_val_index, Ellipsis]
            Y_valid = self.Y[end_train_index:end_val_index, Ellipsis]
            X_test = self.X[end_val_index:, Ellipsis]
            Y_test = self.Y[end_val_index:, Ellipsis]

        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

    def generate_X_Y(self):
        return NotImplementedError


class TimeSerieDataset(Dataset):
    def __init__(self, dataset_path, name, load_dataset, history, horizon):
        self.history = history
        self.horizon = horizon
        super().__init__(dataset_path, name, load_dataset)

    def generate_covariates(self):
        initial_shape = self.np_data.shape  # [TS X Time]

        total_TS = self.np_data.shape[0]
        total_timesteps = self.np_data.shape[1]

        array_with_covariate = np.zeros(
            shape=(total_TS, total_timesteps, 2))  # [TS X Time X Features]
        # [TS , Time, 0] = signal ---- [TS , Time, 1] = time index feature
        array_with_covariate[:, :, 0] = self.np_data
        time_index_covariate = np.array([i/total_timesteps
                                         for i in range(total_timesteps)]).reshape(-1, 1)  # [[0, 0.0..,  ... , 1]]

        array_with_covariate[:, :, 1] = np.repeat(
            time_index_covariate, total_TS, axis=1).T

        print('generated and added covariates : (number TS, Time)->(number TS, Time, [signal,covariate])',
              initial_shape, '->', array_with_covariate.shape)
        return array_with_covariate

    def generate_X_Y(self):
        array_with_covariate = self.generate_covariates()
        # self.np_data [TS X Time X Features]
        window_length = self.history + self.horizon
        total_TS = self.np_data.shape[0]
        total_timesteps = self.np_data.shape[1]

        X = []
        Y = []
        for time_step in range(total_timesteps):
            if time_step + window_length < total_timesteps:
                X.append(
                    array_with_covariate[:, time_step:time_step+self.history, :])
                Y.append(
                    array_with_covariate[:, time_step+self.history:time_step+self.history+self.horizon, :])
            else:
                break
        X = np.array(X)  # [Time X TS X history X Features]
        Y = np.array(Y)  # [Time X TS X horizon X Features]
        return {'X': X, 'Y': Y}

    def get_train_val_test_split(self, seed=0):

        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = super(
        ).get_train_val_test_split(shuffle=False, seed=seed)
        self.verify_X_Y_split_done_right(X_train, Y_train, plot_ts=False)
        ts_scaler_computed_from_train = np.mean(
            X_train[:, :, 0, 0], axis=0)

        non_zero_series = np.argwhere(
            ts_scaler_computed_from_train).reshape(-1).tolist()

        initial_shape = X_train.shape
        X_train = X_train[:, non_zero_series, :, :]
        Y_train = Y_train[:, non_zero_series, :, :]
        X_valid = X_valid[:, non_zero_series, :, :]
        Y_valid = Y_valid[:, non_zero_series, :, :]
        X_test = X_test[:, non_zero_series, :, :]
        Y_test = Y_test[:, non_zero_series, :, :]
        print('removed zero valued train time series : ',
              initial_shape, '->', X_train.shape)
        ts_scaler_computed_from_train = ts_scaler_computed_from_train[
            non_zero_series]
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train

    def scale_ts(self,  multi_ts, ts_scaler_computed_from_train):
        multi_ts[:, :, :, 0] = multi_ts[:, :, :, 0] / \
            ts_scaler_computed_from_train[:, np.newaxis]
        return multi_ts

        # take a X Y pair and plot to see if it fits the initial ts.
    def scale_all_ts(self, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, ts_scaler_computed_from_train):
        X_train = self.scale_ts(X_train, ts_scaler_computed_from_train)
        Y_train = self.scale_ts(Y_train, ts_scaler_computed_from_train)
        X_valid = self.scale_ts(X_valid, ts_scaler_computed_from_train)
        Y_valid = self.scale_ts(Y_valid, ts_scaler_computed_from_train)
        X_test = self.scale_ts(X_test, ts_scaler_computed_from_train)
        Y_test = self.scale_ts(Y_test, ts_scaler_computed_from_train)
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

    def verify_X_Y_split_done_right(self, X, Y, offset_start_XY_ts=0, plot_ts=False):
        univar_ts_index = 1  # Select one of the time series
        start_index = 42  # timestamp index
        ts_input = X[start_index, univar_ts_index, :, 0]
        ts_output = Y[start_index, univar_ts_index, :, 0]
        window_length = self.history + self.horizon

        initial_ts = self.np_data[univar_ts_index, offset_start_XY_ts +
                                  start_index:offset_start_XY_ts+start_index+window_length]
        assert np.array_equal(
            ts_input, initial_ts[0:self.history]), 'Something is wrong with the input->output generation'
        assert np.array_equal(
            ts_output, initial_ts[self.horizon:window_length]), 'Something is wrong with the input->output generation'

        if plot_ts:
            delta = 0.05 * np.mean(initial_ts)
            x_index = list(range(0, self.horizon))
            y_index = list(range(self.horizon, window_length))
            init_index = list(range(0, window_length))
            plot_time_series([ts_input, ts_output, initial_ts -
                              delta], [x_index, y_index, init_index])


def load_data(data_path: str, data_configuration: dict) -> Dataset:
    dataset_name = data_configuration['dataset_name']
    dataset_path = os.path.join(data_path, dataset_name)
    if dataset_name == 'elec':
        dataset = TimeSerieDataset(
            dataset_path, dataset_name, load_elec_data, history=data_configuration['history'], horizon=data_configuration['horizon'])
    elif dataset_name == 'synthetic':
        dataset = TimeSerieDataset(dataset_path, dataset_name, load_synthetic_data,
                                   history=data_configuration['history'], horizon=data_configuration['horizon'])
    return dataset
