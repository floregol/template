import pickle as pk
import os

HP_TUNING_FILE = 'hp_tuning.pkl'
RESULT_FILE = 'pred.pkl'


def store_file(file_path, dict_to_store):
    with open(file_path, 'wb') as f:
        pk.dump(dict_to_store, f)


def load_file(file_path):
    with open(file_path, 'rb') as f:
        dict_to_store = pk.load(f)
    return dict_to_store


def store_final_results(result_path, trials_results):
    file_path = os.path.join(result_path, RESULT_FILE)
    store_file(file_path, trials_results)


def load_results(result_path):
    file_path = os.path.join(result_path, RESULT_FILE)
    return load_file(file_path)


def store_hp_results(result_path, dict_to_store):
    file_path = os.path.join(result_path, HP_TUNING_FILE)
    store_file(file_path, dict_to_store)


def load_hp_results(result_path):
    file_path = os.path.join(result_path, HP_TUNING_FILE)
    return load_file(file_path)
