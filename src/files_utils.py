import os.path
from os import path
import pickle as pkl


def check_path(path_dir):
    if not path.exists(path_dir):
        os.mkdir(path_dir)


def if_not_load_generate_and_store(store_path: str, filename: str, generate_func):
    file_store_path = os.path.join(store_path, filename)
    if os.path.exists(file_store_path):
        with open(file_store_path, 'rb') as f:
            data_dict_store = pkl.load(f)
    else:
        print(file_store_path, ' is not there, generating')
        check_path(store_path)
        data_dict_store = generate_func()
        pkl.dump(data_dict_store, open(file_store_path, "wb"))
    return data_dict_store
