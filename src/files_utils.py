import os.path
from os import path

def check_path(path_dir):
    if not path.exists(path_dir):
        os.mkdir(path_dir)

