import yaml
import os
import shutil

def parse_config(path_to_config):
    with open(path_to_config) as config_file:
        res_config = yaml.load(config_file, Loader=yaml.FullLoader)
    return res_config

def clean_test_env(log_path):

    if os.path.exists("data_station.db"):
        os.remove("data_station.db")

    if os.path.exists(log_path):
        os.remove(log_path)

    folders = ['SM_storage', 'Staging_storage']
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)