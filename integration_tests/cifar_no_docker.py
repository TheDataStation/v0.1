# This script loads a state of the data station needed to test out cifar example in no_trust.
import pathlib
import os
import shutil
from crypto import cryptoutils as cu
import pickle
import torch
from torch.utils.data import DataLoader

from common import general_utils
from ds import DataStation
from common.pydantic_models.user import User
from common.pydantic_models.policy import Policy

if __name__ == '__main__':

    if os.path.exists("data_station.db"):
        os.remove("data_station.db")

    # System initialization

    ds_config = general_utils.parse_config("data_station_config.yaml")
    app_config = general_utils.parse_config("app_connector_config.yaml")

    ds_storage_path = str(pathlib.Path(ds_config["storage_path"]).absolute())
    mount_point = str(pathlib.Path(ds_config["mount_path"]).absolute())

    ds = DataStation(ds_config, app_config)

    # Remove the code block below if testing out durability of log
    log_path = ds.data_station_log.log_path
    if os.path.exists(log_path):
        os.remove(log_path)

    # Remove the code block below if testing out durability of wal
    wal_path = ds.write_ahead_log.wal_path
    if os.path.exists(wal_path):
        os.remove(wal_path)

    # Save data station's public key
    ds_public_key = ds.key_manager.ds_public_key

    # We upload 8 users, one holds each partition of the data (X and y)
    num_users = 8

    # We generate keys outside the loop because we don't want to count its time

    cipher_sym_key_list = []
    public_key_list = []

    for cur_num in range(num_users):
        sym_key = cu.generate_symmetric_key()
        cipher_sym_key = cu.encrypt_data_with_public_key(sym_key, ds_public_key)
        cipher_sym_key_list.append(cipher_sym_key)
        cur_private_key, cur_public_key = cu.generate_private_public_key_pair()
        public_key_list.append(cur_public_key)
        cur_uname = "user" + str(cur_num)
        ds.create_user(User(user_name=cur_uname, password="string"),
                       cipher_sym_key_list[cur_num],
                       public_key_list[cur_num], )

    # First clear ml_file_no_trust/training_income

    no_trust_folder = 'integration_tests/ml_file_no_trust/training_cifar'
    for filename in os.listdir(no_trust_folder):
        file_path = os.path.join(no_trust_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Clear the storage place

    folder = 'SM_storage'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Now we create the encrypted files

    for cur_num in range(num_users):
        cur_t_path = "integration_tests/ml_file_full_trust/training_cifar/train" + str(cur_num) + ".pt"
        cur_user_sym_key = ds.key_manager.agents_symmetric_key[cur_num + 1]
        # Load torch object
        cur_torch = torch.load(cur_t_path)
        # print(type(cur_torch))
        # torch object to pkl bytes
        pkl_t = pickle.dumps(cur_torch)
        # pkl bytes encrypted
        ciphertext_bytes = cu.encrypt_data_with_symmetric_key(pkl_t, cur_user_sym_key)
        cur_cipher_name = "integration_tests/ml_file_no_trust/training_cifar/train" + str(cur_num) + ".pkl"
        cur_cipher_file = open(cur_cipher_name, "wb")
        cur_cipher_file.write(ciphertext_bytes)
        cur_optimistic_flag = False
        name_to_upload = "train" + str(cur_num) + ".pt"
        cur_cipher_file.close()

    # For each user, we upload his partition of the data
    for cur_num in range(num_users):
        # Log in the current user and get a token
        cur_uname = "user" + str(cur_num)

        # Upload his partition X of the data
        cur_train_t = "integration_tests/ml_file_no_trust/training_cifar/train" + str(cur_num) + ".pkl"
        cur_file_t = open(cur_train_t, "rb")
        cur_file_bytes = cur_file_t.read()
        cur_optimistic_flag = False
        name_to_upload = "train" + str(cur_num) + ".pkl"
        cur_res = ds.upload_dataset(cur_uname,
                                    name_to_upload,
                                    cur_file_bytes,
                                    "file",
                                    cur_optimistic_flag, )
        cur_file_t.close()

        # Add a policy saying user with id==1 can call train_cifar_model on the datasets
        ds.upload_policy(cur_uname, Policy(user_id=1, api="train_cifar_model", data_id=cur_num + 1))

    # In here, DB construction is done. We just need to call train_cifar_model

    # Call the NN model
    test_data = torch.load('integration_tests/ml_file_full_trust/testing_cifar/test.pt')
    testloader = DataLoader(test_data, batch_size=32)

    accuracy = ds.call_api("user0", "train_cifar_model", "optimistic", 1, testloader)
    print("Model accuracy is: "+str(accuracy))

    # Shutting down

    ds.shut_down()