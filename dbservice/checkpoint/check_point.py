import pickle
import os
from collections import namedtuple
from dbservice import database_api
from crypto import cryptoutils as cu

TableContent = namedtuple("TableContent",
                          "content")

class CheckPoint:

    def __init__(self):
        self.table_paths = []

    def set_table_paths(self, table_paths):
        self.table_paths = table_paths

    def check_point_all_tables(self, key_manager):

        # First we erase all the existing snapshots
        for cur_path in self.table_paths:
            if os.path.exists(cur_path):
                os.remove(cur_path)

        # Pick a symmetric key to encrypt the tables
        sym_key_to_use = cu.get_symmetric_key_from_bytes(key_manager.agents_symmetric_key[1])

        # First we check point the user table
        user_res = database_api.get_all_users()
        user_table_as_list = user_res.data
        user_table_plain_bytes = pickle.dumps(user_table_as_list)
        user_table_cipher_bytes = sym_key_to_use.encrypt(user_table_plain_bytes)

        user_table = TableContent(content=user_table_cipher_bytes)

        with open(self.table_paths[0], "ab") as user_file:
            user_to_add = pickle.dumps(user_table)
            user_file.write(user_to_add)

        # Then we check point the data table
        data_res = database_api.get_all_datasets()
        data_table_as_list = data_res.data
        data_table_plain_bytes = pickle.dumps(data_table_as_list)
        data_table_cipher_bytes = sym_key_to_use.encrypt(data_table_plain_bytes)

        data_table = TableContent(content=data_table_cipher_bytes)

        with open(self.table_paths[1], "ab") as data_file:
            data_to_add = pickle.dumps(data_table)
            data_file.write(data_to_add)

        # Then we check point the policy table
        policy_res = database_api.get_all_policies()
        policy_table_as_list = policy_res.data
        policy_table_plain_bytes = pickle.dumps(policy_table_as_list)
        policy_table_cipher_bytes = sym_key_to_use.encrypt(policy_table_plain_bytes)

        policy_table = TableContent(content=policy_table_cipher_bytes)

        with open(self.table_paths[2], "ab") as policy_file:
            policy_to_add = pickle.dumps(policy_table)
            policy_file.write(policy_to_add)

    def recover_db_from_snapshots(self, key_manager):

        # Pick a symmetric key to encrypt the tables
        sym_key_to_use = cu.get_symmetric_key_from_bytes(key_manager.agents_symmetric_key[1])

        # We first decrypt all the tables and look at the list of objects

        # User table
        with open(self.table_paths[0], "rb") as f:
            user_res = pickle.load(f)

        user_content_cipher = user_res.content
        user_content_plain = sym_key_to_use.decrypt(user_content_cipher)
        user_content_list = pickle.loads(user_content_plain)

        # print(user_content_list)

        database_api.recover_users(user_content_list)

        # Data table
        with open(self.table_paths[1], "rb") as f:
            data_res = pickle.load(f)

        data_content_cipher = data_res.content
        data_content_plain = sym_key_to_use.decrypt(data_content_cipher)
        data_content_list = pickle.loads(data_content_plain)

        # print(data_content_list)

        database_api.recover_datas(data_content_list)

        # Policy table
        with open(self.table_paths[2], "rb") as f:
            policy_res = pickle.load(f)

        policy_content_cipher = policy_res.content
        policy_content_plain = sym_key_to_use.decrypt(policy_content_cipher)
        policy_content_list = pickle.loads(policy_content_plain)

        # print(policy_content_list)

        database_api.bulk_upload_policies(policy_content_list)

    def clear(self):
        self.table_paths = []


check_point = CheckPoint()