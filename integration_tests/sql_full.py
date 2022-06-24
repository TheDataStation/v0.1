# This script loads a state of the data station needed to test out the SQL example
# in full trust mode.
import os
import pathlib
import main
import shutil

from common.general_utils import parse_config
from common.pydantic_models.user import User
from common.pydantic_models.policy import Policy

if __name__ == '__main__':

    if os.path.exists("data_station.db"):
        os.remove("data_station.db")

    # System initialization

    ds_config = parse_config("data_station_config.yaml")
    app_config = parse_config("app_connector_config.yaml")

    ds_storage_path = str(pathlib.Path(ds_config["storage_path"]).absolute())
    mount_point = str(pathlib.Path(ds_config["mount_path"]).absolute())

    client_api = main.initialize_system(ds_config, app_config)

    # Remove the code block below if testing out durability of log
    log_path = client_api.log.log_path
    if os.path.exists(log_path):
        os.remove(log_path)

    # We upload one data owner: owner of the DB company.db, with two relations INFO and PAYMENT
    # We also upload a data user
    num_users = 2

    for cur_num in range(num_users):
        cur_uname = "user" + str(cur_num)
        client_api.create_user(User(user_name=cur_uname, password="string"))

    # Clear the storage place

    folder = 'SM_storage'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Now we upload company.db to DS
    cur_token = client_api.login_user("user1", "string")["access_token"]

    DE_to_upload = "integration_tests/sql_test/company.db"
    cur_file = open(DE_to_upload, "rb")
    cur_file_bytes = cur_file.read()
    cur_optimistic_flag = False
    name_to_upload = "company.db"
    client_api.upload_dataset(name_to_upload,
                              cur_file_bytes,
                              "file",
                              cur_optimistic_flag,
                              cur_token, )

    # Upload policy saying user0 can access company.db
    client_api.upload_policy(Policy(user_id=1, api="run_predefined_query", data_id=1), cur_token)

    # Run analytics on the DB
    cur_token = client_api.login_user("user0", "string")["access_token"]
    res = client_api.call_api("run_predefined_query", cur_token, "optimistic")
    print(res)

    # Shut down
    client_api.shut_down(ds_config)

