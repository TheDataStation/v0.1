import os
import sys
from dbservice.database_api import set_checkpoint_table_paths, recover_db_from_snapshots
from storagemanager.storage_manager import StorageManager
from verifiability.log import Log
from writeaheadlog.write_ahead_log import WAL
from crypto.key_manager import KeyManager
from gatekeeper import gatekeeper
from clientapi.client_api import ClientAPI
from common.utils import parse_config
from Interceptor import interceptor
import multiprocessing
import pathlib
import time

def initialize_system(ds_config, app_config, need_to_recover=False):

    # print(ds_config)
    # print(app_config)

    # In this function we set up all components that need to be initialized

    # get the trust mode for the data station
    trust_mode = ds_config["trust_mode"]

    # set up an instance of the storage_manager
    storage_path = ds_config["storage_path"]
    storage_manager = StorageManager(storage_path)

    # set up an instance of the log
    log_in_memory_flag = ds_config["log_in_memory"]
    log_path = ds_config["log_path"]
    data_station_log = Log(log_in_memory_flag, log_path, trust_mode)

    # set up an instance of the write ahead log
    wal_path = ds_config["wal_path"]
    check_point_freq = ds_config["check_point_freq"]
    write_ahead_log = WAL(wal_path, check_point_freq)

    # set up an instance of the key manager
    key_manager = KeyManager()

    # zz: start interceptor process
    ds_storage_path = str(pathlib.Path(ds_config["storage_path"]).absolute())
    mount_point = str(pathlib.Path(ds_config["mount_path"]).absolute())

    # with multiprocessing.Manager() as :
    manager = multiprocessing.Manager()

    accessible_data_dict = manager.dict()
    data_accessed_dict = manager.dict()
    # signal = multiprocessing.Event()

    interceptor_process = multiprocessing.Process(target=interceptor.main,
                                                  args=(ds_storage_path,
                                                        mount_point,
                                                        accessible_data_dict,
                                                        data_accessed_dict))
    interceptor_process.start()
    print("starting interceptor...")
    # time.sleep(1)
    counter = 0
    while not os.path.ismount(mount_point):
        time.sleep(1)
        counter += 1
        if counter == 10:
            print("mount time out")
            exit(1)
    print("Mounted {} to {}".format(ds_storage_path, mount_point))

    # set up the application registration in the gatekeeper
    connector_name = app_config["connector_name"]
    connector_module_path = app_config["connector_module_path"]
    gatekeeper_response = gatekeeper.gatekeeper_setup(connector_name, connector_module_path)
    if gatekeeper_response.status == 1:
        print("something went wrong in gatekeeper setup")
        exit(1)

    # set up the table_paths in dbservice.check_point
    table_paths = ds_config["table_paths"]
    set_checkpoint_table_paths(table_paths)

    context = manager.dict()

    # lastly, set up an instance of the client_api
    client_api = ClientAPI(storage_manager,
                           data_station_log,
                           write_ahead_log,
                           key_manager,
                           trust_mode,
                           interceptor_process, accessible_data_dict, data_accessed_dict,
                           context)

    # Lastly, if we are in recover mode, we need to call
    if need_to_recover:
        client_api.load_symmetric_keys()
        recover_db_from_snapshots(client_api.key_manager)
        client_api.recover_db_from_wal()

    # return an instance of the client API
    return client_api

def run_system(ds_config):

    # start frontend
    if ds_config["front_end"] == "fastapi":
        os.system("uvicorn fast_api:app --reload")

    return


if __name__ == "__main__":
    print("Main")
    # First parse test_config files and command line arguments

    # https://docs.python.org/3/library/argparse.html
    # (potentiall need to re-write some key-values from clg)
    data_station_config = parse_config(sys.argv[1])
    app_connector_config = parse_config(sys.argv[2])
    initialize_system(data_station_config, app_connector_config)

    # run_system(parse_config("data_station_config.yaml"))
