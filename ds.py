import os
import sys
from dbservice.database_api import set_checkpoint_table_paths, recover_db_from_snapshots
from storagemanager.storage_manager import StorageManager
from stagingstorage.staging_storage import StagingStorage
from verifiability.log import Log
from writeaheadlog.write_ahead_log import WAL
from crypto.key_manager import KeyManager
from gatekeeper import gatekeeper
from clientapi.client_api import ClientAPI
from common.general_utils import parse_config
from dbservice import database_api
from Interceptor import interceptor
import multiprocessing
import pathlib
import time

class DSConfig:
    def __init__(self, ds_config):
        # get the trust mode for the data station
        self.trust_mode = ds_config["trust_mode"]

        # get storage path for data
        self.storage_path = ds_config["storage_path"]

        # staging path
        self.staging_path = ds_config["staging_path"]

        # log arguments
        self.log_in_memory_flag = ds_config["log_in_memory"]
        self.log_path = ds_config["log_path"]

        # wal arguments
        self.wal_path = ds_config["wal_path"]
        self.check_point_freq = ds_config["check_point_freq"]

        # the table_paths in dbservice.check_point
        self.table_paths = ds_config["table_paths"]

        # interceptor paths
        self.ds_storage_path = str(pathlib.Path(ds_config["storage_path"]).absolute())
        self.mount_point = str(pathlib.Path(ds_config["mount_path"]).absolute())


class DataStation:
    def __init__(self, ds_config, app_config, need_to_recover=False):
        # parse config file
        self.config = DSConfig(ds_config)

        # set up trust mode
        trust_mode = self.config.trust_mode

        # set up an instance of the storage_manager
        storage_path = self.config.storage_path
        self.storage_manager = StorageManager(storage_path)

        # set up an instance of the staging_storage
        staging_path = self.config.staging_path
        self.staging_storage = StagingStorage(staging_path)

        # set up an instance of the log
        log_in_memory_flag = self.config.log_in_memory_flag
        log_path = self.config.log_path
        self.data_station_log = Log(log_in_memory_flag, log_path, trust_mode)

        # set up an instance of the write ahead log
        wal_path = self.config.wal_path
        check_point_freq = self.config.check_point_freq
        self.write_ahead_log = WAL(wal_path, check_point_freq)

        # start interceptor process
        ds_storage_path = self.config.ds_storage_path
        mount_point = self.config.mount_point

        manager = multiprocessing.Manager()

        accessible_data_dict = manager.dict()
        data_accessed_dict = manager.dict()

        self.interceptor_process = multiprocessing.Process(target=interceptor.main,
                                                    args=(ds_storage_path,
                                                            mount_point,
                                                            accessible_data_dict,
                                                            data_accessed_dict))

        self.interceptor_process.start()
        print("starting interceptor...")
        counter = 0
        while not os.path.ismount(mount_point):
            time.sleep(1)
            counter += 1
            if counter == 10:
                print("mount time out")
                exit(1)
        print("Mounted {} to {}".format(ds_storage_path, mount_point))
        print(os.path.dirname(os.path.realpath(__file__)))

        # set up the application registration in the gatekeeper
        connector_name = app_config["connector_name"]
        connector_module_path = app_config["connector_module_path"]
        gatekeeper_response = gatekeeper.gatekeeper_setup(connector_name, connector_module_path)
        if gatekeeper_response.status == 1:
            print("something went wrong in gatekeeper setup")
            exit(1)

        # set up the table_paths in dbservice.check_point
        table_paths = self.config.table_paths
        set_checkpoint_table_paths(table_paths)

        # set up an instance of the key manager
        # self.key_manager = KeyManager()

        # lastly, set up an instance of the client_api
        # client_api = ClientAPI(storage_manager,
        #                     staging_storage,
        #                     data_station_log,
        #                     write_ahead_log,
        #                     key_manager,
        #                     trust_mode,
        #                     interceptor_process,
        #                     accessible_data_dict,
        #                     data_accessed_dict,
        #                     )

        # Lastly, if we are in recover mode, we need to call
        # if need_to_recover:
        #     client_api.load_symmetric_keys()
        #     recover_db_from_snapshots(client_api.key_manager)
        #     client_api.recover_db_from_wal()

        # The following field decides which data_id we should use when we upload a new DE
        # Right now we are just incrementing by 1
        data_id_resp = database_api.get_data_with_max_id()
        if data_id_resp.status == 1:
            self.cur_data_id = data_id_resp.data[0].id + 1
        else:
            self.cur_data_id = 1
        # print("Starting data id should be:")
        # print(self.cur_data_id)

        # The following fields decides which staging_data_id we should use at a new insertion
        staging_id_resp = database_api.get_staging_with_max_id()
        if staging_id_resp.status == 1:
            self.cur_staging_data_id = staging_id_resp.data[0].id + 1
        else:
            self.cur_staging_data_id = 1
