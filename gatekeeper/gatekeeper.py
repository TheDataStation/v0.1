import os
import pathlib
from dsapplicationregistration.dsar_core import (get_api_endpoint_names,
                                                 get_functions_names,
                                                 get_registered_functions, )
from dbservice import database_api
from sharemanager import share_manager
from policybroker import policy_broker
from common.pydantic_models.function import Function
from common.abstraction import DataElement

from verifiability.log import Log
from writeaheadlog.write_ahead_log import WAL
from crypto.key_manager import KeyManager
from ds_dev_utils.jail_utils import DSDocker, FlaskDockerServer


class Gatekeeper:
    def __init__(self,
                 data_station_log: Log,
                 write_ahead_log: WAL,
                 key_manager: KeyManager,
                 trust_mode: str,
                 epf_path,
                 mount_dir,
                 development_mode,
                 ):
        """
        The general class for the gatekeeper, which brokers access to data elements and
         runs jail functions
        """

        print("Start setting up the gatekeeper")

        # save variables
        self.data_station_log = data_station_log
        self.write_ahead_log = write_ahead_log
        self.key_manager = key_manager
        self.trust_mode = trust_mode

        self.epf_path = epf_path
        self.mount_dir = mount_dir
        self.docker_id = 1
        self.server = FlaskDockerServer()
        self.server.start_server()

        # register all api_endpoints that are functions in database_api
        function_names = get_functions_names()
        # now we call dbservice to register these info in the DB
        for cur_f in function_names:
            database_service_response = database_api.create_function(cur_f)
            if database_service_response["status"] == 1:
                print("database_api.create_function: internal database error")
                raise RuntimeError(
                    "database_api.create_function: internal database error")

        f_res = database_api.get_all_functions()
        if f_res["status"] == 0:
            print("all function registered: ", f_res["data"])

        if not development_mode:
            docker_image_realpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".")
            print("docker image path: ", docker_image_realpath)
            self.docker_session = DSDocker(
                self.server,
                docker_image_realpath,
            )

        print("Gatekeeper setup success")

    def get_new_docker_id(self):
        ret = self.docker_id
        self.docker_id += 1
        return ret

    # We add times to the following function to record the overheads

    def call_api(self,
                 api,
                 cur_user_id,
                 contract_id,
                 *args,
                 **kwargs):
        """
        Calls the API specified, ensuring that
          - data is only exposed to the API if permitted
          - data accessed by API is allowed

        Parameters:
         api: api to call
         cur_user_id: the user id to decide what data is exposed
         contract_id: id of contract from which the api is called,

        Returns:
         Response based on what happens
        """

        # print(trust_mode)

        # Check if caller is in destination agent
        dest_a_ids = share_manager.get_dest_ids_for_share(share_id)
        if cur_user_id not in dest_a_ids:
            print("Caller not a destination agent")
            return None

        # Check if the share has been approved by all approval agents
        share_ready_flag = share_manager.check_share_ready(share_id)
        if not share_ready_flag:
            print("This share has not been approved to execute yet.")
            return None

        # If yes, set the accessible_de to be the entirety of P
        all_accessible_de_id = share_manager.get_de_ids_for_contract(contract_id)
        # print(f"all accessible data elements are: {all_accessible_de_id}")

        get_des_by_ids_res = database_api.get_des_by_ids(all_accessible_de_id)
        if get_des_by_ids_res.status == -1:
            err_msg = "No accessible data for " + api
            print(err_msg)
            return {"status": 1, "message": err_msg}

        accessible_de = set()
        for cur_de in get_des_by_ids_res.data:
            if self.trust_mode == "no_trust":
                data_owner_symmetric_key = self.key_manager.get_agent_symmetric_key(cur_de.owner_id)
            else:
                data_owner_symmetric_key = None
            cur_de = DataElement(cur_de.id,
                                 cur_de.name,
                                 cur_de.type,
                                 cur_de.access_param,
                                 data_owner_symmetric_key)
            accessible_de.add(cur_de)

        # print(accessible_de)

        # actual api call
        ret = call_actual_api(api,
                              self.epf_path,
                              self.mount_dir,
                              self.key_manager.agents_symmetric_key,
                              accessible_de,
                              self.get_new_docker_id(),
                              self.docker_session,
                              *args,
                              )

        api_result = ret["return_info"][0]
        data_path_accessed = ret["return_info"][1]
        decryption_time = ret["return_info"][2]

        data_ids_accessed = []
        for path in data_path_accessed:
            print(path)
            data_ids_accessed.append(int(path.split("/")[-2]))
        # print("API result is", api_result)

        print("data accessed is", data_ids_accessed)
        # print("accessible data by policy is", accessible_data_policy)
        print("all accessible data is", all_accessible_de_id)
        # print("Decryption time is", decryption_time)

        if set(data_ids_accessed).issubset(set(all_accessible_de_id)):
            # print("All data access allowed by policy.")
            # log operation: logging intent_policy match
            self.data_station_log.log_intent_policy_match(cur_user_id,
                                                          api,
                                                          data_ids_accessed,
                                                          self.key_manager, )
            # In this case, we can return the result to caller.
            response = {"status": 0,
                        "message": "Contract result can be released",
                        "result": [api_result, decryption_time]}
        # elif set(data_ids_accessed).issubset(all_accessible_de_id):
        #     # print("Some access to optimistic data not allowed by policy.")
        #     # log operation: logging intent_policy mismatch
        #     self.data_station_log.log_intent_policy_mismatch(cur_user_id,
        #                                                      api,
        #                                                      data_ids_accessed,
        #                                                      set(accessible_de_policy),
        #                                                      self.key_manager, )
        #     response = APIExecResponse(status=-1,
        #                                message="Some access to optimistic data not allowed by policy.",
        #                                result=[api_result, data_ids_accessed], )
        else:
            # log operation: logging intent_policy mismatch
            self.data_station_log.log_intent_policy_mismatch(cur_user_id,
                                                             api,
                                                             data_ids_accessed,
                                                             set(all_accessible_de_id),
                                                             self.key_manager, )
            response = {"status": 1,
                        "message": "Access to illegal DE happened. Something went wrong."}

        return response

    def shut_down(self):
        self.server.stop_server()


def call_actual_api(api_name,
                    epf_path,
                    mount_dir,
                    agents_symmetric_key,
                    accessible_de,
                    docker_id,
                    docker_session:DSDocker,
                    *args,
                    **kwargs,
                    ):
    """
    The thread that runs the API on the Docker container

    Parameters:
     api_name: name of API to run on Docker container
     epf_path: path to the epf file
     mount_dir: directory of filesystem mount for Interceptor
     agents_symmetric_key: key manager storing all the sym keys
     accessible_de: a set of accessible DataElement
     docker_id: id assigned to docker container
     server: flask server to receive communications with docker container
     *args / *kwargs for api

    Returns:
     Result of api
    """

    print(os.path.dirname(os.path.realpath(__file__)))
    # print(api_name, *args, **kwargs)
    epf_realpath = os.path.dirname(os.path.realpath(__file__)) + "/../" + epf_path

    config_dict = {"accessible_de": accessible_de, "docker_id": docker_id, "agents_symmetric_key": agents_symmetric_key}
    print("The real epf path is", epf_realpath)

    # print(session.container.top())

    # run function
    list_of_functions = get_registered_functions()

    for cur_f in list_of_functions:
        if api_name == cur_f.__name__:
            print("call", api_name)
            docker_session.flask_run(api_name, epf_realpath, mount_dir, config_dict, *args, **kwargs)
            ret = docker_session.server.q.get(block=True)
            # print(ret)
            return ret

    # TODO clean up: uncomment line below in production
    # session.stop_and_prune()


# We add times to the following function to record the overheads

if __name__ == '__main__':
    print("Gatekeeper starting.")
