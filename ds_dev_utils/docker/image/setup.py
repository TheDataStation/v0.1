import sys
import time
import socket
import pickle
import os
import requests
from ast import literal_eval

from run_function import load_connectors
from Interceptor import interceptor
from escrow_api_docker import EscrowAPIDocker
from escrowapi.escrow_api import EscrowAPI
import multiprocessing
from dsapplicationregistration.dsar_core import get_registered_functions

def run_function(f_res_queue, function_name, f_args, f_kwargs):
    """
    runs the function from the function name and provided pickle file for inputs

    Parameters:
     function name

    Returns: function return value
    """

    list_of_functions = get_registered_functions()
    # print("list_of_apis:", list_of_apis)
    for cur_api in list_of_functions:
        if function_name == cur_api.__name__:
            # print("call", function_name)
            try:
                result = cur_api(*f_args, **f_kwargs)
                f_res_queue.put((result, EscrowAPI.get_comp().derived_des_to_create))
            except:
                f_res_queue.put((None, []))

def main():

    # Step 0: get all the information we need

    with open("/usr/src/ds/args.pkl", "rb") as f:
        config_dict_data_bytes = f.read()
        config_dict = pickle.loads(config_dict_data_bytes)

    curr_os = config_dict["operating_system"]

    if curr_os == "mac":
        addr = "http://host.docker.internal"
    elif curr_os == "linux":
        addr = "http://127.0.0.1"
    else:
        print("unsupported OS")
        addr = ""

    addr += ":3030"

    accessible_de = config_dict["accessible_de"]
    docker_id = config_dict["docker_id"]
    agents_symmetric_key = config_dict["agents_symmetric_key"]
    start_de_id = config_dict["start_de_id"]
    approved_de_sets = config_dict["approved_de_sets"]
    derived_de_origin_dict = config_dict["derived_de_origin_dict"]

    connector_dir = "/usr/src/ds/functions"
    load_connectors(connector_dir)
    print("setting up...")

    # Set up escrow_api docker
    escrow_api_docker = EscrowAPIDocker(accessible_de, agents_symmetric_key, start_de_id)
    EscrowAPI.set_comp(escrow_api_docker)

    # Step 1: Set up the interceptor
    accessible_data_set = set()
    accessible_data_key_dict = {}

    # Set up the key for application states
    app_state_path = "/mnt/data/app_state.pkl"
    accessible_data_set.add(app_state_path)
    # Set sym key for app_state to None if agents_symmetric_key is None
    if agents_symmetric_key is not None:
        accessible_data_key_dict[app_state_path] = agents_symmetric_key[0]
    else:
        accessible_data_key_dict[app_state_path] = None

    for cur_de in accessible_de:
        if cur_de.store_type == "csv":
            cur_de_name = f"{cur_de.id}.csv"
        elif cur_de.store_type == "object":
            cur_de_name = f"{cur_de.id}"
        cur_de_path = os.path.join("/mnt/data", str(cur_de.id), cur_de_name)
        accessible_data_set.add(cur_de_path)
        accessible_data_key_dict[cur_de_path] = cur_de.enc_key

    accessible_data_obj = (accessible_data_set, accessible_data_key_dict)

    print("Docker setup.py: all DEs are", accessible_data_set)
    print("Docker setup.py: all DEs with their symmetric keys are", accessible_data_key_dict)
    print("Docker setup.py: All approve DE sets are", approved_de_sets)

    manager = multiprocessing.Manager()

    accessible_data_dict = manager.dict()
    data_accessed_dict = manager.dict()
    experiment_time_dict = manager.dict()
    approved_de_sets_shared = manager.list(approved_de_sets)
    derived_de_origin_shared = manager.dict()
    derived_de_origin_shared.update(derived_de_origin_dict)

    accessible_data_dict[0] = accessible_data_obj

    storage_path = "/mnt/data"
    mount_path = "/mnt/data_mount"

    interceptor_process = multiprocessing.Process(target=interceptor.main,
                                                  args=(storage_path,
                                                        mount_path,
                                                        approved_de_sets_shared,
                                                        derived_de_origin_shared,
                                                        accessible_data_dict,
                                                        data_accessed_dict,
                                                        experiment_time_dict,))

    interceptor_process.start()

    print("starting interceptor...")
    counter = 0
    while not os.path.ismount(mount_path):
        time.sleep(1)
        counter += 1
        if counter == 10:
            print("mount time out")
            exit(1)
    print("Mounted {} to {}".format(storage_path, mount_path))
    print(os.path.dirname(os.path.realpath(__file__)))

    # # send signal that the container has started, and keep trying until it succeeds
    # #  stack overflow: https://stackoverflow.com/questions/27062099/python-requests-retrying-until-a-valid-response-is-received
    # cnt = 0
    # max_retry = 3
    # while cnt < max_retry:
    #     try:
    #         response = requests.get(f'{addr}/started', params=send_params)
    #         if response.status_code == requests.codes.ok:
    #             break
    #         else:
    #             raise RuntimeError(
    #                 "requests.get: wrong status code")
    #     except requests.exceptions.RequestException as e:
    #         time.sleep(2 ** cnt)
    #         cnt += 1
    #         if cnt >= max_retry:
    #             raise e
    #
    # print(response.content)

    # Step 2: Start a process to run the function

    # print("Start counting time......")
    # start = time.time()

    # request function to run
    send_params = {"docker_id": docker_id}
    response = requests.get(f'{addr}/get_function_dict', params=send_params)
    function_dict = pickle.loads(response.content)
    # print("function dictionary: ", function_dict)
    # print(time.time() - start)
    # start = time.time()

    f_start_time = time.time()

    f_res_queue = multiprocessing.Queue()
    f_process = multiprocessing.Process(target=run_function, args=(f_res_queue, function_dict["function"],
                                                                   function_dict["args"], function_dict["kwargs"]))
    f_process.start()
    # f_process_id = f_process.pid
    # print("Docker setup.py: PID for process running the function is", f_process_id)

    # Following code block is with short-circuiting
    ret = None
    while len(approved_de_sets_shared) > 0:
        print("Looping to get function return")
        try:
            ret = f_res_queue.get(timeout=2)
            break
        except:
            pass

    # If approve_de_sets is empty at this point, we set return value to None and terminate f_process
    if not len(approved_de_sets_shared):
        ret = None
        f_process.terminate()

    # # Following code block is without short-circuiting
    # ret = f_res_queue.get(block=True)

    f_end_time = time.time()

    # Check what is produced for this function run
    if ret is not None:
        print("Return value is", ret[0])
        print("Derived DEs to create is", ret[1])
    print(approved_de_sets_shared)
    print(dict(data_accessed_dict))
    # print(dict(decryption_time_dict))
    # print("Got function return......")
    # print(time.time() - start)
    # print(f"Size of returned value is {sys.getsizeof(ret)}")

    # if f_process_id in dict(data_accessed_dict):
    #     data_accessed = dict(data_accessed_dict)[f_process_id]
    # else:
    #     data_accessed = []
    if 0 in dict(data_accessed_dict):
        data_accessed = dict(data_accessed_dict)[0]
    else:
        data_accessed = []
    decryption_time = dict(experiment_time_dict)["total_decryption_time"]
    read_time = dict(experiment_time_dict)["total_read_time"]
    experiment_time_arr = [f_start_time, f_end_time, decryption_time, read_time]

    # Remove newly created files
    filtered_de_accessed = set(filter(lambda x: x.split("/")[-3] != "Staging_storage", data_accessed))
    # Remove app_state.pkl from de_accessed
    filtered_de_accessed.discard(app_state_path)

    return_value = None
    derived_des_to_create = []
    if ret is not None:
        return_value = ret[0]
        derived_des_to_create = ret[1]
    to_send_back = pickle.dumps({"return_value": return_value,
                                 "data_accessed": filtered_de_accessed,
                                 "derived_des_to_create": derived_des_to_create,
                                 "approved_de_sets": list(approved_de_sets_shared),
                                 "experiment_time_arr": experiment_time_arr})
    # print("To send back pickle constructed......")
    # print(time.time() - start)
    # start = time.time()

    # Before we return the result of the function, look at the data elements accessed
    # print("In setup.py: data accessed is", data_accessed_dict)
    # print(to_send_back)

    # send the return value of the function
    response = requests.post(f"{addr}/send_function_return",
                             data=to_send_back,
                             params=send_params,
                             )
    # print(response, response.content)
    # print("Send function return finished......")
    # print(time.time() - start)


if __name__ == '__main__':
    main()
