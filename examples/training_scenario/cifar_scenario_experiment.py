import os
import sys
import numpy as np
import csv
import pandas as pd
from faker import Faker
import time
from torchvision import datasets

from main import initialize_system
from common.general_utils import clean_test_env
from crypto import cryptoutils as cu

NUMBERS_DIR = "./examples/training_scenario/cifar"

if __name__ == '__main__':
    
    _ = datasets.CIFAR10(root='./cifar_data', train=True, download=True)
    _ = datasets.CIFAR10(root='./cifar_data', train=False, download=True)
    
    # Experiment setups
    # num_MB = sys.argv[1]

    num_trials = int(sys.argv[1])

    # Clean up
    clean_test_env()

    # Step 0: System initialization
    ds_config = "./examples/training_scenario/data_station_config.yaml"
    ds = initialize_system(ds_config)

    ds_public_key = ds.key_manager.ds_public_key

    log_path = ds.data_station_log.log_path
    if os.path.exists(log_path):
        os.remove(log_path)

    wal_path = ds.write_ahead_log.wal_path
    if os.path.exists(log_path):
        os.remove(log_path)

    # # Step 0.5: Generate the data (only need to run once).
    # # They will be stored to integration_new/test_files/advertising_p/exp folder.
    # advertising_data_gen(num_MB)

    # Step 1: Agent creation. 2 agents: facebook and youtube.
    cipher_sym_key_list = []
    public_key_list = []
    agents = ["1", "2"]
    for i in range(len(agents)):
        sym_key = b'oHRZBUvF_jn4N3GdUpnI6mW8mts-EB7atAUjhVNMI58='
        cipher_sym_key = cu.encrypt_data_with_public_key(sym_key, ds_public_key)
        cipher_sym_key_list.append(cipher_sym_key)
        cur_private_key, cur_public_key = cu.generate_private_public_key_pair()
        public_key_list.append(cur_public_key)
        ds.create_agent(agents[i], "string", cipher_sym_key_list[i], public_key_list[i], )

    agent_1_token = ds.login_agent("1", "string")["data"]
    agent_2_token = ds.login_agent("2", "string")["data"]

    # Step 2: Upload the data.
    for agent in agents:
        if agent == "1":
            cur_token = agent_1_token
            prefix = "train"
            # agent 1 has train, agent 2 has test. This doesn't matter because 
            # integration in this case is simply an append.
            agent_des = [f"cifar_data/cifar-10-batches-py/data_batch_1",
                f"cifar_data/cifar-10-batches-py/data_batch_2",
                f"cifar_data/cifar-10-batches-py/data_batch_3",
                f"cifar_data/cifar-10-batches-py/data_batch_4",
                f"cifar_data/cifar-10-batches-py/data_batch_5"]
        else:
            cur_token = agent_2_token
            prefix = "t10k"
            agent_des = [f"cifar_data/cifar-10-batches-py/test_batch"]
        
        # agent_de = f"integration_new/test_files/advertising_p/exp/{agent}_{int(num_MB)}.csv"
        for agent_de in agent_des:
            f = open(agent_de, "rb")
            plaintext_bytes = f.read()
            f.close()
            print(ds.call_api(cur_token, "upload_data_in_csv", plaintext_bytes))

    # Step 3: Train the joint model.
    dest_agents = [1]
    data_elements = [1, 2, 3, 4, 5, 6]
    f = "train_cifar"
    
    # Create experiment directory
    os.makedirs(NUMBERS_DIR, exist_ok=True)

    for datasize in [6250, 12500, 25000, 50000]:
        api_info = ds.call_api(agent_1_token, "propose_contract",
                        dest_agents, data_elements, f,
                        # function parameters
                        datasize
                        )
        contract_id = api_info['contract_id']
        print(api_info, contract_id)
        print(ds.call_api(agent_1_token, "approve_contract", contract_id))
        print(ds.call_api(agent_2_token, "approve_contract", contract_id))


        for _ in range(num_trials):
            run_start_time = time.perf_counter()
            res = ds.call_api(agent_1_token, f, datasize)
            run_end_time = time.perf_counter()
            print("result: ", res)
            
            # res_df = pd.DataFrame(res)
            res['result'].to_csv(f"{NUMBERS_DIR}/cifar.csv", mode='a', index=False)
            # exp_start = res["experiment_time_arr"][0]
            # exp_end = res["experiment_time_arr"][1]
            # decrypt_time = res["experiment_time_arr"][2]
            # print("Experiment time:", exp_end - exp_start)
            # print("Decrypt time:", decrypt_time)
            # # 1: fixed overhead 2: join DE time 3: model train time 4: fixed overhead
            with open(f"{NUMBERS_DIR}/cifar_total_time.csv", "a") as file:
                writer = csv.writer(file)
                if res["result"] is not None:
                    writer.writerow([run_end_time - run_start_time, datasize])
    # print("Time for all runs", time.perf_counter() - start_time)

    # Last step: shut down the Data Station
    ds.shut_down()
