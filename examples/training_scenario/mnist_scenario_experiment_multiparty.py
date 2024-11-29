import os
import sys
import numpy as np
import csv
import pandas as pd
from faker import Faker
import time
import pickle

from main import initialize_system
from common.general_utils import clean_test_env
from crypto import cryptoutils as cu

NUMBERS_DIR = "./examples/training_scenario/mnist_mp"

# code is modified from mnist_reader.py in fashion-mnist repository
def load_mnist(image_path, label_path):
    print("image_path: ", image_path)
    with open(label_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    print("label path: ", label_path)
    with open(image_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def split_mnist(image_path, label_path, num_agents):
    images, labels = load_mnist(image_path, label_path)
    image_splits = np.array_split(images, num_agents)
    label_splits = np.array_split(labels, num_agents)
    return image_splits, label_splits

def save_split_mnist(image_splits, label_splits, prefix):
    num_agents = len(image_splits)
    for i in range(num_agents):
        image_path = f"fashion-mnist/data/fashion/{prefix}-images-mp-{i+1}-{num_agents}"
        label_path = f"fashion-mnist/data/fashion/{prefix}-labels-mp-{i+1}-{num_agents}"
        with open(image_path, 'wb') as f:
            pickle.dump(image_splits[i], f)
        with open(label_path, 'wb') as f:
            pickle.dump(label_splits[i], f)

if __name__ == '__main__':

    # Experiment setups
    # num_MB = sys.argv[1]

    num_trials = int(sys.argv[1])

    # Clean up
    clean_test_env()
    
    num_agents = int(sys.argv[2])
    
    if os.path.exists(f"fashion-mnist/data/fashion/train-images-mp-1-{num_agents}"):
        print("Data already split")
    else:
        # Split the data
        train_image_path = "fashion-mnist/data/fashion/train-images-idx3-ubyte"
        train_label_path = "fashion-mnist/data/fashion/train-labels-idx1-ubyte"
        test_image_path = "fashion-mnist/data/fashion/t10k-images-idx3-ubyte"
        test_label_path = "fashion-mnist/data/fashion/t10k-labels-idx1-ubyte"
        
        train_image_splits, train_label_splits = split_mnist(train_image_path, train_label_path, num_agents)
        test_image_splits, test_label_splits = split_mnist(test_image_path, test_label_path, num_agents)
        
        save_split_mnist(train_image_splits, train_label_splits, "train")
        save_split_mnist(test_image_splits, test_label_splits, "t10k")
        # sys.exit(0)

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
        else:
            cur_token = agent_2_token
        # agent 1 has train, agent 2 has test. This doesn't matter because 
        # integration in this case is simply an append.
        agent_des = []
        for prefix in ["train", "t10k"]:
            agent_des.append(f"fashion-mnist/data/fashion/{prefix}-images-mp-{agent}-{num_agents}")
            agent_des.append(f"fashion-mnist/data/fashion/{prefix}-labels-mp-{agent}-{num_agents}")
        # agent_de = f"integration_new/test_files/advertising_p/exp/{agent}_{int(num_MB)}.csv"
        print(agent_des)
        for agent_de in agent_des:
            f = open(agent_de, "rb")
            plaintext_bytes = f.read()
            f.close()
            print(ds.call_api(cur_token, "upload_data_in_csv", plaintext_bytes))

    # Step 3: Train the joint model.
    dest_agents = [1]
    data_elements = range(1, num_agents * 4 + 1)
    print(data_elements)
    f = "train_mnist"
    
    # Create experiment directory
    os.makedirs(NUMBERS_DIR, exist_ok=True)

    for datasize in [7500, 15000, 30000, 60000]:
    # for datasize in [15000, 30000, 60000]:
        api_info = ds.call_api(agent_1_token, "propose_contract",
                        dest_agents, data_elements, f,
                        # function parameters
                        datasize,
                        num_agents
                        )
        contract_id = api_info['contract_id']
        print(api_info, contract_id)
        print(ds.call_api(agent_1_token, "approve_contract", contract_id))
        print(ds.call_api(agent_2_token, "approve_contract", contract_id))


        for _ in range(num_trials):
            run_start_time = time.perf_counter()
            res = ds.call_api(agent_1_token, f, datasize, num_agents)
            run_end_time = time.perf_counter()
            print("result: ", res)
            
            # res_df = pd.DataFrame(res)
            res['result'].to_csv(f"{NUMBERS_DIR}/mnist-mp-{num_agents}.csv", mode='a', index=False)
        # exp_start = res["experiment_time_arr"][0]
        # exp_end = res["experiment_time_arr"][1]
        # decrypt_time = res["experiment_time_arr"][2]
        # print("Experiment time:", exp_end - exp_start)
        # print("Decrypt time:", decrypt_time)
        # # 1: fixed overhead 2: join DE time 3: model train time 4: fixed overhead
            with open(f"{NUMBERS_DIR}/mnist_total_time-mp-{num_agents}.csv", "a") as file:
                writer = csv.writer(file)
                if res["result"] is not None:
                    writer.writerow([run_end_time - run_start_time, datasize])

    # Last step: shut down the Data Station
    ds.shut_down()
