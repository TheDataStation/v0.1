from dsapplicationregistration.dsar_core import api_endpoint, function
from escrowapi.escrow_api import EscrowAPI

import time
import duckdb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import pandas as pd

import numpy as np

@api_endpoint
def upload_data_in_csv(de_in_bytes):
    return EscrowAPI.CSVDEStore.write(de_in_bytes)

@api_endpoint
def propose_contract(dest_agents, des, f, *args, **kwargs):
    return EscrowAPI.propose_contract(dest_agents, des, f, *args, **kwargs)

@api_endpoint
def approve_contract(contract_id):
    return EscrowAPI.approve_contract(contract_id)

@api_endpoint
def upload_data_in_csv(content):
    return EscrowAPI.CSVDEStore.write(content)

# What should the income query look like? We will do a query replacement
# select * from facebook
# select facebook.firstname from facebook

# update_query will do replace("facebook") with read_csv_auto("path_to_facebook") as facebook
# (and similarly for YouTube)
def update_query(query):
    formatted = query.format(de1_filepath = EscrowAPI.CSVDEStore.read(1), de2_filepath = EscrowAPI.CSVDEStore.read(2))
    return formatted

@api_endpoint
@function
def run_query(query):
    """
    Run a user given query.
    """
    updated_query = update_query(query)
    print(updated_query)
    conn = duckdb.connect()
    res_df = conn.execute(updated_query).fetchdf()
    conn.close()
    return res_df

# code is modified from mnist_reader.py in fashion-mnist repository
def load_mnist(image_path, label_path):
    print("image_path: ", image_path)
    with open(label_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    print("label path: ", label_path)
    print("labels: ", labels)
    with open(image_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def create_mnist_dataloader(train_image_idx, train_label_idx, test_image_idx, test_label_idx, data_size, batch_size):
    X_train, y_train = load_mnist(
        EscrowAPI.CSVDEStore.read(train_image_idx),
        EscrowAPI.CSVDEStore.read(train_label_idx)
    )
    X_test, y_test = load_mnist(
        EscrowAPI.CSVDEStore.read(test_image_idx),
        EscrowAPI.CSVDEStore.read(test_label_idx)
    )
    
    print(X_train.shape)
    print(type(X_train))
    
    X_train = torch.from_numpy(X_train).float()/255.0
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()/255.0
    y_test = torch.from_numpy(y_test).long()
    
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataset = Subset(train_dataset, range(data_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cpu"), y.to("cpu")

        # Compute prediction error
        pred = model(X.float())
        # print("pred: ",pred.shape)
        # print("y:",y.shape)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

@api_endpoint
@function
def train_mnist(data_size):
    """
    trains a simple model on mnist dataset
    """
    
    # return dataframe
    return_df = pd.DataFrame(columns=["epoch_duration", "epoch", "batch_size", "data_size", "accuracy", "test_duration"])

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
        
    train_loader, test_loader = create_mnist_dataloader(1, 2, 3, 4, data_size, 64)
    
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3
    batch_size = 64
    for datasize in [7500,15000,30000,60000]:
        model = NeuralNetwork().to("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            # train_dataloader_subset = DataLoader(subset_training_data, batch_size=batch_size)
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_time_start = time.perf_counter()
            train(train_loader, model, loss_fn, optimizer)
            epoch_time_end = time.perf_counter()

            test_start_time = epoch_time_end
            accuracy = test(test_loader, model, loss_fn)
            test_end_time = time.perf_counter()

            epoch_duration = epoch_time_end - epoch_time_start
            test_duration = test_end_time - test_start_time

            print(f"Epoch {epoch+1} took {epoch_duration} seconds")
            return_df = pd.concat([return_df, pd.DataFrame({"epoch_duration": epoch_duration,
                                    "epoch": epoch,
                                    "batch_size": batch_size,
                                    "data_size": datasize,
                                    "accuracy": accuracy,
                                    "test_duration": test_duration}, index=[0])], ignore_index=True)

            # with open("datasize_nn.csv", "a") as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     # epoch_duration, epoch, batch_size, data_size, accuracy, test_duration
            #     wr.writerow([epoch_duration, epoch, batch_size, datasize, accuracy, test_duration])

    print("Done!")
    return return_df
