from dsapplicationregistration.dsar_core import api_endpoint, function
from escrowapi.escrow_api import EscrowAPI

import time
import duckdb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import pickle

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

def load_cifar(cifar_paths_train, cifar_path_test):
    """
    Takes in a list of 5 cifar paths plus the test path, and returns the data in np arrays
    """
    X_train = None
    y_train = None
    for cifar_path in cifar_paths_train:
        with open(cifar_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        
        if X_train is None:
            X_train = data[b'data']
            y_train = data[b'labels']
        else:
            X_train = np.vstack((X_train, data[b'data']))
            y_train = np.hstack((y_train, data[b'labels']))
    with open(cifar_path_test, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    X_test = data[b'data']
    y_test = data[b'labels']
    return X_train, y_train, X_test, y_test

def get_np_cifar_data(train_data_idxs, test_data_idx):
    """
    Returns the cifar data in np arrays
    """
    cifar_paths_train = []
    for idx in train_data_idxs:
        cifar_paths_train.append(EscrowAPI.CSVDEStore.read(idx))
    cifar_path_test = EscrowAPI.CSVDEStore.read(test_data_idx)
    X_train, y_train, X_test, y_test = load_cifar(cifar_paths_train, cifar_path_test)
    return X_train, y_train, X_test, y_test

def create_cifar_dataloader(X_train, y_train, X_test, y_test, data_size, batch_size):
    """
    Given sets of train and test data in np arrays, turns the cifar data into dataloaders
    """
    # convert from numpy to tensor and normalize
    X_train = torch.from_numpy(X_train).float()/255.0
    X_test = torch.from_numpy(X_test).float()/255.0
    
    # convert to tensor
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # normalize over -1 to 1
    X_train = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(X_train)
    X_test = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(X_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_dataset = Subset(train_dataset, range(data_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


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

def unpickle_mnist(image_path, label_path):
    with open(image_path, 'rb') as f:
        images = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    return images, labels

def get_raw_mnist_data(train_image_idx, train_label_idx, test_image_idx, test_label_idx):
    """
    Given the indices of the mnist data elements, returns the raw data
    """
    X_train, y_train = load_mnist(
        EscrowAPI.CSVDEStore.read(train_image_idx),
        EscrowAPI.CSVDEStore.read(train_label_idx)
    )
    X_test, y_test = load_mnist(
        EscrowAPI.CSVDEStore.read(test_image_idx),
        EscrowAPI.CSVDEStore.read(test_label_idx)
    )
    
    return X_train, y_train, X_test, y_test


def create_mnist_dataloader(X_train, y_train, X_test, y_test, data_size, batch_size):
    """
    Given sets of train and test data, turns the mnist data into dataloaders
    """
    
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
def train_cifar(datasize):
    """
    trains a simple model on cifar dataset
    """
    torch.set_num_threads(1)

    # return dataframe
    return_df = pd.DataFrame(columns=["epoch_duration", "epoch", "batch_size", "data_size", "accuracy", "test_duration"])

    class CifarNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    X_train, y_train, X_test, y_test = get_np_cifar_data([1, 2, 3, 4, 5], 6)
    
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 4
    epochs = 3
    
    # train model
    model = CifarNetwork().to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        train_dataloader_subset, test_dataloader = create_cifar_dataloader(X_train, y_train, X_test, y_test, datasize, batch_size)

        print(f"Epoch {epoch+1}\n-------------------------------")
        epoch_time_start = time.perf_counter()
        train(train_dataloader_subset, model, loss_fn, optimizer)
        epoch_time_end = time.perf_counter()

        test_start_time = epoch_time_end
        accuracy = test(test_dataloader, model, loss_fn)
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

            # print(f"Epoch {epoch+1} took {epoch_duration} seconds")
            # with open("datasize_cifar_nn.csv", "a") as fp:
            #     wr = csv.writer(fp, dialect='excel')
            #     # epoch_duration, epoch, batch_size, data_size, accuracy, test_duration
            #     wr.writerow([epoch_duration, epoch, batch_size, datasize, accuracy, test_duration])

    print("Done!")
    return return_df

@api_endpoint
@function
def train_mnist(datasize, num_parties):
    """
    trains a simple model on mnist dataset
    """
    torch.set_num_threads(1)

    
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
    
    train_image_idxs = []
    train_label_idxs = []
    test_image_idxs = []
    test_label_idxs = []
    for i in range(1, num_parties+1):
        addition = 4*i
        train_image_idxs.append(1+addition)
        train_label_idxs.append(2+addition)
        test_image_idxs.append(3+addition)
        test_label_idxs.append(4+addition)
    X_train, y_train, X_test, y_test = get_raw_mp_mnist_data(train_image_idxs, train_label_idxs, test_image_idxs, test_label_idxs)

    loss_fn = nn.CrossEntropyLoss()

    epochs = 3
    batch_size = 64
    # for datasize in [7500,15000,30000,60000]:
    
    # train model
    model = NeuralNetwork().to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        train_loader, test_loader = create_mnist_dataloader(X_train, y_train, X_test, y_test, datasize, 64)

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

def get_raw_mp_mnist_data(train_image_idxs, train_label_idxs, test_image_idxs, test_label_idxs):
    """
    Given the indices of the mnist data elements, returns the raw data
    """
    if len(train_image_idxs) != len(train_label_idxs) or len(test_image_idxs) != len(test_label_idxs):
        print("train_image_idxs: ", train_image_idxs)
        print("The number of image and label data elements must be the same")
        raise ValueError("The number of image and label data elements must be the same")

    if len(train_image_idxs) == 1:
        return get_raw_mnist_data(train_image_idxs[0], train_label_idxs[0], test_image_idxs[0], test_label_idxs[0])

    X_train, y_train = unpickle_mnist(
        EscrowAPI.CSVDEStore.read(train_image_idxs[0]),
        EscrowAPI.CSVDEStore.read(train_label_idxs[0])
    )
    X_test, y_test = unpickle_mnist(
        EscrowAPI.CSVDEStore.read(test_image_idxs[0]),
        EscrowAPI.CSVDEStore.read(test_label_idxs[0])
    )
    
    for i in range(1, len(train_image_idxs)):
        X_train_temp, y_train_temp = unpickle_mnist(
            EscrowAPI.CSVDEStore.read(train_image_idxs[i]),
            EscrowAPI.CSVDEStore.read(train_label_idxs[i])
        )
        X_train = np.vstack((X_train, X_train_temp))
        y_train = np.hstack((y_train, y_train_temp))
    
    for i in range(1, len(test_image_idxs)):
        X_test_temp, y_test_temp = unpickle_mnist(
            EscrowAPI.CSVDEStore.read(test_image_idxs[i]),
            EscrowAPI.CSVDEStore.read(test_label_idxs[i])
        )
        X_test = np.vstack((X_test, X_test_temp))
        y_test = np.hstack((y_test, y_test_temp))
    
    return X_train, y_train, X_test, y_test