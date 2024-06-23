# %%
import os
import cv2
import json
import time

import numpy as np
import pandas as pd
from pathlib import Path
from keras.utils import to_categorical

import torch
T = torch.Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# %%
SIZE = 10
EPOCHS = 50
CONV_OUT_1 = 50
CONV_OUT_2 = 100
BATCH_SIZE = 128

TEST_PATH = Path('data/evaluation/')

# %%
test_task_files = sorted(os.listdir(TEST_PATH))

test_tasks = []
for task_file in test_task_files:
    with open(str(TEST_PATH / task_file), 'r') as f:
        task = json.load(f)
        test_tasks.append(task)

# %%
Xs_test, Xs_train, ys_train = [], [], []

for task in test_tasks:
    X_test, X_train, y_train = [], [], []

    for pair in task["test"]:
        X_test.append(pair["input"])

    for pair in task["train"]:
        X_train.append(pair["input"])
        y_train.append(pair["output"])
    
    Xs_test.append(X_test)
    Xs_train.append(X_train)
    ys_train.append(y_train)

# %%
matrices = []
for X_test in Xs_test:
    for X in X_test:
        matrices.append(X)
        
values = []
for matrix in matrices:
    for row in matrix:
        for value in row:
            values.append(value)
            
df = pd.DataFrame(values)
df.columns = ["values"]

# %%
data_path = Path('./data')
training_path = data_path / 'training'
training_tasks = sorted(os.listdir(training_path))

for i in [1, 19, 8, 15, 9]:

    task_file = str(training_path / training_tasks[i])

    with open(task_file, 'r') as f:
        task = json.load(f)

# %%
heights = [np.shape(matrix)[0] for matrix in matrices]
widths = [np.shape(matrix)[1] for matrix in matrices]

# %%
def replace_values(a, d):
    return np.array([d.get(i, -1) for i in range(a.min(), a.max() + 1)])[a - a.min()]

def repeat_matrix(a):
    return np.concatenate([a]*((SIZE // len(a)) + 1))[:SIZE]

def get_new_matrix(X):
    if len(set([np.array(x).shape for x in X])) > 1:
        X = np.array([X[0]])
    return X

def get_outp(outp, dictionary=None, replace=True):
    if replace:
        outp = replace_values(outp, dictionary)

    outp_matrix_dims = outp.shape
    outp_probs_len = outp.shape[0]*outp.shape[1]*10
    outp = to_categorical(outp.flatten(),
                          num_classes=10).flatten()

    return outp, outp_probs_len, outp_matrix_dims

# %% [markdown]
# ### PyTorch DataLoader

# %%
class ARCDataset(Dataset):
    def __init__(self, X, y, stage="train"):
        self.X = get_new_matrix(X)
        self.X = repeat_matrix(self.X)
        
        self.stage = stage
        if self.stage == "train":
            self.y = get_new_matrix(y)
            self.y = repeat_matrix(self.y)
        
    def __len__(self):
        return SIZE
    
    def __getitem__(self, idx):
        inp = self.X[idx]
        if self.stage == "train":
            outp = self.y[idx]

        if idx != 0:
            rep = np.arange(10)
            orig = np.arange(10)
            np.random.shuffle(rep)
            dictionary = dict(zip(orig, rep))
            inp = replace_values(inp, dictionary)
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = get_outp(outp, dictionary)

        if idx == 0:
            if self.stage == "train":
                outp, outp_probs_len, outp_matrix_dims = get_outp(outp, None, False)

        inp = torch.tensor(inp, dtype=torch.float32)  # Convert input to tensor
        if self.stage == "train":
            outp = torch.tensor(outp, dtype=torch.float32)  # Convert output to tensor
            outp_probs_len = torch.tensor(outp_probs_len, dtype=torch.int64)  # Convert outp_probs_len to tensor
            outp_matrix_dims = torch.tensor(outp_matrix_dims, dtype=torch.int64)  # Convert outp_matrix_dims to tensor
            return inp, outp, outp_probs_len, outp_matrix_dims, self.y
        else:
            return inp

# %%
class BasicCNNModel(nn.Module):
    def __init__(self, inp_dim=(10, 10), outp_dim=(10, 10)):
        super(BasicCNNModel, self).__init__()
        
        CONV_IN = 3
        KERNEL_SIZE = 3
        DENSE_IN = CONV_OUT_2
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dense_1 = nn.Linear(DENSE_IN, outp_dim[0]*outp_dim[1]*10)
        
        if inp_dim[0] < 5 or inp_dim[1] < 5:
            KERNEL_SIZE = 1

        self.conv2d_1 = nn.Conv2d(CONV_IN, CONV_OUT_1, kernel_size=KERNEL_SIZE)
        self.conv2d_2 = nn.Conv2d(CONV_OUT_1, CONV_OUT_2, kernel_size=KERNEL_SIZE)

    def forward(self, x, outp_dim):
        x = torch.cat([x.unsqueeze(0)]*3)
        x = x.permute((1, 0, 2, 3)).float()
        self.conv2d_1.in_features = x.shape[1]
        conv_1_out = self.relu(self.conv2d_1(x))
        self.conv2d_2.in_features = conv_1_out.shape[1]
        conv_2_out = self.relu(self.conv2d_2(conv_1_out))
        
        self.dense_1.out_features = outp_dim
        feature_vector, _ = torch.max(conv_2_out, 2)
        feature_vector, _ = torch.max(feature_vector, 2)
        logit_outputs = self.dense_1(feature_vector)
        
        out = []
        for idx in range(logit_outputs.shape[1]//10):
            out.append(self.softmax(logit_outputs[:, idx*10: (idx+1)*10]))
        return torch.cat(out, axis=1)


# %%
def transform_dim(inp_dim, outp_dim, test_dim):
    return (test_dim[0]*outp_dim[0]/inp_dim[0],
            test_dim[1]*outp_dim[1]/inp_dim[1])

def resize(x, test_dim, inp_dim):
    if inp_dim == test_dim:
        return x
    else:
        return cv2.resize(flt(x), inp_dim,
                          interpolation=cv2.INTER_AREA)

def flt(x): return np.float32(x)
def npy(x): return x.cpu().detach().numpy()
def itg(x): return np.int32(np.round(x))

# %%
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# %%
idx = 0
start = time.time()
test_predictions = []

for X_train, y_train in zip(Xs_train, ys_train):
    print("TASK " + str(idx + 1))

    train_set = ARCDataset(X_train, y_train, stage="train")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    inp_dim = np.array(X_train[0]).shape
    outp_dim = np.array(y_train[0]).shape
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    network = BasicCNNModel(inp_dim, outp_dim).to(device)
    optimizer = Adam(network.parameters(), lr=0.01)
    # Model initialization
    # Training loop
    for epoch in range(EPOCHS):
        for train_batch in train_loader:
            if train_set.stage == "train":
                train_X, train_y, out_d, d, out = [x.to(device) for x in train_batch]
            else:
                train_X = train_batch.to(device)
            train_preds = network(train_X, out_d)
            train_loss = nn.MSELoss()(train_preds, train_y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    out_d = torch.tensor(outp_dim[0]*outp_dim[1]*10, dtype=torch.int64)

    # Prediction
    X_test = np.array([resize(flt(X), np.shape(X), inp_dim) for X in Xs_test[idx-1]])
    for X in X_test:
        test_dim = np.array(T(X)).shape
        
        test_preds = npy(network(T(X).unsqueeze(0).to(device), out_d.to(device)).cpu())
        test_preds = np.argmax(test_preds.reshape((10, *outp_dim)), axis=0)
        test_predictions.append(itg(resize(test_preds, np.shape(test_preds), tuple(itg(transform_dim(inp_dim, outp_dim, test_dim))))))

    idx += 1

# %%
test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]

for idx, pred in enumerate(test_predictions):
    test_predictions[idx] = flattener(pred)

print(test_predictions)

import argparse
def main(eval_only=False):
    idx = 0
    start = time.time()
    test_predictions = []

    model_path = 'trained_model.pt'

    if eval_only and not os.path.exists(model_path):
        print("No trained model found for evaluation. Please train the model first.")
        return

    for X_train, y_train in zip(Xs_train, ys_train):
        print("TASK " + str(idx + 1))

        train_set = ARCDataset(X_train, y_train, stage="train")
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        inp_dim = np.array(X_train[0]).shape
        outp_dim = np.array(y_train[0]).shape

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        network = BasicCNNModel(inp_dim, outp_dim).to(device)

        if not eval_only:
            if os.path.exists(model_path):
                print("Loading existing trained model...")
                network.load_state_dict(torch.load(model_path))
            else:
                optimizer = Adam(network.parameters(), lr=0.01)

                # Training loop
                for epoch in range(EPOCHS):
                    for train_batch in train_loader:
                        train_X, train_y, out_d, d, out = [x for x in train_batch]
                        train_preds = network(train_X, out_d)
                        train_loss = nn.MSELoss()(train_preds, train_y)

                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                print("Saving trained model...")
                torch.save(network.state_dict(), model_path)

        # Prediction
        X_test = np.array([resize(flt(X), np.shape(X), inp_dim) for X in Xs_test[idx-1]])
        correct_predictions = 0
        total_predictions = len(X_test)

        for X, y in zip(X_test, ys_train[idx-1]):
            test_dim = np.array(T(X)).shape
            test_preds = npy(network(T(X).unsqueeze(0).to(device), out_d.to(device)).cpu())
            test_preds = np.argmax(test_preds.reshape((10, *outp_dim)), axis=0)
            test_predictions.append(itg(resize(test_preds, np.shape(test_preds), tuple(itg(transform_dim(inp_dim, outp_dim, test_dim))))))

            if np.array_equal(test_predictions[-1], y):
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions * 100
        print(f"Accuracy: {accuracy:.2f}%")

        idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate the trained model")
    args = parser.parse_args()

    main(eval_only=args.eval)
