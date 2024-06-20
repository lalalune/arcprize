from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

padded_test_data = []
num_tokens = 10

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# %%
# Gilbert2D - Generalized Hilbert Curve for 2D space-filling 
def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield(x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield(x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        yield from generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))
        
def flatten_2d_to_1d(array_2d):
    height, width = len(array_2d), len(array_2d[0])
    array_1d = [None] * (width * height)
    
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        array_1d[idx] = array_2d[y][x]
    
    return array_1d


def unflatten_1d_to_2d(array_1d, width, height):
    array_2d = [[None] * width for _ in range(height)]
    
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        array_2d[y][x] = array_1d[idx]
    
    return array_2d

# %%
# ARC Setup stuff
colors_rgb = {
    0: (0x00, 0x00, 0x00),
    1: (0x00, 0x74, 0xD9),
    2: (0xFF, 0x41, 0x36),
    3: (0x2E, 0xCC, 0x40),
    4: (0xFF, 0xDC, 0x00),
    5: (0xA0, 0xA0, 0xA0),
    6: (0xF0, 0x12, 0xBE),
    7: (0xFF, 0x85, 0x1B),
    8: (0x7F, 0xDB, 0xFF),
    9: (0x87, 0x0C, 0x25),
}

_float_colors = [tuple(c / 255 for c in col) for col in colors_rgb.values()]
colormap = ListedColormap(_float_colors)

class ArcColors:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    FUCHSIA = 6
    ORANGE = 7
    TEAL = 8
    BROWN = 9

def plot_grid(grid1: np.ndarray, grid2: np.ndarray = None):
    if grid2 is None:
        fig, ax = plt.subplots()
        ax.pcolormesh(
            grid1,
            cmap=colormap,
            rasterized=True,
            vmin=0,
            vmax=9,
        )
        ax.set_xticks(np.arange(0, grid1.shape[1], 1))
        ax.set_yticks(np.arange(0, grid1.shape[0], 1))
        ax.grid()
        ax.set_aspect(1)
        ax.invert_yaxis()
        plt.show()
        return
    
    fig, axs = plt.subplots(1, 2)

    axs[0].pcolormesh(
        grid1,
        cmap=colormap,
        rasterized=True,
        vmin=0,
        vmax=9,
    )
    axs[0].set_xticks(np.arange(0, grid1.shape[1], 1))
    axs[0].set_yticks(np.arange(0, grid1.shape[0], 1))
    axs[0].grid()
    axs[0].set_aspect(1)
    axs[0].invert_yaxis()

    axs[1].pcolormesh(
        grid2,
        cmap=colormap,
        rasterized=True,
        vmin=0,
        vmax=9,
    )
    axs[1].set_xticks(np.arange(0, grid2.shape[1], 1))
    axs[1].set_yticks(np.arange(0, grid2.shape[0], 1))
    axs[1].grid()
    axs[1].set_aspect(1)
    axs[1].invert_yaxis()
    plt.show()

def load_data(file_paths):
    train_data = []
    test_data = []
    for file_path in file_paths:
        rules_input = []
        rules_input_hilbert = []
        test_input = []
        test_input_hilbert = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data['train']:
                input_hilbert = flatten_2d_to_1d(np.array(item['input']))
                output_hilbert = flatten_2d_to_1d(np.array(item['output']))
                                
                rules_input.append([
                    np.array(item['input'], dtype=np.int64),
                    np.array(item['output'], dtype=np.int64)
                ])
                
                rules_input_hilbert.append([
                    np.array(input_hilbert, dtype=np.int64),
                    np.array(output_hilbert, dtype=np.int64)
                ])
                
            for item in data['test']:
                test_input.append([
                    np.array(item['input'], dtype=np.int64),
                    np.array(item['output'], dtype=np.int64)
                ])
                
                input_hilbert = flatten_2d_to_1d(np.array(item['input']))
                output_hilbert = flatten_2d_to_1d(np.array(item['output']))
                
                test_input_hilbert.append([
                    np.array(input_hilbert, dtype=np.int64),
                    np.array(output_hilbert, dtype=np.int64)
                ])
                
                
        train_data.append(rules_input)
        test_data.append(test_input)
    return train_data, test_data


# %%
def load_data(file_paths):
    train_data = []
    test_data = []
    for file_path in file_paths:
        rules_input = []
        rules_input_hilbert = []
        test_input = []
        test_input_hilbert = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data['train']:
                input_hilbert = flatten_2d_to_1d(np.array(item['input']))
                output_hilbert = flatten_2d_to_1d(np.array(item['output']))
                                
                rules_input.append([
                    np.array(item['input'], dtype=np.int64),
                    np.array(item['output'], dtype=np.int64)
                ])
                
                rules_input_hilbert.append([
                    np.array(input_hilbert, dtype=np.int64),
                    np.array(output_hilbert, dtype=np.int64)
                ])
                
            for item in data['test']:
                test_input.append([
                    np.array(item['input'], dtype=np.int64),
                    np.array(item['output'], dtype=np.int64)
                ])
                
                input_hilbert = flatten_2d_to_1d(np.array(item['input']))
                output_hilbert = flatten_2d_to_1d(np.array(item['output']))
                
                test_input_hilbert.append([
                    np.array(input_hilbert, dtype=np.int64),
                    np.array(output_hilbert, dtype=np.int64)
                ])
                
                
        # train_data.append(rules_input)
        # test_data.append(test_input)
        train_data.append(rules_input_hilbert)
        test_data.append(test_input_hilbert)
    return train_data, test_data

# %%
# Load training data
training_data_dir = "./data/training"
evaluating_data_dir = "./data/evaluation"

# get all files in training_data_dir that end with .json
training_file_paths = [os.path.join(training_data_dir, f) for f in os.listdir(training_data_dir) if f.endswith('.json')]
evaluating_file_paths = [os.path.join(evaluating_data_dir, f) for f in os.listdir(evaluating_data_dir) if f.endswith('.json')]

training_train_data, training_test_data = load_data(training_file_paths)
evaluating_train_data, evaluating_test_data = load_data(evaluating_file_paths)

print("Training data loaded")

# %%
# all data has different dimensions, from 3x3 up to 30x30, both square and rectangular
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification




embedding_dict = {
    0: 0,  # BLACK
    1: 1,  # BLUE
    2: 2,  # RED
    3: 3,  # GREEN
    4: 4,  # YELLOW
    5: 5,  # GREY
    6: 6,  # FUCHSIA
    7: 7,  # ORANGE
    8: 8,  # TEAL
    9: 9,  # BROWN
    'A': 'A',  # Empty/Padding
}

def pad_examples(examples, max_length=1024):
    padded_examples = []
    for example in examples:
        input_example, output_example = example
        input_padded = np.pad(input_example, (max_length - len(input_example), 0), 'constant', constant_values=10)
        output_padded = np.pad(output_example, (0, max_length - len(output_example)), 'constant', constant_values=10)
        padded_examples.append((input_padded, output_padded))
    return padded_examples

padded_train_data = [pad_examples(data) for data in training_train_data]
padded_test_data = [pad_examples(data) for data in training_test_data]

def pad_examples(examples, max_length=1024):
    padded_examples = []
    for example in examples:
        input_example, output_example = example
        input_padded = np.pad(input_example, (max_length - len(input_example), 0), 'constant', constant_values=10)
        output_padded = np.pad(output_example, (0, max_length - len(output_example)), 'constant', constant_values=10)
        padded_examples.append((input_padded, output_padded))
    return padded_examples

padded_train_data = [pad_examples(data) for data in training_train_data]
padded_test_data = [pad_examples(data) for data in training_test_data]

def save_to_file(data, file_path):
    with open(file_path, 'w') as f:
        for examples in data:
            for example in examples:
                input_example, output_example = example
                f.write(' '.join(map(str, input_example)) + '\t' + ' '.join(map(str, output_example)) + '\n')

save_to_file(padded_train_data, 'padded_train_data.txt')
save_to_file(padded_test_data, 'padded_test_data.txt')


## Split file here


with open('padded_test_data.txt', 'r') as f:
    for line in f:
        tokens = line.strip().split(' ')
        for i in range(len(tokens)):
            tokens[i] = int(tokens[i])
        input_example = np.array(tokens, dtype=np.int64)
        padded_test_data.append(input_example)
        
padded_train_data = []
with open('padded_train_data.txt', 'r') as f:
    for line in f:
        tokens = line.strip().split(' ')
        # for each token, read as an int
        for i in range(len(tokens)):
            tokens[i] = int(tokens[i])
        input_example = np.array(tokens, dtype=np.int64)
        padded_train_data.append(input_example)

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, dim_feedforward, num_layers, num_context_tokens, num_pred_tokens, device):
        super().__init__()
        self.device = device
        self.num_context_tokens = num_context_tokens
        self.num_pred_tokens = num_pred_tokens
        self.embedding = nn.Embedding(num_tokens + 1, d_model)  # +1 for padding token
        
        # Initialize padding token embedding to zero
        self.embedding.weight.data[10] = torch.zeros(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True  # This should match your input dimensions

        )
        self.fc_out = nn.Linear(d_model, num_tokens + 1)  # output layer
        self.to(device)


    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones((sz, sz), device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def forward(self, src):
        # Mask creation
        src_key_padding_mask = (src == 10)  # True where src is padding

        # Apply to correct portions of src for context and target
        memory_key_padding_mask = src_key_padding_mask[:, :self.num_context_tokens]
        tgt_key_padding_mask = src_key_padding_mask[:, self.num_context_tokens:]

        context = self.embedding(src[:, :self.num_context_tokens])
        target = self.embedding(src[:, self.num_context_tokens:])

        # Generate target mask for sequence-to-sequence models
        target_mask = self.generate_square_subsequent_mask(target.size(1))

        # Make sure that the masks are correctly sized and Boolean type
        output = self.transformer(
            tgt=target,
            src=context,
            tgt_mask=target_mask,
            src_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output)
        return output

def train(model, loader, num_epochs, checkpoint_interval, checkpoint_path, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=10)
    
    start_epoch = 0  # Default start epoch

    # Load checkpoint if it exists
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        for i, (src,) in enumerate(loader):
            src = src.to(device)
            output = model(src)
            target = src[:, model.num_context_tokens:].reshape(-1)
            loss = criterion(output.view(-1, num_tokens + 1), target.view(-1))

            if torch.isnan(loss).any():
                print(f"NaN loss detected at batch {i} of epoch {epoch}")
                print("Output:", output)
                print("Target:", target)
                return  # Exit to avoid corrupting weights

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

            # Save checkpoint at the interval
            if (epoch + 1) % checkpoint_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch + 1}")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")


            
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for src in loader:
            src = src[0].to(device)
            output = model(src)
            _, predicted = torch.max(output.data, -1)
            
            target = src[:, model.num_context_tokens:]
            correct = (predicted == target).sum().item()
            total_correct += correct
            total_tokens += target.numel()
    
    accuracy = total_correct / total_tokens
    return accuracy

def eval(checkpoint_path, num_context_tokens, num_pred_tokens, device):
    # Load the preprocessed test data
    test_inputs = np.array(padded_test_data)
    test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerModel(num_tokens=10, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                             num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    model.eval()
    total_correct = 0
    total_tokens = 0
    total_samples = 0  # Initialize total_samples here

    with torch.no_grad(), open('predictions.txt', 'w') as f:
        for src in test_loader:
            src = src[0].to(device)
            output = model(src)
            _, predicted = torch.max(output.data, -1)

            target = src[:, model.num_context_tokens:]
            
            # Iterate over each example in the batch
            for i in range(src.size(0)):
                input_seq = src[i, :model.num_context_tokens].cpu().numpy()
                predicted_seq = predicted[i].cpu().numpy()
                target_seq = target[i].cpu().numpy()

                # Find the first padding token index in target sequence
                pad_index = np.where(target_seq == 10)[0]
                if pad_index.any():
                    end_index = pad_index[0]
                    predicted_seq = predicted_seq[:end_index]
                    target_seq = target_seq[:end_index]
                else:
                    end_index = len(target_seq)

                correct = (predicted_seq == target_seq).sum()
                total_correct += correct
                total_tokens += end_index

                # remove all 10s from the input sequence (leading padding)
                input_seq = input_seq[input_seq != 10]
                # Save trimmed and cleaned outputs
                total_samples += 1
                f.write(f"Example {total_samples}:\n")
                f.write("Input: ")
                np.savetxt(f, input_seq.reshape(1, -1), fmt='%d', delimiter=',')
                f.write("Predicted: ")
                np.savetxt(f, predicted_seq.reshape(1, -1), fmt='%d', delimiter=',')
                f.write("Target: ")
                np.savetxt(f, target_seq.reshape(1, -1), fmt='%d', delimiter=',')
                f.write(f"Correct: {correct}/{end_index}\n\n")

    accuracy = total_correct / total_tokens
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Percentage of correct predictions: {(total_correct / total_tokens * 100):.2f}%")

# Assuming `padded_train_data` is already loaded and preprocessed
train_inputs = np.array(padded_train_data)
train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_context_tokens = 1024
num_pred_tokens = 1024
model = TransformerModel(num_tokens=10, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                         num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)

num_epochs = 5
checkpoint_interval = 1
checkpoint_path = Path('checkpoint.pt')

train(model, train_loader, num_epochs, checkpoint_interval, checkpoint_path, device)

eval(checkpoint_path, num_context_tokens, num_pred_tokens, device)