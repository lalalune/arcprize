from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

from .data import load_data, padded_train_data, padded_test_data, evaluating_file_paths
from .model import TransformerModel
from matplotlib.colors import ListedColormap
import os

num_tokens = 10
# Assuming `padded_train_data` is already loaded and preprocessed
train_inputs = np.array(padded_train_data)
train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_epochs = 5
checkpoint_interval = 1
checkpoint_path = Path('checkpoint.pt')

num_context_tokens = 1024
num_pred_tokens = 1024

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

def eval(checkpoint_path, num_context_tokens, num_pred_tokens, device, filenames):
    test_inputs = np.array(padded_test_data)
    test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerModel(num_tokens=10, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                             num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    os.makedirs('prediction_plots', exist_ok=True)
    sample_counter = 0  # To keep track of which sample we're on

    with torch.no_grad():
        for src, filename in zip(test_loader, filenames):
            src = src[0].to(device)
            output = model(src)
            _, predicted = torch.max(output.data, -1)

            for i in range(src.size(0)):
                input_seq = src[i, :num_context_tokens].cpu().numpy()
                predicted_seq = predicted[i].cpu().numpy()
                target_seq = src[i, num_context_tokens:num_context_tokens+num_pred_tokens].cpu().numpy()

                # plot_prediction(input_seq, predicted_seq, target_seq, sample_counter, filename)
                sample_counter += 1

# def plot_prediction(input_seq, predicted_seq, target_seq, sample_index, filename):
#     # Remove padding tokens (value 10)
#     input_seq = input_seq[input_seq != 10]
#     predicted_seq = predicted_seq[predicted_seq != 10]
#     target_seq = target_seq[target_seq != 10]
    
#     # resize predicted_seq to match the length of target_seq
#     predicted_seq = np.pad(predicted_seq, (0, len(target_seq) - len(predicted_seq)), 'constant', constant_values=10)
    
#     # Calculate the size of the grid
#     # TODO: These are wrong. These values are NOT square, they can be rectangles, so they need to come from the input
#     # We can probably solve this by saving the input size with our data
#     input_size = int(np.sqrt(len(input_seq)))
#     pred_size = int(np.sqrt(len(target_seq)))
#     target_size = int(np.sqrt(len(target_seq)))
    
#     # Reshape the sequences into 2D grids
#     input_grid = input_seq.reshape(input_size, input_size)
#     predicted_grid = predicted_seq.reshape(pred_size, pred_size)
#     target_grid = target_seq.reshape(target_size, target_size)

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     grids = [input_grid, predicted_grid, target_grid]
#     titles = ['Input', 'Predicted', 'Target']

#     for ax, grid, title in zip(axs, grids, titles):
#         c = ax.pcolormesh(grid, cmap=colormap, rasterized=True, vmin=0, vmax=9)
#         ax.set_title(title)
#         ax.set_xticks(np.arange(0, grid.shape[1]+1, 1))
#         ax.set_yticks(np.arange(0, grid.shape[0]+1, 1))
#         ax.grid(which='both', color='w', linestyle='-', linewidth=2)
#         ax.set_aspect('equal')
#         fig.colorbar(c, ax=ax)

#     plt.tight_layout()
#     plt.savefig(f'prediction_plots/prediction_{sample_index}_{filename}.png')
#     plt.close(fig)
    
model = TransformerModel(num_tokens=num_tokens, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                         num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)

# get the filename from the path without the extension
filenames = [os.path.splitext(os.path.basename(f))[0] for f in evaluating_file_paths]

eval(checkpoint_path, num_context_tokens, num_pred_tokens, device, filenames)
