from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from .gilbert2d import unflatten_1d_to_2d, gilbert2d

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

    total_correct = 0
    total_tokens = 0
    all_accuracies = []

    with torch.no_grad():
        for batch, filename in zip(test_loader, filenames):
            src = batch[0].to(device)
            output = model(src)
            _, predicted = torch.max(output.data, -1)

            target = src[:, num_context_tokens:num_context_tokens+num_pred_tokens]
            
            # Remove padding tokens for accuracy calculation
            mask = target != 10
            predicted_no_pad = predicted[mask]
            target_no_pad = target[mask]
            
            correct = (predicted_no_pad == target_no_pad).sum().item()
            total_correct += correct
            total_tokens += target_no_pad.numel()

            batch_accuracy = correct / target_no_pad.numel() if target_no_pad.numel() > 0 else 0
            all_accuracies.append(batch_accuracy)

            for i in range(src.size(0)):
                input_seq = src[i, :num_context_tokens].cpu().numpy()
                predicted_seq = predicted[i].cpu().numpy()
                target_seq = src[i, num_context_tokens:num_context_tokens+num_pred_tokens].cpu().numpy()

                # remove all 10s from all sequences
                input_seq = input_seq[input_seq != 10]
                predicted_seq = predicted_seq[predicted_seq != 10]
                target_seq = target_seq[target_seq != 10]
                
                # resize predicted seq to be same length as target seq
                predicted_seq = predicted_seq[:len(target_seq)]
                
                # print the values of these
                print(f"Input: {input_seq}")
                print(f"Predicted: {predicted_seq}")
                print(f"Target: {target_seq}")

                plot_hilbert_curves(input_seq, predicted_seq, target_seq, i, filename)

    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Mean Batch Accuracy: {np.mean(all_accuracies):.4f}")
    print(f"Std Dev of Batch Accuracy: {np.std(all_accuracies):.4f}")


def plot_hilbert_curves(input_seq, predicted_seq, target_seq, sample_index, filename):
    # Remove padding tokens (value 10)
    input_seq = input_seq[input_seq != 10]
    predicted_seq = predicted_seq[predicted_seq != 10]
    target_seq = target_seq[target_seq != 10]
    
    # Ensure predicted_seq is the same length as target_seq
    predicted_seq = predicted_seq[:len(target_seq)]

    # Calculate the size of the grids
    input_height = int(np.ceil(np.sqrt(len(input_seq))))
    input_width = int(np.ceil(len(input_seq) / input_height))
    
    target_height = int(np.ceil(np.sqrt(len(target_seq))))
    target_width = int(np.ceil(len(target_seq) / target_height))

    # Unflatten the sequences into 2D grids
    input_grid = unflatten_1d_to_2d(input_seq, input_width, input_height)
    predicted_grid = unflatten_1d_to_2d(predicted_seq, target_width, target_height)
    target_grid = unflatten_1d_to_2d(target_seq, target_width, target_height)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    grids = [input_grid, predicted_grid, target_grid]
    titles = ['Input', 'Predicted', 'Target']

    for ax, grid, title in zip(axs, grids, titles):
        im = ax.imshow(grid, cmap='tab10', vmin=0, vmax=9)
        ax.set_title(title)
        ax.axis('off')

    plt.colorbar(im, ax=axs.ravel().tolist(), label='Token Value')
    plt.tight_layout()
    plt.savefig(f'prediction_plots/hilbert_prediction_{sample_index}_{filename}.png')
    plt.close(fig)

# Update the unflatten_1d_to_2d function in gilbert2d.py:
def unflatten_1d_to_2d(array_1d, width, height):
    array_2d = np.full((height, width), 10)  # Fill with padding value
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        if idx < len(array_1d):
            array_2d[y][x] = array_1d[idx]
        else:
            break
    return array_2d
    
model = TransformerModel(num_tokens=num_tokens, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                         num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)

# get the filename from the path without the extension
filenames = [os.path.splitext(os.path.basename(f))[0] for f in evaluating_file_paths]

eval(checkpoint_path, num_context_tokens, num_pred_tokens, device, filenames)
