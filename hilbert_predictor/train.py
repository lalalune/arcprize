from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from .data import padded_train_data
from .model import model, checkpoint_path, num_tokens

# Assuming `padded_train_data` is already loaded and preprocessed
train_inputs = np.array(padded_train_data)
train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_epochs = 5
checkpoint_interval = 1
num_context_tokens = 1024
num_pred_tokens = 1024

# Our general plan is to load up 2048 tokens with a forward attention mask, then predict token by token
# 

def train(model, loader, num_epochs, checkpoint_interval, checkpoint_path, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=10, label_smoothing=0.1)
    
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
            
            # Print more detailed information
            if i % 10 == 0:  # Print every 100 batches
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
                print("Output sample:", output[0, 0, :10])  # Print first 10 logits of first token
                print("Target sample:", target[:10])  # Print first 10 target tokens
                print("Predicted:", torch.argmax(output[0, 0, :10], dim=-1))
                print("Softmax of output:", torch.softmax(output[0, 0, :10], dim=-1))

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

train(model, train_loader, num_epochs, checkpoint_interval, checkpoint_path, device)