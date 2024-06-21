from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from .data import padded_train_data
from .model import TransformerModel

num_tokens = 10
# Assuming `padded_train_data` is already loaded and preprocessed
train_inputs = np.array(padded_train_data)
train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_epochs = 5
checkpoint_interval = 1
checkpoint_path = Path('checkpoint.pt')
num_context_tokens = 1024
num_pred_tokens = 1024

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

model = TransformerModel(num_tokens=num_tokens, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                         num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)

train(model, train_loader, num_epochs, checkpoint_interval, checkpoint_path, device)