import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .model import model, device
from .data import NUM_TOKENS, PAD_TOKEN, training_data

batch_size = 1 # you lose implicit regularization by doing this, but it will work on a macbook for testing
if torch.cuda.is_available():
    batch_size = 8 # this seems to fit on an a100 with 8192 context


# Prepare data loaders
train_dataset = TensorDataset(torch.tensor([x[0] for x in training_data], dtype=torch.long),
                              torch.tensor([x[1] for x in training_data], dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # ignore padding tokens
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print more detailed information
        if batch_idx % NUM_TOKENS == 0:  # Print every 100 batches
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            print("Predicted:", torch.argmax(output[0, 0, :NUM_TOKENS], dim=-1))
            print("Softmax of output:", torch.softmax(output[0, 0, :NUM_TOKENS], dim=-1))
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'checkpoint_epoch_{epoch+1}.pt')

print("Training completed.")