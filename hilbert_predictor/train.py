import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .model import model, device, checkpoint_path
from .data import NUM_TOKENS, PAD_TOKEN, training_data

batch_size = 1 # you lose implicit regularization by doing this, but it will work on a macbook for testing
if torch.cuda.is_available():
    batch_size = 8 # this seems to fit on an a100 with 8192 context


# Prepare data loaders
train_dataset = TensorDataset(torch.tensor([x[0] for x in training_data], dtype=torch.long),
                              torch.tensor([x[1] for x in training_data], dtype=torch.long))

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(src, dtype=torch.long) for src in src_list], batch_first=True, padding_value=PAD_TOKEN)
    tgt_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(tgt, dtype=torch.long) for tgt in tgt_list], batch_first=True, padding_value=PAD_TOKEN)
    src_lengths = torch.LongTensor([len(src) for src in src_list])
    tgt_lengths = torch.LongTensor([len(tgt) for tgt in tgt_list])
    return src_padded, tgt_padded, src_lengths, tgt_lengths



# Prepare data loaders
train_dataset = TensorDataset(torch.tensor([x[0] for x in training_data], dtype=torch.long),
                              torch.tensor([x[1] for x in training_data], dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-100, zero_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.zero_weight = zero_weight

    def forward(self, input, target):
        # input shape: (batch_size, num_classes, sequence_length)
        # target shape: (batch_size, sequence_length)
        
        log_probs = F.log_softmax(input, dim=1)
        
        # Create a mask for non-ignored indices
        non_ignored_mask = target != self.ignore_index
        
        # Create a weight tensor based on the target values
        weights = torch.ones_like(target, dtype=torch.float)
        weights[target == 0] = self.zero_weight
        weights[target == self.ignore_index] = 0
        
        # Calculate the loss
        loss = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1) * weights
        
        # Sum the loss and divide by the number of non-ignored elements
        return loss.sum() / non_ignored_mask.sum()

criterion = WeightedCrossEntropyLoss(num_classes=NUM_TOKENS, ignore_index=PAD_TOKEN, zero_weight=0.1)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths, tgt_lengths = src_lengths.to(device), tgt_lengths.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_lengths, tgt_lengths)
        loss = criterion(output.transpose(1, 2), tgt[:, 1:])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print more detailed information
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            print("Predicted:", torch.argmax(output[0, 0, :NUM_TOKENS], dim=-1))
            print("Softmax of output:", torch.softmax(output[0, 0, :NUM_TOKENS], dim=-1))
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)

print("Training completed.")


