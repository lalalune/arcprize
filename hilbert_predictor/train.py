import torch
import torch.nn as nn
import torch.nn.functional as F
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
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # ignore padding tokens


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = torch.ones_like(target).float()
        weight[target == 0] = 0.2  # Adjust this value to change the weight of 0 tokens
        weight[target == self.ignore_index] = 0

        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index, reduction='sum') / weight.sum()

criterion = WeightedCrossEntropyLoss(ignore_index=PAD_TOKEN)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths, tgt_lengths = src_lengths.to(device), tgt_lengths.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_lengths, tgt_lengths)  # Remove [:, :-1] from tgt_lengths
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print more detailed information
        if batch_idx % 10 == 0:  # Print every 10 batches
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

