from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

padded_test_data = []
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
    def __init__(self, num_tokens, d_model, nhead, dim_feedforward, num_layers, device):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(num_tokens + 1, d_model)  # +1 for padding token
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc_out = nn.Linear(d_model, num_tokens + 1)  # output layer
        self.to(device)

    def forward(self, src):
        src = src.view(src.size(0), -1)
        src = self.embedding(src)
        output = self.transformer(src, src)
        output = self.fc_out(output)
        return output

    def backward(self, src, optimizer, criterion):
        optimizer.zero_grad()
        output = self(src)
        loss = criterion(output.view(-1, output.size(-1)), src.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

def train(model, loader, num_epochs, checkpoint_interval, checkpoint_path, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, num_epochs):
        for src in loader:
            src = src[0].to(device)
            loss = model.backward(src, optimizer, criterion)
            print(f"Epoch {epoch}, Loss: {loss}")

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

# Assuming `padded_train_data` is already loaded and preprocessed
train_inputs = np.array(padded_train_data)
train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TransformerModel(num_tokens=10, d_model=512, nhead=8, dim_feedforward=2048, num_layers=3, device=device)

num_epochs = 100
checkpoint_interval = 10
checkpoint_path = Path('checkpoint.pt')

train(model, train_loader, num_epochs, checkpoint_interval, checkpoint_path, device)