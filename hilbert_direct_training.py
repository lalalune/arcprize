from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

padded_test_data = []
num_tokens = 10
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

    for epoch in range(num_epochs):
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
    # Assuming `padded_test_data` is already loaded and preprocessed
    test_inputs = np.array(padded_test_data)
    test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TransformerModel(num_tokens=10, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                             num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate the model on the test set
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save predictions and actual values to a file
    model.eval()
    with torch.no_grad(), open('predictions.txt', 'w') as f:
        total_correct = 0
        total_samples = 0
        for i, src in enumerate(test_loader):
            src = src[0].to(device)
            output = model(src)
            _, predicted = torch.max(output.data, -1)
            
            target = src[:, model.num_context_tokens:]
            
            for j in range(src.size(0)):
                input_seq = src[j, :model.num_context_tokens].cpu().numpy()
                predicted_seq = predicted[j].cpu().numpy()
                target_seq = target[j].cpu().numpy()
                
                f.write(f"Example {total_samples + 1}:\n")
                f.write("Input: ")
                np.savetxt(f, input_seq.reshape(1, -1), fmt='%d', delimiter=',')
                f.write("Predicted: ")
                np.savetxt(f, predicted_seq.reshape(1, -1), fmt='%d', delimiter=',')
                f.write("Target: ")
                np.savetxt(f, target_seq.reshape(1, -1), fmt='%d', delimiter=',')
                
                correct = np.sum(predicted_seq == target_seq)
                total_correct += correct
                total_samples += 1
                
                f.write(f"Correct: {correct}/{len(target_seq)}\n\n")
    
    # Calculate the percentage of correct predictions
    percentage_correct = total_correct / (total_samples * num_pred_tokens) * 100
    print(f"Percentage of correct predictions: {percentage_correct:.2f}%")

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

num_epochs = 10
checkpoint_interval = 1
checkpoint_path = Path('checkpoint.pt')

train(model, train_loader, num_epochs, checkpoint_interval, checkpoint_path, device)

# eval(checkpoint_path, num_context_tokens, num_pred_tokens, device)