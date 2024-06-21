import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from .model import (
    model,
    device,
    checkpoint_path,
    max_context_length,
    max_prediction_length,
)
from .data import (
    NUM_TOKENS,
    PAD_TOKEN,
    START_SEQUENCE_TOKEN,
    END_SEQUENCE_TOKEN,
    training_data,
)
from torch.cuda.amp import autocast, GradScaler

batch_size = 1
if torch.cuda.is_available():
    batch_size = 32
accumulation_steps = 16  # Accumulate gradients over 16 batches
use_amp = True  # Use Automatic Mixed Precision

# Prepare data loaders
train_dataset = TensorDataset(
    torch.tensor([x[0] for x in training_data], dtype=torch.long),
    torch.tensor([x[1] for x in training_data], dtype=torch.long),
)

# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler(enabled=use_amp)

# Load checkpoint if it exists
start_epoch = 0
if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-100, zero_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.zero_weight = zero_weight

    def forward(self, input, target):
        input = input.view(-1, self.num_classes + 1).float()  # Ensure input is float
        target = target.view(-1)

        log_probs = F.log_softmax(input, dim=1)

        non_ignored_mask = target != self.ignore_index
        weights = torch.ones_like(target, dtype=torch.float)
        weights[target == 0] = self.zero_weight
        weights[target == self.ignore_index] = 0

        loss = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1) * weights
        return loss.sum() / non_ignored_mask.sum()


criterion = WeightedCrossEntropyLoss(
    num_classes=NUM_TOKENS, ignore_index=PAD_TOKEN, zero_weight=0.1
)


def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(src[:max_context_length], dtype=torch.long) for src in src_list],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [
            torch.tensor(tgt[:max_prediction_length], dtype=torch.long)
            for tgt in tgt_list
        ],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    src_lengths = torch.LongTensor(
        [min(len(src), max_context_length) for src in src_list]
    )
    tgt_lengths = torch.LongTensor(
        [min(len(tgt), max_prediction_length) for tgt in tgt_list]
    )
    return src_padded, tgt_padded, src_lengths, tgt_lengths


train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
)


def train_step(model, src, tgt, src_lengths, tgt_lengths, criterion, teacher_forcing_ratio=1.0):
    batch_size, max_len = tgt.shape
    
    with autocast(enabled=use_amp):
        logits = model(src)  # Assuming model now outputs logits directly
        
        logits = logits[:, :tgt.size(1), :]  # Ensure logits are the same length as targets
        
        # Flatten for cross-entropy loss
        logits = logits.view(-1, NUM_TOKENS + 1)  # Reshape to [batch_size * sequence_length, NUM_TOKENS + 1]
        tgt = tgt.view(-1)  # Flatten target
        
        # Calculate loss
        loss = criterion(logits, tgt)

    return logits, loss



# Training loop
num_epochs = 50
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(train_loader):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths, tgt_lengths = src_lengths.to(device), tgt_lengths.to(device)

        with autocast(enabled=use_amp):
            generated_ids, loss = train_step(model, src, tgt, src_lengths, tgt_lengths, criterion)

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

        print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        if batch_idx % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_path,
            )

    # Perform the last optimization step if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        checkpoint_path,
    )

print("Training completed.")
