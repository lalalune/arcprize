import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .model import (
    model,
    device,
    checkpoint_path,
    max_context_length,
    max_prediction_length,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    dropout_rate,
)
from .data import (
    NUM_TOKENS,
    PAD_TOKEN,
    START_SEQUENCE_TOKEN,
    END_SEQUENCE_TOKEN,
    training_data,
)
from torch.cuda.amp import autocast, GradScaler
import wandb
from torch.nn import CrossEntropyLoss

batch_size = 32 if torch.cuda.is_available() else 1
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


def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(src[:max_context_length], dtype=torch.long) for src in src_list],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(tgt[:max_prediction_length], dtype=torch.long) for tgt in tgt_list],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    return src_padded, tgt_padded

def train_step(model, src, tgt, criterion):
    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")  # Debugging shape of inputs

    with autocast(enabled=use_amp):
        generated = model.generate(src, max_length=tgt.size(1))
        print(f"Shape after generation: {generated.shape}")  # Shape after token generation
        
        # Ensure generated sequence is the same length as target
        generated = generated[:, :tgt.size(1)]
        
        print(f"Shape after slicing to target length: {generated.shape}")  # Check shape after slicing

        # Flatten for cross-entropy loss
        logits = generated.reshape(-1, NUM_TOKENS + 1).float()  # Reshape logits correctly
        print(f"Logits reshaped for loss: {logits.shape}")  # Log reshaped logits shape

        tgt = tgt.contiguous().view(-1)  # Flatten target
        print(f"Target reshaped for loss: {tgt.shape}")  # Log reshaped target shape
        
        # Calculate loss
        loss = criterion(logits, tgt)
        print(f"Calculated loss: {loss.item()}")  # Output the loss

    return loss



if __name__ == "__main__":
    criterion = CrossEntropyLoss(ignore_index=PAD_TOKEN)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    
    num_epochs = 50

    wandb.init(project="hilbert_predictor", config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout_rate": dropout_rate,
    })

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            with autocast(enabled=use_amp):
                loss = train_step(model, src, tgt, criterion)

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
        
        wandb.log({"epoch": epoch+1, "avg_loss": avg_loss})
        wandb.save(str(checkpoint_path))

    print("Training completed.")