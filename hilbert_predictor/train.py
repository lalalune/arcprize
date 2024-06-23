import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from .model import (
    model,
    checkpoint_path,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    MAX_CONTEXT_LENGTH,
    MAX_PREDICTION_LENGTH,
    dropout_rate,
    device,
    batch_size,
)
from .data import (
    NUM_TOKENS,
    PAD_TOKEN,
    training_data,
)
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from collections import deque
from typing import Dict, Optional, Literal

def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(src[:MAX_CONTEXT_LENGTH], dtype=torch.long) for src in src_list],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [
            torch.tensor(tgt[:MAX_PREDICTION_LENGTH], dtype=torch.long)
            for tgt in tgt_list
        ],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    src_lengths = torch.LongTensor(
        [min(len(src), MAX_CONTEXT_LENGTH) for src in src_list]
    )
    tgt_lengths = torch.LongTensor(
        [min(len(tgt), MAX_PREDICTION_LENGTH) for tgt in tgt_list]
    )
    return src_padded, tgt_padded, src_lengths, tgt_lengths


def train_step(
    model,
    src,
    tgt,
    src_lengths,
    tgt_lengths,
    criterion,
    train_loader,
    teacher_forcing_ratio=1.0,
):
    with autocast(enabled=use_amp):
        dimensions = src.size(0), src.size(1)  # Pass the dimensions of the input
        logits = model(src, dimensions)  # Pass dimensions to the model

        logits = logits[
            :, : tgt.size(1), :
        ]  # Ensure logits are the same length as targets

        # Flatten for cross-entropy loss
        logits = logits.reshape(
            -1, NUM_TOKENS + 1
        )  # Reshape to [batch_size * sequence_length, NUM_TOKENS + 1]
        tgt = tgt.view(-1)  # Flatten target

        # Calculate loss
        loss = criterion(logits, tgt)

    return logits, loss


if __name__ == "__main__":
    start_epoch = 0
    grads = None
    accumulation_steps = 16  # Accumulate gradients over 16 batches
    use_amp = True  # Use Automatic Mixed Precision
    print("Training the model.")
    print("training_data", training_data[0])
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.tensor([x[0] for x in training_data], dtype=torch.long),
        torch.tensor([x[1] for x in training_data], dtype=torch.long),
    )

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        grads = checkpoint.get('grads', None)
        
        
    scaler = GradScaler(enabled=use_amp)

    # Load checkpoint if it exists
    start_epoch = 0
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    args = parser.parse_args()

    criterion = CrossEntropyLoss(ignore_index=PAD_TOKEN)
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.tensor([x[0] for x in training_data], dtype=torch.long),
        torch.tensor([x[1] for x in training_data], dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    # Training loop
    num_epochs = 10000

    if args.wandb:
        import wandb
        wandb.init(
            project="hilbert_predictor",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "accumulation_steps": accumulation_steps,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout_rate": dropout_rate,
            },
            resume="auto"
        )
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")


    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            src_lengths, tgt_lengths = src_lengths.to(device), tgt_lengths.to(device)

            with autocast(enabled=use_amp):
                generated_ids, loss = train_step(
                    model, src, tgt, src_lengths, tgt_lengths, criterion, train_loader
                )

            scaler.scale(loss).backward()
            
            grads = gradfilter_ma(model, grads=grads)

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        # If there are any accumulation steps left, do a final step
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
                "loss": loss.item(),
                "grads": grads,  # Add this line
            },
            checkpoint_path,
        )

        if args.wandb:
            # Log epoch metrics to Wandb
            wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

            # Save model checkpoint to Wandb
            wandb.save(str(checkpoint_path))

    print("Training completed.")
