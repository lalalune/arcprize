import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import argparse
from .model import (
    model,
    checkpoint_path,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_seq_length,
    max_context_length,
    max_prediction_length,
    dropout_rate,
    device,
    batch_size,
)
from .data import (
    NUM_TOKENS,
    PAD_TOKEN,
    START_SEQUENCE_TOKEN,
    END_SEQUENCE_TOKEN,
    training_data,
)
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss


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
        logits = model(src)  # Assuming model now outputs logits directly

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

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

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

        if args.wandb:
            # Log epoch metrics to Wandb
            wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

            # Save model checkpoint to Wandb
            wandb.save(str(checkpoint_path))

    print("Training completed.")
