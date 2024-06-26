import os
import numpy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm

from collections import deque
from typing import Dict, Optional, Literal

from torch.utils.data import DataLoader, TensorDataset
from .model import (
    model,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    device,
)
from .args import checkpoint_path, dropout_rate, batch_size, use_schedulefree, use_grokfast, use_wandb
from .data import (
    END_OUTPUT_MATRIX_TOKEN,
    NUM_TOKENS,
    PAD_TOKEN,
    MAX_CONTEXT_LENGTH,
    MAX_PREDICTION_LENGTH,
    SPECIAL_TOKENS,
    training_data,
    is_special_token,
    SPECIAL_TOKENS
)

def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal["mean", "sum"] = "mean",
    warmup: bool = True,
    trigger: bool = False,  # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {
            n: deque(maxlen=window_size)
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

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
    src_list, tgt_list, dimensions_list = zip(*batch)
    # print(f"Batch dimensions: {dimensions_list}")
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [src[:MAX_CONTEXT_LENGTH] for src in src_list],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [tgt[:MAX_PREDICTION_LENGTH] for tgt in tgt_list],
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    src_lengths = torch.IntTensor(
        [min(len(src), MAX_CONTEXT_LENGTH) for src in src_list]
    )
    tgt_lengths = torch.IntTensor(
        [min(len(tgt), MAX_PREDICTION_LENGTH) for tgt in tgt_list]
    )
    return src_padded, tgt_padded, src_lengths, tgt_lengths, dimensions_list


last_loss = 0
confidence_history = deque(maxlen=10)
last_threshold = 0.5

def train_step(
    model,
    src,
    tgt,
    src_lengths,
    tgt_lengths,
    dimensions,
    criterion,
    confidence_values,  # Added parameter to collect confidence values
    teacher_forcing_ratio=1.2,
):
    batch_size, seq_len = tgt.size()
    outputs = torch.zeros(batch_size, seq_len, NUM_TOKENS + 1, device=device)

    # Find the first non-padding token
    first_non_pad = torch.where(
        src != PAD_TOKEN, src, src.new_tensor([src.shape[1]])
    ).min(dim=1)[1]
    input_token = src[torch.arange(src.size(0)), first_non_pad].unsqueeze(1)
    for t in range(seq_len - 1):
        logits, confidences = model(input_token, dimensions)
        
        # dynamic confidence thresholding
        # store average of the last 10 high confidence percentages
        if len(confidence_history) == 10:
            avg_confidence = sum(confidence_history) / len(confidence_history)
        else:
            avg_confidence = 0.5  # Default value if not enough history
        
        min_threshold = 0.6
        max_threshold = 0.999
        min_confidence = 50
        max_confidence = 90
        
        # Set threshold based on average confidence
        if avg_confidence <= min_confidence:
            threshold = min_threshold
        elif avg_confidence >= max_confidence:
            threshold = max_threshold
        else:
            threshold = min_threshold + (avg_confidence - min_confidence) * (max_threshold - min_threshold) / (max_confidence - min_confidence)
        
        global last_threshold
        last_threshold = threshold
        
        logits, confidences, refined_tokens, high_confidence_percentage = (
            model.refine_predictions(
                input_token, logits, confidences, dimensions, threshold=threshold
            )
        )
        confidence_history.append(high_confidence_percentage)

        # Collect confidence values for later averaging
        confidence_values.append(high_confidence_percentage)

        outputs[:, t, :] = logits.squeeze(1)

        global last_loss
        teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio - (
            1.0 - last_loss
        )

        if teacher_forcing:
            next_input = tgt[:, t + 1].unsqueeze(1)
        else:
            next_input = logits.argmax(-1)
            if next_input.dim() == 1:
                next_input = next_input.unsqueeze(1)
            elif next_input.dim() == 3:
                next_input = next_input.squeeze(1)

        special_mask = is_special_token(tgt[:, t + 1].unsqueeze(1), SPECIAL_TOKENS)
        input_token = torch.where(special_mask, tgt[:, t + 1].unsqueeze(1), next_input)

    loss_mask = ~is_special_token(tgt, SPECIAL_TOKENS)
    loss = criterion(
        outputs[loss_mask].view(-1, NUM_TOKENS + 1), tgt[loss_mask].view(-1)
    )

    last_loss = loss.item()

    return outputs, loss

if __name__ == "__main__":
    start_epoch = 0
    grads = None
    use_amp = False  # Use Automatic Mixed Precision
    print("Training the model.")
    # Loss function and optimizer
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except:
                print("Optimizer state dict not found in checkpoint.")
        start_epoch = checkpoint["epoch"] + 1
        grads = checkpoint.get("grads", None)

    # Load checkpoint if it exists
    start_epoch = 0

    criterion = CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.tensor(numpy.array([x[0] for x in training_data]), dtype=torch.long),
        torch.tensor(numpy.array([x[1] for x in training_data]), dtype=torch.long),
        torch.tensor(
            numpy.array([x[2] for x in training_data]), dtype=torch.long
        ),  # Dimensions
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Training loop
    num_epochs = 10000
    model.train()

    # Needed for schedule-free optimizer
    if use_schedulefree:
        model.optimizer.train()

    if use_wandb:
        import wandb

        wandb.init(
            project="hilbert_predictor",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout_rate": dropout_rate,
            },
            resume="auto",
        )

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        confidence_values = []
        model.optimizer.zero_grad()

        for batch_idx, (src, tgt, src_lengths, tgt_lengths, dimensions) in enumerate(
            train_loader
        ):
            src, tgt = src.to(device), tgt.to(device)
            src_lengths, tgt_lengths = src_lengths.to(device), tgt_lengths.to(device)

            with autocast(enabled=use_amp):
                generated_ids, loss = train_step(
                    model, src, tgt, src_lengths, tgt_lengths, dimensions, criterion, confidence_values
                )

            loss.backward()

            # fastgrok, ignored for now
            if use_grokfast:
                grads = gradfilter_ma(model, grads=grads)

                # Apply gradient clipping
                max_grad_norm = 1.0  # Adjust this value as needed
                clip_grad_norm(model.parameters(), max_grad_norm)

            model.optimizer.step()
            model.optimizer.zero_grad()

            if batch_idx % 100 == 0:
                # run a prediction, print the tokens, expected tokens and accuracy
                _, predictions = torch.max(generated_ids, dim=-1)

                # find the END_SEQUENCE_TOKEN
                end_sequence_token = torch.where(tgt == END_OUTPUT_MATRIX_TOKEN)[1]

                # trim the tgt tensor to the first END_SEQUENCE_TOKEN
                if len(end_sequence_token) > 0:
                    tgt_clipped = tgt[0, : end_sequence_token[0]]
                else:
                    tgt_clipped = tgt[0, :]

                # Clip predictions to match tgt_clipped length
                predictions_clipped = predictions[0, : len(tgt_clipped)]

                total = len(tgt_clipped)
                if total > 0:
                    correct = (predictions_clipped == tgt_clipped).sum().item()
                    accuracy = correct / total
                    print(f"Accuracy: {accuracy:.4f}")
                else:
                    print("No valid target tokens found for accuracy calculation.")

                print("Predicted: ", predictions_clipped.tolist())
                print("Expected:  ", tgt_clipped.tolist())
                print(f"Correct predictions: {correct} out of {total}")

            # Compute total batches
            print(
                f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f} - Loss for first token: {criterion(generated_ids[:, 0, :], tgt[:, 0])} - Threshold: {last_threshold:.2f}"
            )
            if confidence_values:  # Check to avoid division by zero
                average_confidence = sum(confidence_values) / len(confidence_values)
                print(f"Average High Confidence for Epoch {epoch+1}: {average_confidence:.2f}%")


            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
                "loss": loss.item(),
                "grads": grads,
            },
            checkpoint_path,
        )

        if use_wandb:
            wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})
            wandb.save(str(checkpoint_path))

    print("Training completed.")
