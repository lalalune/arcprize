import torch
import torch.nn as nn
import math
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import AttentionConfig, ScaledDotProduct
from .data import NUM_TOKENS, PAD_TOKEN
from torch.utils.checkpoint import checkpoint


# Model initialization tiny
batch_size = 1
if torch.cuda.is_available():
    batch_size = 256
d_model = 16
nhead = 8
num_layers = 3
dim_feedforward = 128
max_seq_length = 384
max_context_length = 256
max_prediction_length = 128
dropout_rate = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("checkpoint.pt")

# Model initialization small
batch_size = 1
if torch.cuda.is_available():
    batch_size = 32
d_model = 32
nhead = 4
num_layers = 4
dim_feedforward = 256
max_seq_length = 384
max_context_length = 256
max_prediction_length = 128
dropout_rate = 0.05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("checkpoint_sm.pt")

# Model initialization large
# batch_size = 1
# if torch.cuda.is_available():
#     batch_size = 48
# d_model = 1024
# nhead = 4
# num_layers = 6
# dim_feedforward = 2048
# max_seq_length = 4096
# max_context_length = 3072
# max_prediction_length = 1024
# dropout_rate = 0.05
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint_path = Path("checkpoint_lg.pt")


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        dropout_rate,
        device,
    ):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=PAD_TOKEN)

        self.layers = nn.ModuleList(
            [
                self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, num_tokens + 1)
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
        attention = ScaledDotProduct(dropout=dropout_rate, causal=True)

        return nn.ModuleDict(
            {
                "self_attn": MultiHeadDispatch(
                    dim_model=d_model,
                    num_heads=nhead,
                    attention=attention,
                    bias=True,
                    residual_dropout=dropout_rate,
                ),
                "ff": nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout_rate),
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
            }
        )

    def forward(self, src):
        assert src.dim() == 2, f"Expected input to be 2D, but got {src.dim()}D"

        x = self.token_embedding(src)
        assert x.shape == (
            src.shape[0],
            src.shape[1],
            self.d_model,
        ), f"Expected shape {(src.shape[0], src.shape[1], self.d_model)}, but got {x.shape}"

        batch_size, seq_len, d_model = x.shape

        for i, layer in enumerate(self.layers):
            q, k, v = x, x, x

            q = q.view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            k = k.view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            v = v.view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)

            assert (
                q.shape
                == k.shape
                == v.shape
                == (seq_len, batch_size, self.nhead, d_model // self.nhead)
            ), f"Expected shape {(seq_len, batch_size, self.nhead, d_model // self.nhead)}, but got q: {q.shape}, k: {k.shape}, v: {v.shape}"

            q = q.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            k = k.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            v = v.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)

            attn_output = checkpoint(layer["self_attn"], q, k, v)

            assert attn_output.shape == (
                batch_size,
                seq_len,
                d_model,
            ), f"Expected shape {(batch_size, seq_len, d_model)}, but got {attn_output.shape}"

            x = layer["norm1"](x + attn_output)
            assert x.shape == (
                batch_size,
                seq_len,
                d_model,
            ), f"Expected shape {(batch_size, seq_len, d_model)}, but got {x.shape}"

            ff_output = checkpoint(layer["ff"], x)

            assert ff_output.shape == (
                batch_size,
                seq_len,
                d_model,
            ), f"Expected shape {(batch_size, seq_len, d_model)}, but got {ff_output.shape}"

            x = layer["norm2"](x + ff_output)
            assert x.shape == (
                batch_size,
                seq_len,
                d_model,
            ), f"Expected shape {(batch_size, seq_len, d_model)}, but got {x.shape}"

        output = self.fc_out(x)
        assert output.shape == (
            batch_size,
            seq_len,
            NUM_TOKENS + 1,
        ), f"Expected shape {(batch_size, seq_len, NUM_TOKENS + 1)}, but got {output.shape}"

        return output


model = DecoderOnlyTransformer(
    NUM_TOKENS,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_seq_length,
    dropout_rate,
    device,
)

import torch
from .model import (
    DecoderOnlyTransformer,
    NUM_TOKENS,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_seq_length,
    dropout_rate,
    device,
    batch_size,
)


def test_model_with_zeros():
    # Create a model instance
    model = DecoderOnlyTransformer(
        NUM_TOKENS,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        dropout_rate,
        device,
    )

    # Create dummy input data (zeros)
    dummy_input = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)

    # Set the model to training mode
    model.train()

    # Create a dummy target (zeros)
    dummy_target = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(
        device
    )

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Perform a forward pass
    output = model(dummy_input)

    # Calculate loss
    loss = criterion(output.view(-1, NUM_TOKENS + 1), dummy_target.view(-1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Test completed. Loss: {loss.item()}")


if __name__ == "__main__":
    test_model_with_zeros()
