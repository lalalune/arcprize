import torch
import torch.nn as nn
import math
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from .data import NUM_TOKENS, PAD_TOKEN, MAX_CONTEXT_LENGTH, MAX_SEQUENCE_LENGTH, MAX_PREDICTION_LENGTH
from torch.utils.checkpoint import checkpoint
from .encoder import PositionEncoder, NUM_ENCODING_DIMENSIONS
from schedulefree import AdamWScheduleFree

# Model initialization tiny
batch_size = 1
if torch.cuda.is_available():
    batch_size = 2048
d_model = 128 - NUM_ENCODING_DIMENSIONS
nhead = 8
num_layers = 12
dim_feedforward = 1024
dropout_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("checkpoint.pt")

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        MAX_SEQUENCE_LENGTH,
        dropout_rate,
        device,
    ):
        super().__init__()
        self.device = device
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.d_model = d_model
        self.nhead = nhead
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=PAD_TOKEN)
        self.position_encoder = PositionEncoder(30, 30, device=device)

        self.layers = nn.ModuleList(
            [
                self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.optimizer = AdamWScheduleFree(self.parameters(), lr=0.01)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.fc_out = nn.Linear(d_model + NUM_ENCODING_DIMENSIONS, num_tokens + 1)
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
        attention = ScaledDotProduct(dropout=dropout_rate, causal=True)
        return nn.ModuleDict(
            {
                "self_attn": MultiHeadDispatch(
                    dim_model=d_model + NUM_ENCODING_DIMENSIONS,
                    num_heads=nhead,
                    attention=attention,
                    bias=True,
                    residual_dropout=dropout_rate,
                ),
                "ff": nn.Sequential(
                    nn.Linear(d_model + NUM_ENCODING_DIMENSIONS, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dim_feedforward, d_model + NUM_ENCODING_DIMENSIONS),
                    nn.Dropout(dropout_rate),
                ),
                "norm1": nn.LayerNorm(d_model + NUM_ENCODING_DIMENSIONS),
                "norm2": nn.LayerNorm(d_model + NUM_ENCODING_DIMENSIONS),
            }
        )

    def forward(self, src, dimensions):
        assert src.dim() == 2, f"Expected input to be 2D, but got {src.dim()}D"
        assert isinstance(dimensions, tuple), f"Expected dimensions to be a tuple, but got {type(dimensions)}"
        assert len(dimensions) == 2, f"Expected dimensions to have length 2, but got {len(dimensions)}"

        x = self.token_embedding(src)
        assert x.shape == (
            src.shape[0],
            src.shape[1],
            self.d_model,
        ), f"Expected shape {(src.shape[0], src.shape[1], self.d_model)}, but got {x.shape}"

        # Add position encodings
        x = self.position_encoder(x, dimensions)

        batch_size, seq_len, _ = x.shape

        for i, layer in enumerate(self.layers):
            q, k, v = x, x, x

            q = q.view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            k = k.view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            v = v.view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)

            assert (
                q.shape
                == k.shape
                == v.shape
                == (seq_len, batch_size, self.nhead, (self.d_model + NUM_ENCODING_DIMENSIONS) // self.nhead)
            ), f"Expected shape {(seq_len, batch_size, self.nhead, (self.d_model + NUM_ENCODING_DIMENSIONS) // self.nhead)}, but got q: {q.shape}, k: {k.shape}, v: {v.shape}"

            q = q.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            k = k.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            v = v.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)

            attn_output = checkpoint(layer["self_attn"], q, k, v)

            assert attn_output.shape == (
                batch_size,
                seq_len,
                self.d_model + NUM_ENCODING_DIMENSIONS,
            ), f"Expected shape {(batch_size, seq_len, self.d_model + NUM_ENCODING_DIMENSIONS)}, but got {attn_output.shape}"

            x = layer["norm1"](x + attn_output)
            assert x.shape == (
                batch_size,
                seq_len,
                self.d_model + NUM_ENCODING_DIMENSIONS,
            ), f"Expected shape {(batch_size, seq_len, self.d_model + NUM_ENCODING_DIMENSIONS)}, but got {x.shape}"

            ff_output = checkpoint(layer["ff"], x)

            assert ff_output.shape == (
                batch_size,
                seq_len,
                self.d_model + NUM_ENCODING_DIMENSIONS,
            ), f"Expected shape {(batch_size, seq_len, self.d_model + NUM_ENCODING_DIMENSIONS)}, but got {ff_output.shape}"

            x = layer["norm2"](x + ff_output)
            assert x.shape == (
                batch_size,
                seq_len,
                self.d_model + NUM_ENCODING_DIMENSIONS,
            ), f"Expected shape {(batch_size, seq_len, self.d_model + NUM_ENCODING_DIMENSIONS)}, but got {x.shape}"

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
    MAX_SEQUENCE_LENGTH,
    dropout_rate,
    device,
)


def test_model_with_zeros():
    # Create dummy input data (zeros)
    dummy_input = torch.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=torch.long).to(device)

    # Create dummy dimensions
    dummy_dimensions = (10, 10)  # Example dimensions

    # Set the model to training mode
    model.train()

    # Create a dummy target (zeros)
    dummy_target = torch.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=torch.long).to(
        device
    )

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Perform a forward pass
    output = model(dummy_input, dummy_dimensions)

    # Calculate loss
    loss = criterion(output.view(-1, NUM_TOKENS + 1), dummy_target.view(-1))

    # Backward pass and optimization
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    print(f"Test completed. Loss: {loss.item()}")


def test_position_encodings():
    # Create dummy input data (zeros)
    dummy_input = torch.zeros((batch_size, MAX_SEQUENCE_LENGTH), dtype=torch.long).to(device)

    # Create dummy dimensions
    dummy_dimensions = (10, 10)  # Example dimensions

    # Perform a forward pass
    output = model(dummy_input, dummy_dimensions)

    # Check if the output has the correct shape
    assert output.shape == (
        batch_size,
        MAX_SEQUENCE_LENGTH,
        NUM_TOKENS + 1,
    ), f"Expected shape {(batch_size, MAX_SEQUENCE_LENGTH, NUM_TOKENS + 1)}, but got {output.shape}"

    # Check if the position encodings are not all zeros
    assert not torch.allclose(output[:, :, :-1], torch.zeros_like(output[:, :, :-1])), "Position encodings are all zeros"

    print("Position encodings test passed.")

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

if __name__ == "__main__":
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    test_model_with_zeros()
    test_position_encodings()
    
