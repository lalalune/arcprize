import torch
import torch.nn as nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from torch.utils.checkpoint import checkpoint

from .data import MAX_CONTEXT_LENGTH, NUM_TOKENS, PAD_TOKEN, MAX_SEQUENCE_LENGTH
from .encoder import PositionEncoder, NUM_ENCODING_DIMENSIONS
from .args import dropout_rate, batch_size

from schedulefree import AdamWScheduleFree

d_model = 128 - NUM_ENCODING_DIMENSIONS
nhead = 2
num_layers = 6
dim_feedforward = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.position_encoder = PositionEncoder(5, 5, device=device)

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
        # print(f"DecoderOnlyTransformer - input shape: {src.shape}")
        # print(f"DecoderOnlyTransformer - dimensions: {dimensions}")
        # print(f"DecoderOnlyTransformer - height: {height}, width: {width}")
        
        # Clamp the input values to be within the valid range
        src = torch.clamp(src, 0, self.token_embedding.num_embeddings - 1)
        
        # Reshape src if it's 4D
        if src.dim() == 4:
            src = src.squeeze(2)
        elif src.dim() == 3:
            src = src.squeeze(1)
        
        x = self.token_embedding(src)
        
        # print(f"DecoderOnlyTransformer - x shape after token embedding: {x.shape}")
        # print(f"DecoderOnlyTransformer - x values after token embedding: {x}")
        
        # Adjust the assertion to allow for flexible batch size and sequence length
        assert x.shape[-1] == self.d_model, f"Expected last dimension to be {self.d_model}, but got {x.shape[-1]}"

        # Add position encodings
        x = self.position_encoder(x, dimensions)

        # print(f"DecoderOnlyTransformer - x shape after position encoding: {x.shape}")

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

        confidences = torch.argmax(output, dim=-1)
        return output, confidences


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
    dummy_input = torch.full((batch_size, MAX_SEQUENCE_LENGTH), PAD_TOKEN, dtype=torch.long).to(device)
    dummy_input[:, :MAX_CONTEXT_LENGTH] = torch.randint(0, NUM_TOKENS, (batch_size, MAX_CONTEXT_LENGTH), device=device)

    # Create dummy dimensions
    dummy_dimensions = [[MAX_CONTEXT_LENGTH // 5, MAX_CONTEXT_LENGTH // 5]]  # Example dimensions

    # Set the model to training mode
    model.train()

    # Create a dummy target (zeros)
    dummy_target = torch.full((batch_size, MAX_SEQUENCE_LENGTH), PAD_TOKEN, dtype=torch.long).to(
        device
    )
    dummy_target[:, :MAX_CONTEXT_LENGTH] = torch.randint(0, NUM_TOKENS, (batch_size, MAX_CONTEXT_LENGTH), device=device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Perform a forward pass
    output, _ = model(dummy_input, dummy_dimensions)

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
    dummy_input[:, :MAX_CONTEXT_LENGTH] = torch.randint(0, NUM_TOKENS, (batch_size, MAX_CONTEXT_LENGTH), device=device)

    # Create dummy dimensions
    dummy_dimensions = [[MAX_CONTEXT_LENGTH // 5, MAX_CONTEXT_LENGTH // 5]]  # Example dimensions

    # Perform a forward pass
    output, _ = model(dummy_input, dummy_dimensions)

    # Check if the output has the correct shape
    assert output.shape == (
        batch_size,
        MAX_SEQUENCE_LENGTH,
        NUM_TOKENS + 1,
    ), f"Expected shape {(batch_size, MAX_SEQUENCE_LENGTH, NUM_TOKENS + 1)}, but got {output.shape}"

    # Check if the position encodings are not all zeros
    assert not torch.allclose(output[:, :MAX_CONTEXT_LENGTH, :-1], torch.zeros_like(output[:, :MAX_CONTEXT_LENGTH, :-1])), "Position encodings are all zeros"

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