import torch
import torch.nn as nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from torch.utils.checkpoint import checkpoint

from .data import MAX_CONTEXT_LENGTH, NUM_TOKENS, PAD_TOKEN, MAX_SEQUENCE_LENGTH
from .encoder import PositionEncoder, NUM_ENCODING_DIMENSIONS
from .args import dropout_rate, batch_size, use_schedulefree
from .data import is_special_token, SPECIAL_TOKENS

from schedulefree import AdamWScheduleFree

d_model = 128 - NUM_ENCODING_DIMENSIONS
nhead = 1
num_layers = 4
dim_feedforward = 256
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
        self.token_embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.position_encoder = PositionEncoder(5, 5, device=device)

        self.layers = nn.ModuleList(
            [
                self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        
        if use_schedulefree:
            self.optimizer = AdamWScheduleFree(self.parameters(), lr=0.001)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

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
        x = self.token_embedding(src)  # Embed all tokens

        # Generate a mask for non-special tokens
        non_special_mask = ~(src >= 10) & (src <= 18)
        x_to_process = x * non_special_mask.unsqueeze(-1).float()

        # Process tokens through the model layers
        for i, layer in enumerate(self.layers):
            q = k = v = self.position_encoder(x_to_process, dimensions)
            attn_output = layer['self_attn'](q, k, v)
            x_to_process = attn_output + x_to_process
            x_to_process = layer['norm1'](x_to_process)
            ff_output = layer['ff'](x_to_process)
            x_to_process = layer['norm2'](ff_output + x_to_process)

        # Final output calculations
        x = x_to_process + x * (~non_special_mask).unsqueeze(-1).float()
        output = self.fc_out(x)
        confidences = torch.softmax(output, dim=-1)

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