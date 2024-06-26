import torch
import torch.nn as nn
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct

from .data import NUM_TOKENS, PAD_TOKEN, MAX_SEQUENCE_LENGTH
from .encoder import PositionEncoder, NUM_ENCODING_DIMENSIONS
from .args import dropout_rate, use_schedulefree

from schedulefree import AdamWScheduleFree

d_model = 128 - NUM_ENCODING_DIMENSIONS
nhead = 1
num_layers = 4
dim_feedforward = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transformer(nn.Module):
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
        self.d_model_with_pos = d_model + NUM_ENCODING_DIMENSIONS
        self.nhead = nhead
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.token_embedding = nn.Embedding(
            num_tokens + 1, d_model, padding_idx=PAD_TOKEN
        )
        self.position_encoder = PositionEncoder(5, 5, device=device)
        self.layers = nn.ModuleList(
            [
                self.create_decoder_layer(
                    self.d_model_with_pos, nhead, dim_feedforward, dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

        if use_schedulefree:
            self.optimizer = AdamWScheduleFree(self.parameters(), lr=0.001)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        self.fc_out = nn.Linear(self.d_model_with_pos, num_tokens + 1)
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
        attention = ScaledDotProduct(dropout=dropout_rate, causal=True)
        return nn.ModuleDict(
            {
                "self_attn": MultiHeadDispatch(
                    dim_model=self.d_model_with_pos,
                    num_heads=nhead,
                    attention=attention,
                    bias=True,
                    residual_dropout=dropout_rate,
                ),
                "ff": nn.Sequential(
                    nn.Linear(self.d_model_with_pos, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dim_feedforward, self.d_model_with_pos),
                    nn.Dropout(dropout_rate),
                ),
                "norm1": nn.LayerNorm(self.d_model_with_pos),
                "norm2": nn.LayerNorm(self.d_model_with_pos),
            }
        )

    def forward(self, src, dimensions):
        x = self.token_embedding(src)  # Embed all tokens

        # Generate a mask for non-special tokens
        non_special_mask = ~(src >= 10) & (src <= 18)
        x_to_process = x * non_special_mask.unsqueeze(-1).float()

        # Add position encodings
        x_to_process = self.position_encoder(x_to_process, dimensions)

        # Process tokens through the model layers
        for i, layer in enumerate(self.layers):
            q = k = v = x_to_process
            attn_output = layer["self_attn"](q, k, v)
            x_to_process = attn_output + x_to_process
            x_to_process = layer["norm1"](x_to_process)
            ff_output = layer["ff"](x_to_process)
            x_to_process = layer["norm2"](ff_output + x_to_process)

        # Final output calculations
        # Pad x to match x_to_process dimensions
        x_padded = torch.cat(
            [x, torch.zeros(*x.shape[:-1], NUM_ENCODING_DIMENSIONS, device=x.device)],
            dim=-1,
        )
        x = x_to_process + x_padded * (~non_special_mask).unsqueeze(-1).float()
        output = self.fc_out(x)
        confidences = torch.softmax(output, dim=-1)

        return output, confidences


model = Transformer(
    NUM_TOKENS,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    MAX_SEQUENCE_LENGTH,
    dropout_rate,
    device,
)
