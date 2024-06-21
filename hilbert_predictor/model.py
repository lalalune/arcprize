import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math
from .data import NUM_TOKENS, PAD_TOKEN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


max_context_length = 4096
max_prediction_length = 1024


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_context_length,
        max_prediction_length,
        dropout_rate,
        device,
    ):
        super().__init__()
        self.device = device
        self.max_context_length = max_context_length
        self.max_prediction_length = max_prediction_length
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(d_model, max_context_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, num_tokens + 1)
        self.to(device)

    def forward(self, src, tgt, src_lengths, tgt_lengths):
        src_mask = None  # Let the model attend to all source tokens
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)

        src_key_padding_mask = self.create_pad_mask(src, src_lengths)
        tgt_key_padding_mask = self.create_pad_mask(tgt, tgt_lengths)

        src_embedded = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)

        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_embedded = self.pos_encoder(tgt_embedded)

        output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        output = self.fc_out(output)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    @staticmethod
    def create_pad_mask(seq, lengths):
        batch_size, max_len = seq.size()
        mask = torch.arange(max_len, device=seq.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask


# Model initialization
d_model = 128
nhead = 4
num_layers = 6
dim_feedforward = 1024
max_seq_length = 4096
dropout_rate = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = Path("checkpoint.pt")
model = TransformerModel(
    NUM_TOKENS,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_context_length,
    max_prediction_length,
    dropout_rate,
    device,
)
