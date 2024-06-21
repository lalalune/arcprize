import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math
from .data import NUM_TOKENS, PAD_TOKEN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_layers, dim_feedforward, max_seq_length, device):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, num_tokens + 1)
        self.to(device)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1).bool()
        return mask

    @staticmethod
    def create_pad_mask(seq, lengths):
        mask = torch.arange(seq.size(1))[None, :] >= lengths[:, None]
        return mask.to(seq.device)
    
    def forward(self, src, tgt=None, src_lengths=None, tgt_lengths=None):
        if src_lengths is not None:
            src_key_padding_mask = self.create_pad_mask(src, src_lengths)
        else:
            src_key_padding_mask = (src == PAD_TOKEN)

        src_non_padding_mask = ~src_key_padding_mask

        src_embedded = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_embedded = self.pos_encoder(src_embedded)
        src_embedded = src_embedded * src_non_padding_mask.unsqueeze(-1).float()

        if tgt is None:
            tgt = torch.zeros((src.size(0), self.max_seq_length), dtype=torch.long, device=self.device)
        
        if tgt_lengths is not None:
            tgt_key_padding_mask = self.create_pad_mask(tgt, tgt_lengths)
        else:
            tgt_key_padding_mask = (tgt == PAD_TOKEN)

        tgt_non_padding_mask = ~tgt_key_padding_mask
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        tgt_embedded = tgt_embedded * tgt_non_padding_mask.unsqueeze(-1).float()

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask
        )

        output = self.fc_out(output)
        output = output * tgt_non_padding_mask.unsqueeze(-1).float()

        return output

# Model initialization
d_model = 128
nhead = 4
num_layers = 6
dim_feedforward = 512
max_seq_length = 8192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = Path("checkpoint.pt")
model = TransformerModel(NUM_TOKENS, d_model, nhead, num_layers, dim_feedforward, max_seq_length, device)