import torch
import torch.nn as nn
import math
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import AttentionConfig, ScaledDotProduct
from .data import NUM_TOKENS, PAD_TOKEN

# Model initialization
batch_size = 1
if torch.cuda.is_available():
    batch_size = 32
d_model = 64 # 512
nhead = 4
num_layers = 6
dim_feedforward = 512 # 2048
max_seq_length = 4096
max_context_length = 3072
max_prediction_length = 1024
dropout_rate = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = Path("checkpoint.pt")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

from xformers.components.attention import ScaledDotProduct

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, num_tokens + 1)
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
        # Create the attention mechanism with causality directly if possible
        attention = ScaledDotProduct(dropout=dropout_rate, causal=True)
        
        return nn.ModuleDict({
            'self_attn': MultiHeadDispatch(
                dim_model=d_model,
                num_heads=nhead,
                attention=attention,
                
                bias=True,
                residual_dropout=dropout_rate,
            ),
            'ff': nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout_rate),
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
        })


    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer['norm1'](x)
            # reshape the attention mask to match the shape
            x = layer['self_attn'](query=x, key=x, value=x)
            x = layer['norm1'](x)  # Applying post-attention layer norm
            x = layer['ff'](x)
            x = layer['norm2'](x)
        
        x = self.fc_out(x)  # Output layer that transforms features into logits
        return x

model = DecoderOnlyTransformer(NUM_TOKENS, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device)