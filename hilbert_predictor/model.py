import torch
import torch.nn as nn
import math
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import AttentionConfig, ScaledDotProduct
from .data import NUM_TOKENS, PAD_TOKEN
from xformers.components.positional_embedding import RotaryEmbedding
from torch.utils.checkpoint import checkpoint

# Model initialization
batch_size = 1
if torch.cuda.is_available():
    batch_size = 16
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

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        # Token embedding layer
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=PAD_TOKEN)

        # Rotary positional embedding
        self.rotary_emb = RotaryEmbedding(d_model)
        
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


    import torch
import torch.nn as nn
import math
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import AttentionConfig, ScaledDotProduct
from .data import NUM_TOKENS, PAD_TOKEN
from xformers.components.positional_embedding import RotaryEmbedding

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

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=PAD_TOKEN)
        self.rotary_emb = RotaryEmbedding(d_model)
        
        self.layers = nn.ModuleList([
            self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, num_tokens + 1)
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
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

    def forward(self, src):
        print(f"Input shape: {src.shape}")
        assert src.dim() == 2, f"Expected input to be 2D, but got {src.dim()}D"
        
        x = self.token_embedding(src)
        print(f"After token embedding: {x.shape}")
        assert x.shape == (src.shape[0], src.shape[1], self.d_model), f"Expected shape {(src.shape[0], src.shape[1], self.d_model)}, but got {x.shape}"
        
        batch_size, seq_len, d_model = x.shape
        
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}")
            q, k, v = x, x, x
            
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            print(f"Before rotary embedding: q.shape = {q.shape}, k.shape = {k.shape}")
            assert q.shape == k.shape == (seq_len, batch_size, d_model), f"Expected shape {(seq_len, batch_size, d_model)}, but got q: {q.shape}, k: {k.shape}"
            
            q, k = self.rotary_emb(q, k)
            print(f"After rotary embedding: q.shape = {q.shape}, k.shape = {k.shape}")
            assert q.shape == k.shape == (1, seq_len, batch_size, d_model), f"Expected shape {(1, seq_len, batch_size, d_model)}, but got q: {q.shape}, k: {k.shape}"
            
            q = q.squeeze(0).permute(1, 0, 2)
            k = k.squeeze(0).permute(1, 0, 2)
            print(f"After permute back: q.shape = {q.shape}, k.shape = {k.shape}")
            assert q.shape == k.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, but got q: {q.shape}, k: {k.shape}"

            q = q.squeeze().view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            k = k.squeeze().view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            v = v.squeeze().view(batch_size, seq_len, self.nhead, -1).permute(1, 0, 2, 3)
            print(f"Before self-attention: q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")
            
            assert q.shape == k.shape == v.shape == (seq_len, batch_size, self.nhead, d_model // self.nhead), \
                f"Expected shape {(seq_len, batch_size, self.nhead, d_model // self.nhead)}, but got q: {q.shape}, k: {k.shape}, v: {v.shape}"

            q = q.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            k = k.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            v = v.permute(1, 0, 2, 3).contiguous().view(batch_size, seq_len, -1)
            
            print(f"After permutation: q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")

            attn_output = checkpoint(layer['self_attn'], q, k, v)

            print(f"After self-attention: attn_output.shape = {attn_output.shape}")
            assert attn_output.shape == (batch_size, seq_len, d_model), \
                f"Expected shape {(batch_size, seq_len, d_model)}, but got {attn_output.shape}"
            
            x = layer['norm1'](x + attn_output)
            print(f"After first norm: x.shape = {x.shape}")
            assert x.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, but got {x.shape}"
            
            ff_output = checkpoint(layer['ff'], x)

            print(f"After feedforward: ff_output.shape = {ff_output.shape}")
            assert ff_output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, but got {ff_output.shape}"
            
            x = layer['norm2'](x + ff_output)
            print(f"After second norm: x.shape = {x.shape}")
            assert x.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, but got {x.shape}"
        
        output = self.fc_out(x)
        print(f"Final output shape: {output.shape}")
        assert output.shape == (batch_size, seq_len, NUM_TOKENS + 1), f"Expected shape {(batch_size, seq_len, NUM_TOKENS + 1)}, but got {output.shape}"
        
        return output

model = DecoderOnlyTransformer(NUM_TOKENS, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device)