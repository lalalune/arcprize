import torch
import torch.nn as nn
import math
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import AttentionConfig, ScaledDotProduct
from xformers.components.positional_embedding import RotaryEmbedding
from .data import NUM_TOKENS, PAD_TOKEN

class Transformer(nn.Module):
    """
    A Transformer model implementation using xformers for efficient attention mechanisms.
    
    This class implements a decoder-only Transformer, suitable for language modeling tasks.
    It uses rotary positional embeddings (RoPE) for enhanced position-aware representations.
    
    Parameters
    ----------
    num_tokens : int
        The size of the vocabulary.
    d_model : int
        The dimensionality of the model's hidden states and embeddings.
    nhead : int
        The number of attention heads in the multi-head attention mechanisms.
    num_layers : int
        The number of transformer layers in the model.
    dim_feedforward : int
        The dimensionality of the feedforward network in each transformer layer.
    max_seq_length : int
        The maximum sequence length the model can handle.
    dropout_rate : float
        The dropout rate used throughout the model for regularization.
    device : torch.device
        The device (CPU or GPU) on which the model will be allocated.
    """

    def __init__(self, num_tokens, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=PAD_TOKEN)
        
        # Rotary positional embedding
        self.rotary_emb = RotaryEmbedding(d_model)

        # Create the stack of transformer layers
        self.layers = nn.ModuleList([
            self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ])

        # Output linear layer to project hidden states to vocabulary size
        self.fc_out = nn.Linear(d_model, num_tokens)
        
        # Move the model to the specified device
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
        """
        Create a single decoder layer for the Transformer.
        
        This method constructs a dictionary containing the components of a decoder layer:
        1. Self-attention mechanism
        2. Feedforward neural network
        3. Layer normalization modules
        
        The self-attention uses a causal mask to prevent attending to future tokens.
        
        Parameters
        ----------
        d_model : int
            The dimensionality of the model's hidden states.
        nhead : int
            The number of attention heads.
        dim_feedforward : int
            The dimensionality of the feedforward network.
        dropout_rate : float
            The dropout rate for regularization.
        
        Returns
        -------
        nn.ModuleDict
            A dictionary containing the components of the decoder layer.
        """
        # Create the attention mechanism with causal masking
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

    def forward(self, x, attention_mask=None):
        """
        Forward pass of the Transformer model.
        
        This method processes the input through the embedding layer, 
        applies rotary positional embeddings, and then passes the result 
        through each transformer layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of token indices, shape (batch_size, seq_len).
        attention_mask : torch.Tensor, optional
            Mask to avoid attending to padding tokens, shape (seq_len, seq_len).
        
        Returns
        -------
        torch.Tensor
            Output logits for each token in the vocabulary, shape (batch_size, seq_len, num_tokens).
        """
        # Apply token embeddings
        x = self.token_embedding(x)
        
        # Adjust dimensions if necessary (this might be needed depending on the input shape)
        if x.dim() == 4:
            batch_size, seq_len, num_heads, head_dim = x.shape
            x = x.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
        
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            
            # Prepare query, key, and value for attention
            q, k, v = x, x, x
            
            # Apply Rotary Positional Embedding (RoPE)
            q, k = self.rotary_emb(q.transpose(0, 1), k.transpose(0, 1))
            q, k = q.transpose(0, 1), k.transpose(0, 1)
            
            # Perform self-attention
            x = layer['self_attn'](query=q, key=k, value=v, att_mask=attention_mask)
            
            # Add residual connection
            x = residual + x
            
            # Feed-forward network with residual connection
            residual = x
            x = layer['norm2'](x)
            x = layer['ff'](x)
            x = residual + x
        
        # Reshape back if necessary
        batch_size, seq_len, _ = x.shape

        # Project to vocabulary size
        x = self.fc_out(x)
        return x

    def generate(self, input_ids, max_length, temperature=1.0):
        """
        Generate a sequence of tokens autoregressively.
        
        This method takes an initial sequence of token IDs and generates
        additional tokens up to the specified maximum length.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Initial sequence of token IDs, shape (batch_size, initial_seq_len).
        max_length : int
            The maximum length of the generated sequence.
        temperature : float, optional
            Temperature for sampling. Higher values increase randomness.
        
        Returns
        -------
        torch.Tensor
            Generated sequence of token IDs, shape (batch_size, max_length).
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Create causal attention mask
                attention_mask = torch.triu(torch.ones(input_ids.size(1), input_ids.size(1)), diagonal=1).bool().to(self.device)
                
                # Get model outputs
                outputs = self(input_ids, attention_mask=attention_mask)
                
                # Apply temperature to logits
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Sample next token
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                
                # Append new token to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

# Model initialization
d_model = 512  # Dimensionality of the model
nhead = 4  # Number of attention heads
num_layers = 6  # Number of transformer layers
dim_feedforward = 2048  # Dimensionality of the feedforward network
max_seq_length = 4096  # Maximum sequence length
max_context_length = 3072  # Maximum context length (not used in the provided code)
max_prediction_length = 1024  # Maximum prediction length (not used in the provided code)
dropout_rate = 0.05  # Dropout rate for regularization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device selection

checkpoint_path = Path("checkpoint.pt")  # Path to save/load model checkpoints
model = Transformer(NUM_TOKENS, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout_rate, device)