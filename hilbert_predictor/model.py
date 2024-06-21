import torch
import torch.nn as nn
import numpy as np

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, dim_feedforward, num_layers, num_context_tokens, num_pred_tokens, device):
        super().__init__()
        self.device = device
        self.num_context_tokens = num_context_tokens
        self.num_pred_tokens = num_pred_tokens
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=10)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True  # This should match your input dimensions

        )
        self.fc_out = nn.Linear(d_model, num_tokens + 1)  # output layer
        self.to(device)


    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size, device=self.device), diagonal=1).bool()
        return mask


    def forward(self, src, tgt=None):
        src_key_padding_mask = (src == 10)  # True where src is padding
        src_non_padding_mask = ~src_key_padding_mask

        context = self.embedding(src)
        context = context * src_non_padding_mask.unsqueeze(-1).float()

        if tgt is None:
            tgt = torch.zeros((src.size(0), self.num_pred_tokens), dtype=torch.long, device=self.device)
        
        tgt_key_padding_mask = (tgt == 10)  # True where tgt is padding
        tgt_non_padding_mask = ~tgt_key_padding_mask
        target = self.embedding(tgt)
        target = target * tgt_non_padding_mask.unsqueeze(-1).float()

        target_mask = self.generate_square_subsequent_mask(target.size(1))

        output = self.transformer(
            tgt=target,
            src=context,
            tgt_mask=target_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output)
        
        # Apply the non-padding mask to the output
        output = output * tgt_non_padding_mask.unsqueeze(-1).float()

        return output

num_tokens = 10

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_context_tokens = 1024
num_pred_tokens = 1024
checkpoint_path = "checkpoint.pt"
model = TransformerModel(num_tokens=num_tokens, d_model=512, nhead=8, dim_feedforward=2048, num_layers=6,
                         num_context_tokens=num_context_tokens, num_pred_tokens=num_pred_tokens, device=device)
