import torch
import torch.nn as nn

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
        # Mask creation
        src_key_padding_mask = (src == 10)  # True where src is padding

        context = self.embedding(src)

        if tgt is None:
            tgt = torch.zeros((src.size(0), self.num_pred_tokens), dtype=torch.long, device=self.device)
        
        tgt_key_padding_mask = (tgt == 10)  # True where tgt is padding
        target = self.embedding(tgt)

        # Generate target mask for sequence-to-sequence models
        target_mask = self.generate_square_subsequent_mask(target.size(1))

        # Make sure that the masks are correctly sized and Boolean type
        output = self.transformer(
            tgt=target,
            src=context,
            tgt_mask=target_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc_out(output)
        return output

