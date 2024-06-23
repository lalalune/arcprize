import torch
import torch.nn as nn
import math

NUM_ENCODING_DIMENSIONS = 68

class PositionEncoder(nn.Module):
    def __init__(self, max_height, max_width):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.feature_dim = NUM_ENCODING_DIMENSIONS

    def compute_encodings(self, height, width):
        encodings = torch.zeros(height, width, self.feature_dim)
        for y in range(height):
            for x in range(width):
                # X and Y linear fractions
                encodings[y, x, 0] = 2 * (x / (width - 1)) - 1 if width > 1 else 0
                encodings[y, x, 1] = 2 * (y / (height - 1)) - 1 if height > 1 else 0

                # Quadrant encoding
                encodings[y, x, 2] = -1 if x < width / 2 else (1 if x > width / 2 else 0)
                encodings[y, x, 3] = -1 if y < height / 2 else (1 if y > height / 2 else 0)

                # Trigonometric encodings
                angle_x = math.pi * (x / (width - 1)) if width > 1 else 0
                angle_y = math.pi * (y / (height - 1)) if height > 1 else 0
                encodings[y, x, 4] = math.sin(angle_x)
                encodings[y, x, 5] = math.sin(angle_y)
                encodings[y, x, 6] = math.cos(angle_x)
                encodings[y, x, 7] = math.cos(angle_y)
                encodings[y, x, 8] = round(math.tan(angle_x)) if not math.isnan(math.tan(angle_x)) else 0
                encodings[y, x, 9] = round(math.tan(angle_y)) if not math.isnan(math.tan(angle_y)) else 0

                # One-hot encoding (adjusted to fit within 68 dimensions)
                if x < 29 and y < 29:
                    encodings[y, x, 10 + x] = 1
                    encodings[y, x, 39 + y] = 1

        return encodings

    def forward(self, x, dimensions):
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
        else:
            raise ValueError(f"Expected input to be 3D, but got {x.dim()}D")

        height, width = dimensions
        encodings = self.compute_encodings(height, width)
        flattened_encodings = encodings.view(-1, self.feature_dim)
        
        # Repeat the encodings to match the batch size and sequence length
        repeated_encodings = flattened_encodings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # If the sequence length is longer than the flattened encodings, repeat to match
        if seq_len > repeated_encodings.size(1):
            repeated_encodings = repeated_encodings.repeat(1, math.ceil(seq_len / repeated_encodings.size(1)), 1)
        
        # Truncate to match the sequence length
        repeated_encodings = repeated_encodings[:, :seq_len, :]
        
        # Combine the original input with the position encodings
        return torch.cat([x, repeated_encodings], dim=-1)


def test_position_encoder():
    encoder = PositionEncoder(30, 30)
    
    # Test 1x1
    x = torch.zeros(1, 1, 1)  # Add an extra dimension for the embedding
    dimensions = (1, 1)
    output = encoder(x, dimensions)
    assert output.shape == (1, 1, 69), f"Incorrect output shape for 1x1: {output.shape}"

    # Test 5x5
    x = torch.zeros(1, 25, 1)  # Add an extra dimension for the embedding
    dimensions = (5, 5)
    output = encoder(x, dimensions)
    assert output.shape == (1, 25, 69), f"Incorrect output shape for 5x5: {output.shape}"

    # Test 30x30
    x = torch.zeros(1, 900, 1)  # Add an extra dimension for the embedding
    dimensions = (30, 30)
    output = encoder(x, dimensions)
    assert output.shape == (1, 900, 69), f"Incorrect output shape for 30x30: {output.shape}"

    print("All PositionEncoder tests passed.")

if __name__ == "__main__":
    test_position_encoder()
    