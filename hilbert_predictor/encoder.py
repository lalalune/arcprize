import torch
import torch.nn as nn
import math

NUM_ENCODING_DIMENSIONS = 18

def compute_quadtree_code(x, y, max_width, max_height, level=0, code=""):
    if max_width == 1 and max_height == 1: # dont encode a 1x1 grid
        return code

    if level == 5:  # Limit the recursion depth to 5 levels (32x32 grid)
        return code
    
    mid_x = max_width // 2
    mid_y = max_height // 2
    
    if x < mid_x and y < mid_y:
        code += "0"
    elif x >= mid_x and y < mid_y:
        code += "1"
    elif x < mid_x and y >= mid_y:
        code += "2"
    else:
        code += "3"
    
    return compute_quadtree_code(x % mid_x if mid_x > 0 else 0,
                                 y % mid_y if mid_y > 0 else 0,
                                 mid_x, mid_y, level + 1, code)

class PositionEncoder(nn.Module):
    def __init__(self, max_height, max_width, device):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.feature_dim = NUM_ENCODING_DIMENSIONS
        self.device = device

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

                # Quadtree encoding
                quadtree_code = compute_quadtree_code(x, y, self.max_width, self.max_height)
                for i, digit in enumerate(quadtree_code):
                    encodings[y, x, 10 + i] = int(digit)

        return encodings


    def forward(self, x, dimensions):
        if x.dim() != 3:
            raise ValueError(f"Expected input to be 3D, but got {x.dim()}D")
        # print(f"PositionEncoder - input shape: {x.shape}")
        # print(f"PositionEncoder - dimensions: {dimensions}")
        batch_size, seq_len, _ = x.shape
        height, width = dimensions[0]
        # print(f"height: {height}, width: {width}")

        if width == 0:
            # Handle the case when width is zero
            repeated_encodings = torch.zeros(batch_size, seq_len, self.feature_dim, device=x.device)
        else:
            encodings = self.compute_encodings(height, width)
            # print(f"encodings.shape: {encodings.shape}")
            flattened_encodings = encodings.view(-1, self.feature_dim)
            # print(f"flattened_encodings.shape: {flattened_encodings.shape}")
            
            # Repeat the encodings to match the batch size and sequence length
            repeated_encodings = flattened_encodings.unsqueeze(0).repeat(batch_size, 1, 1)
            repeated_encodings = repeated_encodings[:, :seq_len, :]
            
            # print(f"repeated_encodings.shape: {repeated_encodings.shape}")
        
        # Move the repeated_encodings to the same device as the input tensor x
        repeated_encodings = repeated_encodings.to(x.device)
        
        # Combine the original input with the position encodings
        return torch.cat([x, repeated_encodings], dim=-1)

    
def test_position_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PositionEncoder(32, 32, device)
    x = torch.zeros(1, 1, 1, device=device)  # Add an extra dimension for the embedding
    dimensions = (1, 1)
    output = encoder(x, dimensions)
    assert output.shape == (1, 1, 19), f"Incorrect output shape for 1x1: {output.shape}"
    assert torch.all(output[0, 0, 10:] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)), "Incorrect quadtree encoding for 1x1"
    
    # Add further tests as needed
    
    print("All PositionEncoder tests passed.")

if __name__ == "__main__":
    test_position_encoder()

def test_position_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PositionEncoder(32, 32, device)
    
    test_configs = [
        (1, 1, [0]),
        (1, 2, [0, 1]),
        (2, 1, [0, 1]),
        (3, 2, [0, 1, 3, 4]),
        (2, 3, [0, 1, 2, 3, 4, 5]),
        (3, 3, [0, 2, 3, 5, 6, 8])
    ]

    for height, width, positions in test_configs:
        dimensions = (height, width)
        x = torch.zeros(1, height * width, 1, device=device)  # Add an extra dimension for the embedding
        output = encoder(x, dimensions)
        assert output.shape == (1, height * width, 19), f"Incorrect output shape for {height}x{width}: {output.shape}"

        # Test for top-left and bottom-right positions
        for pos in positions:
            # Check if the quadtree encoding length matches the expected, typically the same as the recursion depth (5)
            assert len(output[0, pos, 10:].nonzero()) <= 5, f"Incorrect quadtree encoding length for position {pos} in {height}x{width}"

    print("All PositionEncoder tests passed.")

def test_quadtree_encodings():
    # Testing top-left and bottom-right coordinates for a 32x32 grid
    top_left_expected = '00000'
    bottom_right_expected = '33333'
    
    # Calculating actual encodings
    top_left_actual = compute_quadtree_code(0, 0, 32, 32)
    bottom_right_actual = compute_quadtree_code(31, 31, 32, 32)
    
    # Assertions to check if the actual encodings match the expected ones
    assert top_left_actual == top_left_expected, f"Top-left encoding mismatch: expected {top_left_expected}, got {top_left_actual}"
    assert bottom_right_actual == bottom_right_expected, f"Bottom-right encoding mismatch: expected {bottom_right_expected}, got {bottom_right_actual}"

    print("Quadtree encoding tests passed.")

if __name__ == "__main__":
    test_quadtree_encodings()
    test_position_encoder()
