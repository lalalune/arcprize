import torch
import torch.nn as nn
from .args import quadtree

if quadtree:
    NUM_ENCODING_DIMENSIONS = 10
else:
    NUM_ENCODING_DIMENSIONS = 0


def compute_quadtree_code(x, y, max_width, max_height, level=0, code=""):
    if max_width == 1 and max_height == 1:  # dont encode a 1x1 grid
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

    return compute_quadtree_code(
        x % mid_x if mid_x > 0 else 0,
        y % mid_y if mid_y > 0 else 0,
        mid_x,
        mid_y,
        level + 1,
        code,
    )


def compute_quadtree_code(x, y, max_width, max_height, level=0, code=""):
    if max_width == 1 and max_height == 1:  # dont encode a 1x1 grid
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

    return compute_quadtree_code(
        x % mid_x if mid_x > 0 else 0,
        y % mid_y if mid_y > 0 else 0,
        mid_x,
        mid_y,
        level + 1,
        code,
    )


class PositionEncoder(nn.Module):
    def __init__(self, max_height, max_width, device):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.feature_dim = NUM_ENCODING_DIMENSIONS
        self.device = device

    def compute_encodings(self, height, width):
        if not quadtree:
            return torch.zeros(
                height, width, NUM_ENCODING_DIMENSIONS, device=self.device
            )
        encodings = torch.zeros(height, width, self.feature_dim)
        for y in range(height):
            for x in range(width):
                # X and Y linear fractions
                encodings[y, x, 0] = 2 * (x / (width - 1)) - 1 if width > 1 else 0
                encodings[y, x, 1] = 2 * (y / (height - 1)) - 1 if height > 1 else 0

                # Quadrant encoding
                encodings[y, x, 2] = (
                    -1 if x < width / 2 else (1 if x > width / 2 else 0)
                )
                encodings[y, x, 3] = (
                    -1 if y < height / 2 else (1 if y > height / 2 else 0)
                )

                # Trigonometric encodings
                # angle_x = math.pi * (x / (width - 1)) if width > 1 else 0
                # angle_y = math.pi * (y / (height - 1)) if height > 1 else 0
                # encodings[y, x, 4] = math.sin(angle_x)
                # encodings[y, x, 5] = math.sin(angle_y)
                # encodings[y, x, 6] = math.cos(angle_x)
                # encodings[y, x, 7] = math.cos(angle_y)
                # encodings[y, x, 8] = round(math.tan(angle_x)) if not math.isnan(math.tan(angle_x)) else 0
                # encodings[y, x, 9] = round(math.tan(angle_y)) if not math.isnan(math.tan(angle_y)) else 0

                # Quadtree encoding
                quadtree_code = compute_quadtree_code(
                    x, y, self.max_width, self.max_height
                )
                for i, digit in enumerate(quadtree_code):
                    encodings[y, x, i] = int(digit)

        return encodings

    def forward(self, x, dimensions):
        if x.dim() != 3:
            raise ValueError(f"Expected input to be 3D, but got {x.dim()}D")

        if not quadtree:
            return x

        batch_size, seq_len, _ = x.shape
        height, width = dimensions[0]

        if width == 0:
            repeated_encodings = torch.zeros(
                batch_size, seq_len, self.feature_dim, device=x.device
            )
        else:
            encodings = self.compute_encodings(height, width)
            flattened_encodings = encodings.view(-1, self.feature_dim)
            repeated_encodings = flattened_encodings.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            repeated_encodings = repeated_encodings[:, :seq_len, :]

        repeated_encodings = repeated_encodings.to(x.device)
        return torch.cat([x, repeated_encodings], dim=-1)
