import os
import pickle
import numpy as np
import torch
from .encoder import NUM_ENCODING_DIMENSIONS, PositionEncoder
from .args import hilbert, quadtree

# Gilbert2D - Generalized Hilbert Curve for 2D space-filling
# From https://github.com/jakubcerveny/gilbert
def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield (x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield (x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def flatten_2d_to_1d(array):
    array = np.array(array)
    if array.ndim == 1:
        return array.tolist()
    elif array.ndim == 2:
        height, width = array.shape
        if height == 1:
            return array[0].tolist()
        elif width == 1:
            return array.T[0].tolist()
        else:
            # 2D matrix: flatten using Hilbert curve or raw concatenation
            if hilbert:
                array_1d = [None] * (width * height)
                for idx, (x, y) in enumerate(gilbert2d(width, height)):
                    array_1d[idx] = array[y][x]
            else:
                array_1d = array.flatten().tolist()
            return array_1d
    else:
        raise ValueError(f"Input array must be 1D or 2D, got {array.ndim}D")

def unflatten_1d_to_2d(array_1d, width, height):
    array_2d = [[None] * width for _ in range(height)]
    if hilbert:
        for idx, (x, y) in enumerate(gilbert2d(width, height)):
            array_2d[y][x] = array_1d[idx]
    else:
        for i in range(height):
            for j in range(width):
                array_2d[i][j] = array_1d[i * width + j]
    return array_2d

def create_mapping_table():
    mapping_table = {}
    position_encoder = PositionEncoder(30, 30, device='cpu')
    for height in range(1, 31):
        for width in range(1, 31):
            key = (height, width)
            if height == 1 or width == 1:
                mapping = [(0, i) if height == 1 else (i, 0) for i in range(max(height, width))]
            else:
                mapping = list(gilbert2d(width, height))
            encodings = position_encoder.compute_encodings(height, width)
            mapping_table[key] = (mapping, encodings)
    return mapping_table

def save_mapping_table(mapping_table, filename='mapping_table.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(mapping_table, f)

def load_mapping_table(filename='mapping_table.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def test_mapping():
    mapping_table = load_mapping_table()
    for height in range(1, 31):
        for width in range(1, 31):
            mapping, encodings = mapping_table[(height, width)]
            # mapping[0] is a tuple
            coord = mapping[0]
            assert isinstance(coord, tuple), "First element of mapping should be a tuple"
            assert len(mapping) == height * width, f"Incorrect mapping length for {height}x{width}"
            assert encodings.shape == (height, width, NUM_ENCODING_DIMENSIONS), f"Incorrect encoding shape for {height}x{width} at {NUM_ENCODING_DIMENSIONS}: {encodings.shape}"
            assert torch.all(encodings[:, :, 0] >= -1) and torch.all(encodings[:, :, 0] <= 1), "X values out of range"
            assert torch.all(encodings[:, :, 1] >= -1) and torch.all(encodings[:, :, 1] <= 1), "Y values out of range"

    print("All dimension and value tests passed.")


    def test_case(matrix):
        height, width = len(matrix), len(matrix[0]) if isinstance(matrix[0], list) else 1
        flattened = flatten_2d_to_1d(matrix)

        if height == 1 and width == 1:
            reconstructed = [[flattened[0]]]
        elif height == 1:
            reconstructed = [flattened]
        elif width == 1:
            reconstructed = [[value] for value in flattened]
        else:
            if hilbert:
                reconstructed = [[0] * width for _ in range(height)]
                for idx, value in enumerate(flattened):
                    x, y = mapping_table[(height, width)][0][idx]  # Get the x, y coordinates from the tuple
                    reconstructed[y][x] = value
            else:
                reconstructed = unflatten_1d_to_2d(flattened, width, height)

        assert matrix == reconstructed, f"Test failed for matrix: {matrix}"
        print(f"Test passed for {height}x{width} matrix")

    # Test cases
    test_case([[1]])  # 1x1
    test_case([[1, 2, 3]])  # 1x3
    test_case([[1], [2], [3]])  # 3x1
    test_case([[1, 2], [3, 4]])  # 2x2
    test_case([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3
    test_case([[i for i in range(j*5+1, (j+1)*5+1)] for j in range(5)])  # 5x5

mapping_table_file = 'hilbert_mapping_quadtree.pkl'
if quadtree is False:
    mapping_table_file = 'mapping_table.pkl'
    
if os.path.exists(mapping_table_file):
    print("Loading existing mapping table...")
    mapping_table = load_mapping_table(mapping_table_file)
else:
    print("Creating new mapping table...")
    mapping_table = create_mapping_table()
    save_mapping_table(mapping_table, mapping_table_file)
    
if __name__ == "__main__":
    test_mapping()
    print("All tests passed.")
