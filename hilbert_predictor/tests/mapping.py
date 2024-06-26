import torch
import numpy as np
from ..encoder import NUM_ENCODING_DIMENSIONS
from ..args import use_hilbert
from ..mapping import create_mapping_table, load_mapping_table, flatten_2d_to_1d, save_mapping_table, unflatten_1d_to_2d

def test_mapping():
    mapping_table = load_mapping_table()
    for height in range(1, 31):
        for width in range(1, 31):
            mapping, encodings = mapping_table[(height, width)]
            print(f"Testing {height}x{width}: Mapping length={len(mapping)}, Encodings shape={encodings.shape}")
            
            # mapping[0] is a tuple
            coord = mapping[0]
            assert isinstance(coord, tuple), "First element of mapping should be a tuple"
            assert len(mapping) == height * width, f"Incorrect mapping length for {height}x{width}"
            assert encodings.shape == (height, width, NUM_ENCODING_DIMENSIONS), f"Incorrect encoding shape for {height}x{width} at {NUM_ENCODING_DIMENSIONS}: {encodings.shape}"
            
            if NUM_ENCODING_DIMENSIONS > 0:
                if not (torch.all(encodings[:, :, 0] >= -1) and torch.all(encodings[:, :, 0] <= 1)):
                    print(f"X values out of range for {height}x{width}")
                    print("Out of range X values:", encodings[:, :, 0][(encodings[:, :, 0] < -1) | (encodings[:, :, 0] > 1)])
                
                if not (torch.all(encodings[:, :, 1] >= -1) and torch.all(encodings[:, :, 1] <= 1)):
                    print(f"Y values out of range for {height}x{width}")
                    print("Out of range Y values:", encodings[:, :, 1][(encodings[:, :, 1] < -1) | (encodings[:, :, 1] > 1)])
                
                assert torch.all(encodings[:, :, 0] >= -1) and torch.all(encodings[:, :, 0] <= 1), "X values out of range"
                assert torch.all(encodings[:, :, 1] >= -1) and torch.all(encodings[:, :, 1] <= 1), "Y values out of range"
            else:
                print(f"Skipping range checks for {height}x{width} as NUM_ENCODING_DIMENSIONS is 0")

    print("All dimension and value tests passed.")


    def test_case(matrix):
        height, width = len(matrix), (
            len(matrix[0]) if isinstance(matrix[0], list) else 1
        )
        flattened = flatten_2d_to_1d(matrix)

        if height == 1 and width == 1:
            reconstructed = [[flattened[0]]]
        elif height == 1:
            reconstructed = [flattened]
        elif width == 1:
            reconstructed = [[value] for value in flattened]
        else:
            if use_hilbert:
                reconstructed = [[0] * width for _ in range(height)]
                for idx, value in enumerate(flattened):
                    x, y = mapping_table[(height, width)][0][
                        idx
                    ]  # Get the x, y coordinates from the tuple
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
    test_case([[i for i in range(j * 5 + 1, (j + 1) * 5 + 1)] for j in range(5)])  # 5x5


def test_flatten_and_unflatten():
    test_cases = [
        [[1]],  # 1x1
        [[1, 2, 3]],  # 1x3
        [[1], [2], [3]],  # 3x1
        [[1, 2], [3, 4]],  # 2x2
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3x3
        [[i for i in range(j * 5 + 1, (j + 1) * 5 + 1)] for j in range(5)]  # 5x5
    ]

    for matrix in test_cases:
        height, width = len(matrix), len(matrix[0]) if isinstance(matrix[0], list) else 1
        flattened = flatten_2d_to_1d(matrix)
        reconstructed = unflatten_1d_to_2d(flattened, width, height)
        assert matrix == reconstructed, f"Test failed for matrix: {matrix}"
        print(f"Test passed for {height}x{width} matrix")

def test_mapping_consistency():
    mapping_table = create_mapping_table()
    for height in range(1, 31):
        for width in range(1, 31):
            mapping, encodings = mapping_table[(height, width)]
            
            print(f"Testing {height}x{width}:")
            print(f"Mapping: {mapping}")
            
            # Check mapping length
            assert len(mapping) == height * width, f"Incorrect mapping length for {height}x{width}"
            
            # Check encoding shape
            assert encodings.shape == (height, width, NUM_ENCODING_DIMENSIONS), f"Incorrect encoding shape for {height}x{width}"
            
            # Check that all coordinates are unique
            assert len(set(mapping)) == len(mapping), f"Duplicate coordinates in mapping for {height}x{width}"
            
            # Check that all coordinates are within bounds
            in_bounds = all(0 <= x < width and 0 <= y < height for x, y in mapping)
            assert in_bounds, f"Out of bounds coordinates for {height}x{width}"

    print("Mapping consistency test passed.")


def test_encoding_range():
    mapping_table = create_mapping_table()
    for height in range(1, 31):
        for width in range(1, 31):
            _, encodings = mapping_table[(height, width)]
            
            if NUM_ENCODING_DIMENSIONS > 0:
                assert torch.all(encodings >= -1) and torch.all(encodings <= 1), f"Encoding values out of range [-1, 1] for {height}x{width}"

    print("Encoding range test passed.")

def test_load_and_save_mapping_table():
    original_table = create_mapping_table()
    save_mapping_table(original_table)
    loaded_table = load_mapping_table()
    
    assert set(original_table.keys()) == set(loaded_table.keys()), "Loaded table keys don't match original"
    
    for key in original_table:
        assert np.array_equal(original_table[key][0], loaded_table[key][0]), f"Mapping mismatch for {key}"
        assert torch.all(torch.eq(original_table[key][1], loaded_table[key][1])), f"Encoding mismatch for {key}"
    
    print("Load and save mapping table test passed.")

def run_all_tests():
    test_flatten_and_unflatten()
    test_mapping_consistency()
    test_encoding_range()
    test_load_and_save_mapping_table()
    test_mapping()
    print("All tests passed successfully!")

if __name__ == "__main__":
    run_all_tests()
