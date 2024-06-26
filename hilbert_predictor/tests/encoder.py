import torch
import torch.nn as nn
from ..args import quadtree
from ..encoder import PositionEncoder


def test_position_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PositionEncoder(32, 32, device)
    x = torch.zeros(1, 1, 1, device=device)  # Add an extra dimension for the embedding
    dimensions = [[1, 1]]
    output = encoder(x, dimensions)

    # Determine expected output dimensions based on `quadtree` setting
    expected_dim = 1 + (
        10 if quadtree else 0
    )  # adding 10 if quadtree is true, else just 1
    assert output.shape == (
        1,
        1,
        expected_dim,
    ), f"Incorrect output shape for 1x1: {output.shape}"

    if quadtree:
        assert torch.all(
            output[0, 0, 1:]
            == torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device)
        ), "Incorrect quadtree encoding for 1x1"
    else:
        # Verify that the output is same as the input if quadtree is false
        assert torch.all(
            output == x
        ), "Output should be identical to input when quadtree is False"

    print("All PositionEncoder tests passed.")


def test_quadtree_encodings():
    # Testing top-left and bottom-right coordinates for a 32x32 grid
    top_left_expected = "00000"
    bottom_right_expected = "33333"

    # Calculating actual encodings
    top_left_actual = compute_quadtree_code(0, 0, 32, 32)
    bottom_right_actual = compute_quadtree_code(31, 31, 32, 32)

    # Assertions to check if the actual encodings match the expected ones
    assert (
        top_left_actual == top_left_expected
    ), f"Top-left encoding mismatch: expected {top_left_expected}, got {top_left_actual}"
    assert (
        bottom_right_actual == bottom_right_expected
    ), f"Bottom-right encoding mismatch: expected {bottom_right_expected}, got {bottom_right_actual}"

    print("Quadtree encoding tests passed.")


if __name__ == "__main__":
    test_quadtree_encodings()
    test_position_encoder()
