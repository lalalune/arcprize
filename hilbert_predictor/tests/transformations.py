import numpy as np
from ..transformations import identity, flip_horizontal, flip_vertical, rotate_180

def run_tests():
    matrix_1 = np.array([[1, 2], [3, 4]])
    matrix_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Test identity function
    print("Testing identity function:")
    assert np.array_equal(
        identity(matrix_1), matrix_1
    ), "Identity failed for 2x2 matrix"
    assert np.array_equal(
        identity(matrix_2), matrix_2
    ), "Identity failed for 3x3 matrix"
    assert identity.is_identity, "Identity function should have is_identity=True"
    print("Identity function tests passed.")

    # Test flip_horizontal function
    print("\nTesting flip_horizontal function:")
    assert np.array_equal(
        flip_horizontal(matrix_1), np.array([[2, 1], [4, 3]])
    ), "Flip horizontal failed for 2x2 matrix"
    assert np.array_equal(
        flip_horizontal(matrix_2), np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
    ), "Flip horizontal failed for 3x3 matrix"
    assert (
        not flip_horizontal.is_identity
    ), "Flip horizontal function should have is_identity=False"
    print("Flip horizontal function tests passed.")

    # Test flip_vertical function
    print("\nTesting flip_vertical function:")
    assert np.array_equal(
        flip_vertical(matrix_1), np.array([[3, 4], [1, 2]])
    ), "Flip vertical failed for 2x2 matrix"
    assert np.array_equal(
        flip_vertical(matrix_2), np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
    ), "Flip vertical failed for 3x3 matrix"
    assert (
        not flip_vertical.is_identity
    ), "Flip vertical function should have is_identity=False"
    print("Flip vertical function tests passed.")

    # Test rotate_180 function
    print("\nTesting rotate_180 function:")
    assert np.array_equal(
        rotate_180(matrix_1), np.array([[4, 3], [2, 1]])
    ), "Rotate 180 failed for 2x2 matrix"
    assert np.array_equal(
        rotate_180(matrix_2), np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    ), "Rotate 180 failed for 3x3 matrix"
    assert (
        not rotate_180.is_identity
    ), "Rotate 180 function should have is_identity=False"
    print("Rotate 180 function tests passed.")

    # Test composition of transformations
    print("\nTesting composition of transformations:")
    assert np.array_equal(
        rotate_180(flip_horizontal(matrix_1)), flip_vertical(matrix_1)
    ), "Rotate 180 of flip horizontal should equal flip vertical for 2x2 matrix"
    assert np.array_equal(
        rotate_180(flip_vertical(matrix_2)), flip_horizontal(matrix_2)
    ), "Rotate 180 of flip vertical should equal flip horizontal for 3x3 matrix"
    print("Composition of transformations tests passed.")

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    run_tests()
