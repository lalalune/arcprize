import numpy as np
import os

from ..sequencing import unflatten_1d_to_2d
from ..data import training_data_dir, load_and_process_training_data, process_data


def analyze_dataset(data):
    """
    Analyze the dataset to find:
    1. Maximum number of unique tokens.
    2. Maximum example width and height.
    3. Maximum sequence length.
    """
    max_tokens = 0
    max_width = 0
    max_height = 0
    max_sequence_length = 0
    token_set = set()

    print("data", data[0])

    for context, target, dimensions in data:
        # Update maximum sequence lengths
        # strip 10s from convert (padding tokens)
        stripped_context = [x for x in context if x != 10]
        # strip the target
        stripped_target = [x for x in target if x != 10]
        max_sequence_length = max(
            max_sequence_length, len(stripped_context), len(stripped_target)
        )

        # Flatten the list to find unique tokens
        tokens = set(context).union(set(target))
        token_set.update(tokens)

        # Check all matrices
        for item in [
            context,
            target,
        ]:  # Assuming context and target include raw matrices
            print("width and height are", dimensions[0], dimensions[1])
            # strip pad tokens
            # extract the matrix by removing all other special tokens
            item = [
                x for x in item if x not in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            ]
            print("item is")
            print(item)
            array_2d = unflatten_1d_to_2d(
                item, width=dimensions[0], height=dimensions[1]
            )  # Adjust width and height if known differently
            height, width = np.array(array_2d).shape
            max_height = max(max_height, height)
            max_width = max(max_width, width)

    max_tokens = len(token_set)
    print("Maximum number of unique tokens:", max_tokens)
    print("Maximum width and height:", max_width, max_height)
    print("Maximum sequence length:", max_sequence_length)


if __name__ == "__main__":
    training_file_paths = [
        os.path.join(training_data_dir, f)
        for f in os.listdir(training_data_dir)
        if f.endswith(".json")
    ]
    training_data = load_and_process_training_data(training_file_paths)

    test_data_1x1 = {
        "train": [{"input": [[1]], "output": [[0]]}],
        "test": [{"input": [[1]], "output": [[0]]}],
    }
    test_data_2x2 = {
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
    }

    print("test_data_1x1", test_data_1x1)

    processed_1x1 = process_data([test_data_1x1])
    processed_2x2 = process_data([test_data_2x2])

    print("1x1 test data:")
    print(f"Input: {processed_1x1[0][0]}")
    print(f"Output: {processed_1x1[0][1]}")
    print(f"Dimensions: {processed_1x1[0][2]}")

    print("2x2 test data:")
    print(f"Input: {processed_2x2[0][0]}")
    print(f"Output: {processed_2x2[0][1]}")
    print(f"Dimensions: {processed_2x2[0][2]}")
    analyze_dataset(training_data)
