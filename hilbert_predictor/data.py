# binary rule predictor data loader
import pickle
import numpy as np
import os
import json
import numpy as np
import torch
import random

from .args import args
from .encoder import PositionEncoder, NUM_ENCODING_DIMENSIONS

# Token definitions
PAD_TOKEN = 10
START_EXAMPLE_TOKEN = 11
END_EXAMPLE_TOKEN = 12
START_INPUT_MATRIX_TOKEN = 13
END_INPUT_MATRIX_TOKEN = 14
START_OUTPUT_MATRIX_TOKEN = 15
END_OUTPUT_MATRIX_TOKEN = 16
START_SEQUENCE_TOKEN = 17
END_SEQUENCE_TOKEN = 18
MASK = 19
NUM_TOKENS = END_SEQUENCE_TOKEN + 1

MAX_CONTEXT_LENGTH = 1024
MAX_SEQUENCE_LENGTH = 1024 + 128
MAX_PREDICTION_LENGTH = 128

evaluating_data = None

simple_dataset = "arc-datasets/datasets/bitdata/"


# Gilbert2D - Generalized Hilbert Curve for 2D space-filling
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
            # 2D matrix: flatten using Hilbert curve
            array_1d = [None] * (width * height)
            for idx, (x, y) in enumerate(gilbert2d(width, height)):
                array_1d[idx] = array[y][x]
            return array_1d
    else:
        raise ValueError(f"Input array must be 1D or 2D, got {array.ndim}D")

def create_mapping_table():
    mapping_table = {}
    position_encoder = PositionEncoder(30, 30)
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

def unflatten_1d_to_2d(array_1d, width, height):
    array_2d = [[None] * width for _ in range(height)]
    for idx, (x, y) in enumerate(gilbert2d(width, height)):
        array_2d[y][x] = array_1d[idx]

    return array_2d

def pad_sequence(sequence, max_length, pad_value, left_pad=False):
    actual_length = len(sequence)
    padding_length = max(0, max_length - actual_length)
    if left_pad:
        return np.pad(sequence, (padding_length, 0), mode='constant', constant_values=pad_value)
    else:
        return np.pad(sequence, (0, padding_length), mode='constant', constant_values=pad_value)

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def process_data(data_list):
    processed_data = []
    for data in data_list:
        train_examples = data["train"]
        test_examples = data["test"]

        # Randomize train examples to prevent sequence memorization
        random.shuffle(train_examples)

        for test_example in test_examples:
            context = [START_SEQUENCE_TOKEN]
            # Use all other examples as part of the input context
            for ex in train_examples + test_examples:
                if ex != test_example:  # Exclude the current test case from context
                    input_flat = flatten_2d_to_1d(np.array(ex['input']))
                    output_flat = flatten_2d_to_1d(np.array(ex['output']))
                    context.extend([START_EXAMPLE_TOKEN, START_INPUT_MATRIX_TOKEN] + input_flat +
                                   [END_INPUT_MATRIX_TOKEN, START_OUTPUT_MATRIX_TOKEN] + output_flat +
                                   [END_OUTPUT_MATRIX_TOKEN, END_EXAMPLE_TOKEN])
            
            # Now process the actual test case
            test_input = np.array(test_example['input'])
            test_input_flat = flatten_2d_to_1d(test_input)
            test_output_flat = flatten_2d_to_1d(np.array(test_example['output']))

            context.extend([START_EXAMPLE_TOKEN, START_INPUT_MATRIX_TOKEN] + test_input_flat + [END_INPUT_MATRIX_TOKEN])
            context = pad_sequence(context, MAX_CONTEXT_LENGTH, PAD_TOKEN, left_pad=True)
            target = pad_sequence([START_OUTPUT_MATRIX_TOKEN] + test_output_flat + [END_OUTPUT_MATRIX_TOKEN, END_SEQUENCE_TOKEN], MAX_PREDICTION_LENGTH, PAD_TOKEN)
            
            dimensions = test_input.shape  # Store the dimensions of the input matrix
            print(f"Context: {context}")
            print(f"Target: {target}")
            print(f"Dimensions: {dimensions}")  # This should now correctly print dimensions without error

            # Append the tuple with dimensions
            processed_data.append((np.array(context), np.array(target), dimensions))

    return processed_data

def is_within_bounds(data, max_dim=9):
    """
    Check if any matrix in the train or test datasets exceeds the maximum dimensions.
    """
    for example in data['train'] + data['test']:
        if any(dim > max_dim for dim in np.array(example['input']).shape):
            return False
        if any(dim > max_dim for dim in np.array(example['output']).shape):
            return False
    return True

def load_and_process_training_data(file_paths):
    processed_data = []
    for file_path in file_paths:
        print("Loading file:", file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
            if is_within_bounds(data):
                processed_data.extend(process_data([data]))
            else:
                print("Skipped due to exceeding dimension limits:", file_path)
    
    print(f"Total processed data points: {len(processed_data)}")
    return processed_data

if args.simple:
    training_data_dir = simple_dataset + "training"
    evaluating_data_dir = simple_dataset + "evaluation"

else:
    # Rest of the code remains the same
    training_data_dir = "./data/training"
    evaluating_data_dir = "./data/evaluation"

# if 
training_file_paths = [
    os.path.join(training_data_dir, f)
    for f in os.listdir(training_data_dir)
    if f.endswith(".json")
]
evaluating_file_paths = [
    os.path.join(evaluating_data_dir, f)
    for f in os.listdir(evaluating_data_dir)
    if f.endswith(".json")
]

# Check for arc-datasets folder
arc_datasets_dir = os.path.join(os.path.dirname(training_data_dir), "../", "arc-datasets")
if os.path.exists(arc_datasets_dir):
    datasets_dir = os.path.join(arc_datasets_dir, "datasets")
    if os.path.exists(datasets_dir):
        print("Found arc-datasets folder. Adding files...")
        additional_files = find_json_files(datasets_dir)
        training_file_paths.extend(additional_files)
        print(f"Added {len(additional_files)} files from arc-datasets")
    else:
        print("arc-datasets folder found, but 'datasets' subdirectory not found.")
else:
    print("arc-datasets folder not found. Proceeding with original data only.")
    print("arc_datasets_dir", arc_datasets_dir)

if args.simple:
    processed_training_file = "processed_training_data_simple.pkl"
    processed_evaluating_file = "processed_evaluating_data_simple.pkl"
else:
    # Check if processed data files exist
    processed_training_file = "processed_training_data.pkl"
    processed_evaluating_file = "processed_evaluating_data.pkl"

if os.path.exists(processed_training_file) and os.path.exists(
    processed_evaluating_file
):
    print("Loading pre-processed data...")
    with open(processed_training_file, 'rb') as f:
        training_data = pickle.load(f)
    with open(processed_evaluating_file, 'rb') as f:
        evaluating_data = pickle.load(f)
    print(f"Loaded {len(training_data)} training data points")
    print(f"Loaded {len(evaluating_data)} evaluation data points")
else:
    print("Processing data...")
    training_data = load_and_process_training_data(
        training_file_paths
    )
    evaluating_data = load_and_process_training_data(
        evaluating_file_paths
    )

    # Save processed data
    with open(processed_training_file, 'wb') as f:
        pickle.dump(training_data, f)
    with open(processed_evaluating_file, 'wb') as f:
        pickle.dump(evaluating_data, f)
    print("Processed data saved.")

print("Data loading completed.")

# Print a few lines to verify the data
print("Training data examples:")
for i in range(min(3, len(training_data))):
    print(f"Input (length {len(training_data[i][0])}): {training_data[i][0]}")
    print(f"Output (length {len(training_data[i][1])}): {training_data[i][1]}")
    print("---")

def test_mapping():
    mapping_table = load_mapping_table()
    for height in range(1, 31):
        for width in range(1, 31):
            mapping, encodings = mapping_table[(height, width)]
            assert len(mapping) == height * width, f"Incorrect mapping length for {height}x{width}"
            assert encodings.shape == (height, width, NUM_ENCODING_DIMENSIONS), f"Incorrect encoding shape for {height}x{width}: {encodings.shape}"
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
            reconstructed = [[0] * width for _ in range(height)]
            for idx, value in enumerate(flattened):
                x, y = mapping_table[(height, width)][0][idx]  # Get the x, y coordinates from the tuple
                reconstructed[y][x] = value

        assert matrix == reconstructed, f"Test failed for matrix: {matrix}"
        print(f"Test passed for {height}x{width} matrix")



    # Test cases
    test_case([[1]])  # 1x1
    test_case([[1, 2, 3]])  # 1x3
    test_case([[1], [2], [3]])  # 3x1
    test_case([[1, 2], [3, 4]])  # 2x2
    test_case([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3
    test_case([[i for i in range(j*5+1, (j+1)*5+1)] for j in range(5)])  # 5x5

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

    print('data', data[0])

    for context, target, dimensions in data:
        # Update maximum sequence lengths
        # strip 10s from convert (padding tokens)
        stripped_context = [x for x in context if x != 10]
        # strip the target
        stripped_target = [x for x in target if x != 10]
        max_sequence_length = max(max_sequence_length, len(stripped_context), len(stripped_target))
        
        # Flatten the list to find unique tokens
        tokens = set(context).union(set(target))
        token_set.update(tokens)

        # Check all matrices
        for item in [context, target]:  # Assuming context and target include raw matrices
            array_2d = unflatten_1d_to_2d(item, width=dimensions[0], height=dimensions[1])  # Adjust width and height if known differently
            height, width = np.array(array_2d).shape
            max_height = max(max_height, height)
            max_width = max(max_width, width)

    max_tokens = len(token_set)
    print("Maximum number of unique tokens:", max_tokens)
    print("Maximum width and height:", max_width, max_height)
    print("Maximum sequence length:", max_sequence_length)

# Usage
if __name__ == "__main__":
    training_file_paths = [
        os.path.join(training_data_dir, f)
        for f in os.listdir(training_data_dir)
        if f.endswith(".json")
    ]
    training_data = load_and_process_training_data(training_file_paths)
    analyze_dataset(training_data)

    mapping_table_file = 'mapping_table.pkl'
    
    if os.path.exists(mapping_table_file):
        print("Loading existing mapping table...")
        mapping_table = load_mapping_table(mapping_table_file)
    else:
        print("Creating new mapping table...")
        mapping_table = create_mapping_table()
        save_mapping_table(mapping_table, mapping_table_file)
    
    print("Running tests...")
    test_mapping()
    
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
