# binary rule predictor data loader
import pickle
import numpy as np
import os
import json
import random

from .gilbert2d import flatten_2d_to_1d, unflatten_1d_to_2d
from .args import args

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

if args.simple:
    MAX_CONTEXT_LENGTH = 256
    MAX_PREDICTION_LENGTH = 32
else:
    MAX_CONTEXT_LENGTH = 1024
    MAX_PREDICTION_LENGTH = 256

MAX_SEQUENCE_LENGTH = MAX_CONTEXT_LENGTH + MAX_PREDICTION_LENGTH

evaluating_data = None

simple_dataset = "arc-datasets/datasets/bitdata/"



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

        random.shuffle(train_examples)  # Randomize train examples

        for test_example in test_examples:
            context = [START_SEQUENCE_TOKEN]
            # Use all other examples as part of the input context
            for ex in train_examples + test_examples:
                if ex != test_example:  # Exclude the current test case from context
                    input_matrix = np.atleast_2d(ex['input'])
                    output_matrix = np.atleast_2d(ex['output'])
                    input_flat = flatten_2d_to_1d(input_matrix)
                    output_flat = flatten_2d_to_1d(output_matrix)
                    context.extend([START_EXAMPLE_TOKEN, START_INPUT_MATRIX_TOKEN] + input_flat +
                                   [END_INPUT_MATRIX_TOKEN, START_OUTPUT_MATRIX_TOKEN] + output_flat +
                                   [END_OUTPUT_MATRIX_TOKEN, END_EXAMPLE_TOKEN])
            
            # Now process the actual test case
            test_input = np.atleast_2d(test_example['input'])
            test_input_flat = flatten_2d_to_1d(test_input)
            test_output_flat = flatten_2d_to_1d(np.atleast_2d(test_example['output']))

            context.extend([START_EXAMPLE_TOKEN, START_INPUT_MATRIX_TOKEN] + test_input_flat + [END_INPUT_MATRIX_TOKEN])
            context = pad_sequence(context, MAX_CONTEXT_LENGTH, PAD_TOKEN, left_pad=True)
            target = pad_sequence([START_OUTPUT_MATRIX_TOKEN] + test_output_flat + [END_OUTPUT_MATRIX_TOKEN, END_SEQUENCE_TOKEN], MAX_PREDICTION_LENGTH, PAD_TOKEN)

            dimensions = test_input.shape  # Already 2D due to atleast_2d
            print(f"Processed data: Dimensions: {dimensions}")

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
