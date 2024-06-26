import pickle
import numpy as np
import os
import json
import random
import torch

from .sequencing import flatten_2d_to_1d
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
NUM_TOKENS = END_SEQUENCE_TOKEN

SPECIAL_TOKENS = set(range(PAD_TOKEN, NUM_TOKENS))


def is_special_token(tensor, special_tokens):
    # Create a mask that is True where the tensor element is in special_tokens
    return torch.stack([tensor == tok for tok in special_tokens]).any(dim=0)


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
        return np.pad(
            sequence, (padding_length, 0), mode="constant", constant_values=pad_value
        )
    else:
        return np.pad(
            sequence, (0, padding_length), mode="constant", constant_values=pad_value
        )


def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
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
                    input_matrix = np.atleast_2d(ex["input"])
                    output_matrix = np.atleast_2d(ex["output"])
                    input_flat = flatten_2d_to_1d(input_matrix)
                    output_flat = flatten_2d_to_1d(output_matrix)
                    context.extend(
                        [START_EXAMPLE_TOKEN, START_INPUT_MATRIX_TOKEN]
                        + input_flat
                        + [END_INPUT_MATRIX_TOKEN, START_OUTPUT_MATRIX_TOKEN]
                        + output_flat
                        + [END_OUTPUT_MATRIX_TOKEN, END_EXAMPLE_TOKEN]
                    )

            # Now process the actual test case
            test_input = np.atleast_2d(test_example["input"])
            test_input_flat = flatten_2d_to_1d(test_input)
            test_output_flat = flatten_2d_to_1d(np.atleast_2d(test_example["output"]))

            context.extend(
                [START_EXAMPLE_TOKEN, START_INPUT_MATRIX_TOKEN]
                + test_input_flat
                + [END_INPUT_MATRIX_TOKEN]
                + [START_OUTPUT_MATRIX_TOKEN]
            )

            target = test_output_flat + [
                END_OUTPUT_MATRIX_TOKEN,
                END_SEQUENCE_TOKEN,
                END_EXAMPLE_TOKEN,
            ]

            print("context", context)
            print("target", target)
            # print the first and last token of each
            print("context first/last", context[0], context[-1])
            print("target first/last", target[0], target[-1])

            context = pad_sequence(
                context, MAX_CONTEXT_LENGTH, PAD_TOKEN, left_pad=True
            )
            target = pad_sequence(target, MAX_PREDICTION_LENGTH, PAD_TOKEN)

            dimensions = test_input.shape  # Already 2D due to atleast_2d
            print(f"Processed data: Dimensions: {dimensions}")

            processed_data.append((np.array(context), np.array(target), dimensions))

    return processed_data


def is_within_bounds(data, max_dim=9):
    """
    Check if any matrix in the train or test datasets exceeds the maximum dimensions.
    """
    for example in data["train"] + data["test"]:
        if any(dim > max_dim for dim in np.array(example["input"]).shape):
            return False
        if any(dim > max_dim for dim in np.array(example["output"]).shape):
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
arc_datasets_dir = os.path.join(
    os.path.dirname(training_data_dir), "../", "arc-datasets"
)
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
    with open(processed_training_file, "rb") as f:
        training_data = pickle.load(f)
    with open(processed_evaluating_file, "rb") as f:
        evaluating_data = pickle.load(f)
    print(f"Loaded {len(training_data)} training data points")
    print(f"Loaded {len(evaluating_data)} evaluation data points")
else:
    print("Processing data...")
    training_data = load_and_process_training_data(training_file_paths)
    evaluating_data = load_and_process_training_data(evaluating_file_paths)

    # Save processed data
    with open(processed_training_file, "wb") as f:
        pickle.dump(training_data, f)
    with open(processed_evaluating_file, "wb") as f:
        pickle.dump(evaluating_data, f)
    print("Processed data saved.")

print("Data loading completed.")
