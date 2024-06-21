import numpy as np
import os
import json
from .gilbert2d import flatten_2d_to_1d
import random
import hashlib
from .transformations import identity, flip_horizontal, flip_vertical, rotate_180

PAD_TOKEN = 10
START_EXAMPLE_TOKEN = 11
END_EXAMPLE_TOKEN = 12
START_SEQUENCE_TOKEN = 13
END_SEQUENCE_TOKEN = 14
NUM_TOKENS = 15

augmentation_functions = [identity, flip_horizontal, flip_vertical, rotate_180]


def apply_transform(example, transform_func):
    return {
        "input": transform_func(np.array(example["input"])).tolist(),
        "output": transform_func(np.array(example["output"])).tolist(),
    }


def flip_input_output(example):
    return {"input": example["output"], "output": example["input"]}


def generate_random_token_mapping():
    tokens = list(range(1, 10))
    shuffled = tokens.copy()
    random.shuffle(shuffled)
    return dict(zip(tokens, shuffled))


def apply_token_mapping(example, mapping):
    def map_array(arr):
        return [[mapping.get(val, val) for val in row] for row in arr]

    return {
        "input": map_array(example["input"]),
        "output": map_array(example["output"]),
    }


def is_valid_size(matrix):
    return len(matrix) <= 16 and all(len(row) <= 16 for row in matrix)


def generate_all_augmentations(example, num_augmentations=6):
    if not is_valid_size(example["input"]) or not is_valid_size(example["output"]):
        return []  # Skip this example if it's too large

    augmentations = [example]  # Include the original, unaugmented example

    for _ in range(num_augmentations):
        transform_func = random.choice(augmentation_functions)
        augmented = apply_transform(example, transform_func)

        if not transform_func.is_identity:
            token_mapping = generate_random_token_mapping()
            augmented = apply_token_mapping(augmented, token_mapping)

        if random.random() < 0.5:  # 50% chance to flip input-output
            augmented = flip_input_output(augmented)

        if is_valid_size(augmented["input"]) and is_valid_size(augmented["output"]):
            augmentations.append(augmented)

    return augmentations


def remove_zeros_from_sequence(sequence):
    return [x for x in sequence if x != 0]


def generate_contexts(train_example, max_context_length=4096):
    context = [START_SEQUENCE_TOKEN]
    unpadded_context = [START_SEQUENCE_TOKEN]

    context.append(START_EXAMPLE_TOKEN)
    unpadded_context.append(START_EXAMPLE_TOKEN)

    train_input = flatten_2d_to_1d(np.array(train_example["input"]))
    train_output = flatten_2d_to_1d(np.array(train_example["output"]))

    context.extend(train_input)
    context.extend(train_output)
    unpadded_context.extend(train_input)
    unpadded_context.extend(train_output)

    context.append(END_EXAMPLE_TOKEN)
    unpadded_context.append(END_EXAMPLE_TOKEN)

    context.append(END_SEQUENCE_TOKEN)
    unpadded_context.append(END_SEQUENCE_TOKEN)

    # Pad the context if necessary
    if len(context) < max_context_length:
        context = context + [PAD_TOKEN] * (max_context_length - len(context))
    else:
        context = context[:max_context_length]

    target = [START_SEQUENCE_TOKEN] + train_output + [END_SEQUENCE_TOKEN]
    target = target + [PAD_TOKEN] * (max_context_length - len(target))

    return [(np.array(context), np.array(target))]


def generate_zero_removed_variant(context, target):
    zero_removed_context = remove_zeros_from_sequence(context.tolist())
    zero_removed_target = remove_zeros_from_sequence(target.tolist())

    # Ensure padding to max_context_length
    if len(zero_removed_context) < len(context):
        zero_removed_context += [PAD_TOKEN] * (len(context) - len(zero_removed_context))
    if len(zero_removed_target) < len(target):
        zero_removed_target += [PAD_TOKEN] * (len(target) - len(zero_removed_target))

    return np.array(zero_removed_context), np.array(zero_removed_target)


def load_and_process_training_data(file_paths, max_context_length=8192):
    processed_data = []
    unpadded_strings = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        original_train = data["train"]

        all_train_data = []
        for ex in original_train:
            all_train_data.extend(generate_all_augmentations(ex, num_augmentations=6))

        unique_train_data = []
        unique_examples = set()
        for ex in all_train_data:
            # Make a hash of the example to ensure uniqueness
            ex_hash = hashlib.md5(str(ex).encode()).hexdigest()
            if ex_hash not in unique_examples:
                unique_train_data.append(ex)
                unique_examples.add(ex_hash)

        for train_example in unique_train_data:
            contexts = generate_contexts(train_example, max_context_length)
            for context, target in contexts:
                processed_data.append((context, target))
                unpadded_strings.append(" ".join(map(str, context)))

            # Generate and add the zero-removed variant
            zero_removed_context, zero_removed_target = generate_zero_removed_variant(
                contexts[0][0], contexts[0][1]
            )
            processed_data.append((zero_removed_context, zero_removed_target))
            unpadded_strings.append(" ".join(map(str, zero_removed_context)))

        print(f"Processed training file: {file_path}")
        print(f"  Original training examples: {len(original_train)}")
        print(f"  Unique augmented training examples: {len(unique_train_data)}")
        print(f"  Total processed examples: {len(unique_train_data)}")
        print("---")

    with open("hilbert_training_data.txt", "w") as f:
        for string in unpadded_strings:
            f.write(string + "\n")

    print(f"Total processed training data points: {len(processed_data)}")
    return processed_data


def load_and_process_eval_data(file_paths, max_context_length=8192):
    processed_data = []
    unpadded_strings = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        train_examples = data["train"]
        test_examples = data["test"]

        # Process each test example
        for test_example in test_examples:
            context = [START_SEQUENCE_TOKEN]
            unpadded_context = [START_SEQUENCE_TOKEN]

            # Add all training examples to the context
            for train_example in train_examples:
                context.append(START_EXAMPLE_TOKEN)
                unpadded_context.append(START_EXAMPLE_TOKEN)

                train_input = flatten_2d_to_1d(np.array(train_example["input"]))
                train_output = flatten_2d_to_1d(np.array(train_example["output"]))
                context.extend(train_input)
                context.extend(train_output)
                unpadded_context.extend(train_input)
                unpadded_context.extend(train_output)

                context.append(END_EXAMPLE_TOKEN)
                unpadded_context.append(END_EXAMPLE_TOKEN)

            # Add the test input
            context.append(START_EXAMPLE_TOKEN)
            unpadded_context.append(START_EXAMPLE_TOKEN)
            test_input = flatten_2d_to_1d(np.array(test_example["input"]))
            context.extend(test_input)
            unpadded_context.extend(test_input)
            context.append(END_EXAMPLE_TOKEN)
            unpadded_context.append(END_EXAMPLE_TOKEN)

            context.append(END_SEQUENCE_TOKEN)
            unpadded_context.append(END_SEQUENCE_TOKEN)

            # Pad the context if necessary
            if len(context) < max_context_length:
                context = context + [PAD_TOKEN] * (max_context_length - len(context))
            else:
                context = context[:max_context_length]

            target = (
                [START_SEQUENCE_TOKEN]
                + flatten_2d_to_1d(np.array(test_example["output"]))
                + [END_SEQUENCE_TOKEN]
            )
            target = target + [PAD_TOKEN] * (max_context_length - len(target))

            processed_data.append((np.array(context), np.array(target)))
            unpadded_strings.append(" ".join(map(str, unpadded_context)))

        print(f"Processed eval file: {file_path}")
        print(f"  Training examples: {len(train_examples)}")
        print(f"  Test examples: {len(test_examples)}")
        print(f"  Total processed examples: {len(test_examples)}")
        print("---")

    with open("hilbert_eval_data.txt", "w") as f:
        for string in unpadded_strings:
            f.write(string + "\n")

    print(f"Total processed eval data points: {len(processed_data)}")
    return processed_data


# Load and process data
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

# Check if processed data files exist
processed_training_file = "processed_training_data.npy"
processed_evaluating_file = "processed_evaluating_data.npy"

if os.path.exists(processed_training_file) and os.path.exists(
    processed_evaluating_file
):
    print("Loading pre-processed data...")
    training_data = np.load(processed_training_file, allow_pickle=True)
    evaluating_data = np.load(processed_evaluating_file, allow_pickle=True)
    print(f"Loaded {len(training_data)} training data points")
    print(f"Loaded {len(evaluating_data)} evaluation data points")
else:
    print("Processing data...")
    training_data = load_and_process_training_data(
        training_file_paths, max_context_length=4096
    )
    evaluating_data = load_and_process_eval_data(
        evaluating_file_paths, max_context_length=4096
    )

    # Save processed data
    np.save(processed_training_file, training_data)
    np.save(processed_evaluating_file, evaluating_data)
    print("Processed data saved.")

print("Data loading completed.")
