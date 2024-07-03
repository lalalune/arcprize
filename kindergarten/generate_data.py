import itertools
import json
import os
import numpy as np
import random
import hashlib
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def identity_transform(shape):
    return shape

def flip_transform(shape):
    return (1 - shape).astype(int)

def shift_left_transform(shape):
    return np.roll(shape, -1, axis=1)

def shift_right_transform(shape):
    return np.roll(shape, 1, axis=1)

def flip_bit_transform(shape, bit_index):
    shape = shape.copy()
    if shape.shape[1] > bit_index:
        shape[0, bit_index] = 1 - shape[0, bit_index]
    return shape.astype(int)

def and_transform(shape):
    return np.all(shape, axis=0, keepdims=True).astype(int)

def or_transform(shape):
    return np.any(shape, axis=0, keepdims=True).astype(int)

def xor_transform(shape):
    return (np.sum(shape, axis=0, keepdims=True) % 2).astype(int)

def not_transform(shape):
    return (1 - shape).astype(int)

def rotate_90_transform(shape):
    return np.rot90(shape, 1).astype(int)

def rotate_180_transform(shape):
    return np.rot90(shape, 2).astype(int)

def rotate_270_transform(shape):
    return np.rot90(shape, 3).astype(int)

def mirror_transform(shape):
    return np.fliplr(shape).astype(int)

def mirror_rotate_90_transform(shape):
    return np.rot90(np.fliplr(shape), 1).astype(int)

def mirror_rotate_180_transform(shape):
    return np.rot90(np.fliplr(shape), 2).astype(int)

def mirror_rotate_270_transform(shape):
    return np.rot90(np.fliplr(shape), 3).astype(int)

def row_and_transform(shape):
    return np.all(shape, axis=1, keepdims=True).astype(int)

def row_or_transform(shape):
    return np.any(shape, axis=1, keepdims=True).astype(int)

def row_xor_transform(shape):
    return (np.sum(shape, axis=1, keepdims=True) % 2).astype(int)

def row_not_transform(shape):
    return (1 - np.all(shape, axis=1, keepdims=True)).astype(int)

def col_and_transform(shape):
    return np.all(shape, axis=0, keepdims=True).astype(int)

def col_or_transform(shape):
    return np.any(shape, axis=0, keepdims=True).astype(int)

def col_xor_transform(shape):
    return (np.sum(shape, axis=0, keepdims=True) % 2).astype(int)

def col_not_transform(shape):
    return (1 - np.all(shape, axis=0, keepdims=True)).astype(int)

def generate_example(shape, transform_func, digits=(0, 1)):
    input_matrix = np.array([[random.choice(digits) for _ in range(shape[1])] for _ in range(shape[0])])
    output_matrix = transform_func(input_matrix).astype(int)
    
    return {
        'input': input_matrix.tolist(),
        'output': output_matrix.tolist()
    }

def generate_challenge(shape, transform_func, num_train_pairs, digits=(0, 1), max_attempts=100):
    train = []
    examples_seen = set()
    attempts = 0
    
    while len(train) < num_train_pairs and attempts < max_attempts:
        example = generate_example(shape, transform_func, digits)
        example_hash = hashlib.sha256(str(example).encode()).hexdigest()
        
        if example_hash not in examples_seen:
            train.append(example)
            examples_seen.add(example_hash)
        
        attempts += 1
    
    test = [generate_example(shape, transform_func, digits)]
    
    return {
        'train': train,
        'test': test,
        'digits': digits
    }

def save_challenge(challenge, output_dir, augment=False):
    # Convert any boolean values to integers
    for example in challenge['train'] + challenge['test']:
        example['output'] = [[int(val) for val in row] for row in example['output']]
    
    challenge_json = json.dumps(challenge, separators=(',', ':'))
    hash_object = hashlib.sha256(challenge_json.encode())
    hash_hex = hash_object.hexdigest()
    
    output_file = os.path.join(output_dir, f"{hash_hex}.json")
    
    with open(output_file, "w") as f:
        f.write(challenge_json)
    
    if augment:
        augmentations = [
            lambda x: x,
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x),
            lambda x: x.T,
            lambda x: np.fliplr(x).T,
            lambda x: np.flipud(x).T,
            lambda x: np.fliplr(x.T),
            lambda x: np.flipud(x.T)
        ]
        
        digit_pairs = [(1, 9), (2, 7), (3, 4), (0, 4), (8, 0)]  # Define specific token pairs
        
        for digits in digit_pairs:
            for augment_input, augment_output in itertools.product(augmentations, repeat=2):
                augmented_challenge = {
                    'train': [],
                    'test': [],
                    'digits': list(digits),
                    'rule': challenge['rule']
                }
                
                for example in challenge['train']:
                    augmented_example = {
                        'input': augment_input(np.array(example['input'])).tolist(),
                        'output': augment_output(np.array(example['output'])).tolist()
                    }
                    augmented_challenge['train'].append(augmented_example)
                
                for example in challenge['test']:
                    augmented_example = {
                        'input': augment_input(np.array(example['input'])).tolist(),
                        'output': augment_output(np.array(example['output'])).tolist()
                    }
                    augmented_challenge['test'].append(augmented_example)
                
                augmented_json = json.dumps(augmented_challenge, separators=(',', ':'))
                augmented_hash = hashlib.sha256(augmented_json.encode()).hexdigest()
                augmented_output_file = os.path.join(output_dir, f"{augmented_hash}.json")
                
                with open(augmented_output_file, "w") as f:
                    f.write(augmented_json)

def generate_all_challenges(shape, transforms, max_challenges, output_dir, augment=False):
    challenges = []
    digit_pairs = [(0, 1)]
    
    if augment:
        # create every possible digit pair
        digit_pair_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        digit_pair_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        digit_pairs = []
        
        # for each possible digit pair, add it to the list
        for digit_1 in digit_pair_1:
            for digit_2 in digit_pair_2:
                if digit_1 != digit_2:
                    digit_pairs.append((digit_1, digit_2))
    
    for transform_name, transform_func in transforms.items():
        for digits in digit_pairs:
            if len(challenges) >= max_challenges:
                return challenges
            
            num_train_pairs = random.randint(3, 5)
            challenge = generate_challenge(shape, transform_func, num_train_pairs, digits)
            
            if len(challenge['train']) == num_train_pairs:
                challenge['rule'] = transform_name
                challenges.append(challenge)
                save_challenge(challenge, output_dir, augment)
    
    return challenges

def generate_bitdata(output_dir, max_challenges, augment):
    # 1x1 challenges
    transforms_1x1 = {
        "IDENTITY": identity_transform,
        "FLIP": flip_transform
    }
    challenges_1x1 = generate_all_challenges((1, 1), transforms_1x1, max_challenges, output_dir, augment)
    
    # 1x2 challenges
    transforms_1x2 = {
        "IDENTITY": identity_transform,
        "SHIFT_LEFT": shift_left_transform,
        "SHIFT_RIGHT": shift_right_transform,
        "FLIP_FIRST_BIT": lambda shape: flip_bit_transform(shape, 0),
        "FLIP_SECOND_BIT": lambda shape: flip_bit_transform(shape, 1),
        "AND": and_transform,
        "OR": or_transform,
        "XOR": xor_transform,
        "NOT": not_transform
    }
    challenges_1x2 = generate_all_challenges((1, 2), transforms_1x2, max_challenges, output_dir, augment)

def generate_bitdata_grade1(output_dir, max_challenges, augment):
    # 1x3 challenges
    transforms_1x3 = {
        "IDENTITY": identity_transform,
        "SHIFT_LEFT": shift_left_transform,
        "SHIFT_RIGHT": shift_right_transform,
        "FLIP_FIRST_BIT": lambda shape: flip_bit_transform(shape, 0),
        "FLIP_SECOND_BIT": lambda shape: flip_bit_transform(shape, 1),
        "FLIP_THIRD_BIT": lambda shape: flip_bit_transform(shape, 2),
        "AND": and_transform,
        "OR": or_transform,
        "XOR": xor_transform,
        "NOT": not_transform
    }
    challenges_1x3 = generate_all_challenges((1, 3), transforms_1x3, max_challenges, output_dir, augment)
    
    # 1x4 challenges
    transforms_1x4 = {
        "IDENTITY": identity_transform,
        "SHIFT_LEFT": shift_left_transform,
        "SHIFT_RIGHT": shift_right_transform,
        "FLIP_FIRST_BIT": lambda shape: flip_bit_transform(shape, 0),
        "FLIP_SECOND_BIT": lambda shape: flip_bit_transform(shape, 1),
        "FLIP_THIRD_BIT": lambda shape: flip_bit_transform(shape, 2),
        "FLIP_FOURTH_BIT": lambda shape: flip_bit_transform(shape, 3),
        "AND": and_transform,
        "OR": or_transform,
        "XOR": xor_transform,
        "NOT": not_transform
    }
    challenges_1x4 = generate_all_challenges((1, 4), transforms_1x4, max_challenges, output_dir, augment)
    
    # 2x2 challenges
    transforms_2x2 = {
        "IDENTITY": identity_transform,
        "ROTATE_90": rotate_90_transform,
        "ROTATE_180": rotate_180_transform,
        "ROTATE_270": rotate_270_transform,
        "MIRROR": mirror_transform,
        "MIRROR_ROTATE_90": mirror_rotate_90_transform,
        "MIRROR_ROTATE_180": mirror_rotate_180_transform,
        "MIRROR_ROTATE_270": mirror_rotate_270_transform,
        "ROW_AND": row_and_transform,
        "ROW_OR": row_or_transform,
        "ROW_XOR": row_xor_transform,
        "ROW_NOT": row_not_transform,
        "COL_AND": col_and_transform,
        "COL_OR": col_or_transform,
        "COL_XOR": col_xor_transform,
        "COL_NOT": col_not_transform
    }
    challenges_2x2 = generate_all_challenges((2, 2), transforms_2x2, max_challenges, output_dir, augment)

def generate_bitdata_grade2(output_dir, max_challenges, augment):
    # 1x5 challenges
    transforms_1x5 = {
        "IDENTITY": identity_transform,
        "SHIFT_LEFT": shift_left_transform,
        "SHIFT_RIGHT": shift_right_transform,
        "FLIP_FIRST_BIT": lambda shape: flip_bit_transform(shape, 0),
        "FLIP_SECOND_BIT": lambda shape: flip_bit_transform(shape, 1),
        "FLIP_THIRD_BIT": lambda shape: flip_bit_transform(shape, 2),
        "FLIP_FOURTH_BIT": lambda shape: flip_bit_transform(shape, 3),
        "FLIP_FIFTH_BIT": lambda shape: flip_bit_transform(shape, 4),
        "AND": and_transform,
        "OR": or_transform,
        "XOR": xor_transform,
        "NOT": not_transform
    }
    challenges_1x5 = generate_all_challenges((1, 5), transforms_1x5, max_challenges, output_dir, augment)
    
    # 1x6 challenges
    transforms_1x6 = {
        "IDENTITY": identity_transform,
        "SHIFT_LEFT": shift_left_transform,
        "SHIFT_RIGHT": shift_right_transform,
        "FLIP_FIRST_BIT": lambda shape: flip_bit_transform(shape, 0),
        "FLIP_SECOND_BIT": lambda shape: flip_bit_transform(shape, 1),
        "FLIP_THIRD_BIT": lambda shape: flip_bit_transform(shape, 2),
        "FLIP_FOURTH_BIT": lambda shape: flip_bit_transform(shape, 3),
        "FLIP_FIFTH_BIT": lambda shape: flip_bit_transform(shape, 4),
        "FLIP_SIXTH_BIT": lambda shape: flip_bit_transform(shape, 5),
        "AND": and_transform,
        "OR": or_transform,
        "XOR": xor_transform,
        "NOT": not_transform
    }
    challenges_1x6 = generate_all_challenges((1, 6), transforms_1x6, max_challenges, output_dir, augment)
    
    # 2x3 challenges
    transforms_2x3 = {
        "IDENTITY": identity_transform,
        "ROTATE_90": rotate_90_transform,
        "ROTATE_180": rotate_180_transform,
        "ROTATE_270": rotate_270_transform,
        "MIRROR": mirror_transform,
        "MIRROR_ROTATE_90": mirror_rotate_90_transform,
        "MIRROR_ROTATE_180": mirror_rotate_180_transform,
        "MIRROR_ROTATE_270": mirror_rotate_270_transform,
        "ROW_AND": row_and_transform,
        "ROW_OR": row_or_transform,
        "ROW_XOR": row_xor_transform,
        "ROW_NOT": row_not_transform,
        "COL_AND": col_and_transform,
        "COL_OR": col_or_transform,
        "COL_XOR": col_xor_transform,
        "COL_NOT": col_not_transform
    }
    challenges_2x3 = generate_all_challenges((2, 3), transforms_2x3, max_challenges, output_dir, augment)
    
    # 3x3 challenges
    transforms_3x3 = transforms_2x3  # Same transforms as 2x3
    challenges_3x3 = generate_all_challenges((3, 3), transforms_3x3, max_challenges, output_dir, augment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 1D-ARC challenges")
    parser.add_argument("--augment", action="store_true", help="Use all digits (0-9) instead of just 0 and 1, and create augmented versions")
    parser.add_argument("--max_challenges", type=int, default=100, help="Maximum number of challenges to generate")
    args = parser.parse_args()

    output_dir = "arc-datasets/datasets/kindergarten2/data/training"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating bitdata_kindergarten (max {args.max_challenges} challenges)...")
    generate_bitdata(output_dir, args.max_challenges, args.augment)
    print("bitdata_kindergarten generated!")
    
    print(f"Generating bitdata_grade1 (max {args.max_challenges} challenges)...")
    generate_bitdata_grade1(output_dir, args.max_challenges, args.augment)
    print("bitdata_grade1 generated!")

    print(f"Generating bitdata_grade2 (max {args.max_challenges} challenges)...")
    generate_bitdata_grade2(output_dir, args.max_challenges, args.augment)
    print("bitdata_grade2 generated!")