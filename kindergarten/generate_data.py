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

def generate_binary_matrix(shape):
    return np.random.randint(0, 2, size=shape).tolist()

def identity_kernel(shape):
    return np.eye(*shape)

def flip_kernel(shape):
    return 1 - np.eye(*shape)

def shift_left_kernel(shape):
    kernel = np.eye(*shape)
    return np.roll(kernel, -1, axis=1)

def shift_right_kernel(shape):
    kernel = np.eye(*shape)
    return np.roll(kernel, 1, axis=1)

def flip_bit_kernel(shape, bit_index):
    kernel = np.eye(*shape)
    if shape[1] > bit_index:
        kernel[0, bit_index] = 1 - kernel[0, bit_index]
    return kernel

def and_kernel(shape):
    return np.ones(shape)

def or_kernel(shape):
    return np.ones(shape)

def xor_kernel(shape):
    return np.ones(shape)

def not_kernel(shape):
    return 1 - np.eye(*shape)

def rotate_90_kernel(shape):
    return np.rot90(np.eye(*shape))

def rotate_180_kernel(shape):
    return np.rot90(np.eye(*shape), 2)

def rotate_270_kernel(shape):
    return np.rot90(np.eye(*shape), 3)

def mirror_kernel(shape):
    return np.fliplr(np.eye(*shape))

def mirror_rotate_90_kernel(shape):
    return np.rot90(np.fliplr(np.eye(*shape)))

def mirror_rotate_180_kernel(shape):
    return np.rot90(np.fliplr(np.eye(*shape)), 2)

def mirror_rotate_270_kernel(shape):
    return np.rot90(np.fliplr(np.eye(*shape)), 3)

def row_and_kernel(shape):
    return np.ones(shape)

def row_or_kernel(shape):
    return np.ones(shape)

def row_xor_kernel(shape):
    return np.ones(shape)

def row_not_kernel(shape):
    return 1 - np.eye(*shape)

def col_and_kernel(shape):
    return np.ones(shape)

def col_or_kernel(shape):
    return np.ones(shape)

def col_xor_kernel(shape):
    return np.ones(shape)

def col_not_kernel(shape):
    return 1 - np.eye(*shape)

def generate_example(shape, kernel_func, digits=(0, 1)):
    kernel = kernel_func(shape)
    input_matrix = [[random.choice(digits) for _ in range(shape[1])] for _ in range(shape[0])]
    output_matrix = apply_kernel(input_matrix, kernel, digits)
    
    return {
        'input': input_matrix,
        'output': output_matrix
    }

def generate_challenge(shape, kernel_func, num_train_pairs, digits=(0, 1)):
    train = [generate_example(shape, kernel_func, digits) for _ in range(num_train_pairs)]
    test = [generate_example(shape, kernel_func, digits)]  # Now a list with one example
    
    return {
        'train': train,
        'test': test,
        'digits': digits
    }


def generate_all_challenges(shape, kernels, all_digits=False):
    challenges = []
    digit_pairs = list(itertools.combinations(range(10), 2)) if all_digits else [(0, 1)]
    
    for kernel_name, kernel_func in kernels.items():
        for digits in digit_pairs:
            num_train_pairs = random.randint(2, 4)
            challenge = generate_challenge(shape, kernel_func, num_train_pairs, digits)
            challenge['rule'] = kernel_name
            challenges.append(challenge)
    return challenges

def apply_kernel(binary_matrix, kernel, digits):
    input_array = np.array([[1 if x == digits[1] else 0 for x in row] for row in binary_matrix])
    
    if input_array.shape == kernel.shape:
        output_array = (input_array * kernel) % 2
    elif input_array.shape[1] == kernel.shape[1]:
        output_array = (input_array + kernel) % 2
    else:
        output_array = np.dot(input_array, kernel) % 2

    output_array = output_array.astype(int)
    return [[digits[x] for x in row] for row in output_array]

def save_challenges(challenges, output_dir, train_ratio=0.9):
    train_dir = os.path.join(output_dir, "training")
    eval_dir = os.path.join(output_dir, "evaluation")
    
    for directory in [train_dir, eval_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    random.shuffle(challenges)
    split_index = int(len(challenges) * train_ratio)
    
    for i, challenge in enumerate(challenges):
        challenge_json = json.dumps(challenge, indent=4)
        hash_object = hashlib.sha256(challenge_json.encode())
        hash_hex = hash_object.hexdigest()
        
        if i < split_index:
            output_file = os.path.join(train_dir, f"{hash_hex}.json")
        else:
            output_file = os.path.join(eval_dir, f"{hash_hex}.json")
        
        with open(output_file, "w") as f:
            f.write(challenge_json)

def generate_bitdata(output_dir, all_digits=False):
    # 1x1 challenges
    kernels_1x1 = {
        "IDENTITY": identity_kernel,
        "FLIP": flip_kernel
    }
    challenges_1x1 = generate_all_challenges((1, 1), kernels_1x1, all_digits)
    
    # 1x2 challenges
    kernels_1x2 = {
        "IDENTITY": identity_kernel,
        "SHIFT_LEFT": shift_left_kernel,
        "SHIFT_RIGHT": shift_right_kernel,
        "FLIP_FIRST_BIT": flip_first_bit,
        "FLIP_SECOND_BIT": flip_second_bit,
        "AND": and_kernel,
        "OR": or_kernel,
        "XOR": xor_kernel,
        "NOT": not_kernel
    }
    challenges_1x2 = generate_all_challenges((1, 2), kernels_1x2, all_digits)
    
    save_challenges(challenges_1x1 + challenges_1x2, output_dir)

def generate_bitdata_grade1(output_dir, all_digits=False):
    # 1x3 challenges
    kernels_1x3 = {
        "IDENTITY": identity_kernel,
        "SHIFT_LEFT": shift_left_kernel,
        "SHIFT_RIGHT": shift_right_kernel,
        "FLIP_FIRST_BIT": flip_first_bit,
        "FLIP_SECOND_BIT": flip_second_bit,
        "FLIP_THIRD_BIT": flip_third_bit,
        "AND": and_kernel,
        "OR": or_kernel,
        "XOR": xor_kernel,
        "NOT": not_kernel
    }
    challenges_1x3 = generate_all_challenges((1, 3), kernels_1x3, all_digits)
    
    # 1x4 challenges
    kernels_1x4 = {
        "IDENTITY": identity_kernel,
        "SHIFT_LEFT": shift_left_kernel,
        "SHIFT_RIGHT": shift_right_kernel,
        "FLIP_FIRST_BIT": flip_first_bit,
        "FLIP_SECOND_BIT": flip_second_bit,
        "FLIP_THIRD_BIT": flip_third_bit,
        "FLIP_FOURTH_BIT": flip_fourth_bit,
        "AND": and_kernel,
        "OR": or_kernel,
        "XOR": xor_kernel,
        "NOT": not_kernel
    }
    challenges_1x4 = generate_all_challenges((1, 4), kernels_1x4, all_digits)
    
    # 2x2 challenges
    kernels_2x2 = {
        "IDENTITY": identity_kernel,
        "ROTATE_90": rotate_90_kernel,
        "ROTATE_180": rotate_180_kernel,
        "ROTATE_270": rotate_270_kernel,
        "MIRROR": mirror_kernel,
        "MIRROR_ROTATE_90": mirror_rotate_90_kernel,
        "MIRROR_ROTATE_180": mirror_rotate_180_kernel,
        "MIRROR_ROTATE_270": mirror_rotate_270_kernel,
        "ROW_AND": row_and_kernel,
        "ROW_OR": row_or_kernel,
        "ROW_XOR": row_xor_kernel,
        "ROW_NOT": row_not_kernel,
        "COL_AND": col_and_kernel,
        "COL_OR": col_or_kernel,
        "COL_XOR": col_xor_kernel,
        "COL_NOT": col_not_kernel
    }
    challenges_2x2 = generate_all_challenges((2, 2), kernels_2x2, all_digits)
    
    save_challenges(challenges_1x3 + challenges_1x4 + challenges_2x2, output_dir)



# Instead of directly using lambdas, define them with named functions for better traceability
def flip_first_bit(shape): return flip_bit_kernel(shape, 0)
flip_first_bit.__name__ = "flip_first_bit"

def flip_second_bit(shape): return flip_bit_kernel(shape, 1)
flip_second_bit.__name__ = "flip_second_bit"

def flip_third_bit(shape): return flip_bit_kernel(shape, 2)
flip_third_bit.__name__ = "flip_third_bit"

def flip_fourth_bit(shape): return flip_bit_kernel(shape, 3)
flip_fourth_bit.__name__ = "flip_fourth_bit"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bitdata challenges")
    parser.add_argument("--all_digits", action="store_true", help="Use all digits (0-9) instead of just 0 and 1")
    args = parser.parse_args()

    print("Generating bitdata_kindergarten...")
    generate_bitdata("kindergarten/datasets/kindergarten", args.all_digits)
    print("bitdata_kindergarten generated!")
    
    print("Generating bitdata_grade1...")
    generate_bitdata_grade1("kindergarten/datasets/kindergarten", args.all_digits)
    print("bitdata_grade1 generated!")
