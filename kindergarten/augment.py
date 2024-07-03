import json
import os
import numpy as np
import hashlib
import argparse

def load_challenges(input_dir):
    challenges = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename), "r") as f:
                challenge = json.load(f)
                challenges.append(challenge)
    return challenges

def shift_colors(matrix, shift):
    return (np.array(matrix) + shift) % 10

def augment_challenge(challenge, augmentations):
    augmented_challenges = []
    for augment_input, augment_output in augmentations:
        augmented_challenge = {
            'train': [],
            'test': []
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
        augmented_challenges.append(augmented_challenge)
    return augmented_challenges

def save_challenge(challenge, output_dir):
    challenge_json = json.dumps(challenge, indent=4)
    hash_object = hashlib.sha256(challenge_json.encode())
    hash_hex = hash_object.hexdigest()
    output_file = os.path.join(output_dir, f"{hash_hex}.json")
    with open(output_file, "w") as f:
        f.write(challenge_json)

def process_challenges(input_dir, output_dir):
    challenges = load_challenges(input_dir)
    
    print("length of challenges")
    print(len(challenges))
    
    augmentations = [
        (lambda x: x, lambda x: x),
        (np.fliplr, np.fliplr),
        (np.flipud, np.flipud),
        (lambda x: x.T, lambda x: x.T),
        (lambda x: np.fliplr(x).T, lambda x: np.fliplr(x).T),
        (lambda x: np.flipud(x).T, lambda x: np.flipud(x).T),
        (lambda x: np.fliplr(x.T), lambda x: np.fliplr(x.T)),
        (lambda x: np.flipud(x.T), lambda x: np.flipud(x.T))
    ]
    
    for challenge in challenges:
        for shift in range(10):
            shifted_challenge = {
                'train': [],
                'test': []
            }
            for example in challenge['train']:
                shifted_example = {
                    'input': shift_colors(example['input'], shift).tolist(),
                    'output': shift_colors(example['output'], shift).tolist()
                }
                shifted_challenge['train'].append(shifted_example)
            for example in challenge['test']:
                shifted_example = {
                    'input': shift_colors(example['input'], shift).tolist(),
                    'output': shift_colors(example['output'], shift).tolist()
                }
                shifted_challenge['test'].append(shifted_example)
            augmented_challenges = augment_challenge(shifted_challenge, augmentations)
            for augmented_challenge in augmented_challenges:
                save_challenge(augmented_challenge, output_dir)

if __name__ == "__main__":
    print("Augmenting challenges...")
    parser = argparse.ArgumentParser(description="Process challenges and create color variations")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSON files")
    args = parser.parse_args()
    
    print(f"Input directory: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    process_challenges(args.input_dir, args.output_dir)