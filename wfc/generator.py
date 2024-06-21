import json
import numpy as np
import os
from random import randint, choice
from .wfc import WaveFunctionCollapse
from PIL import Image
from matplotlib import pyplot as plt

# Create a function to generate random patterns
def generate_random_pattern(size, colors):
    pattern = np.random.choice(colors, size=size[1:])
    return pattern.astype(np.uint8)


# Save the pattern as an image
def save_pattern_as_image(pattern, filename):
    img = Image.fromarray(pattern.astype(np.uint8))
    img.save(filename)

# Generate input/output pairs
def generate_input_output_pairs(grid_size, pattern_size, num_examples, num_tests):
    examples = []
    tests = []

    for _ in range(num_examples):
        input_pattern = generate_random_pattern(pattern_size, list(range(10)))
        output_pattern = generate_random_pattern(pattern_size, list(range(10)))

        # Save patterns as images
        input_image_path = f"input_{_}.png"
        output_image_path = f"output_{_}.png"
        save_pattern_as_image(input_pattern, input_image_path)
        save_pattern_as_image(output_pattern, output_image_path)

        # Load samples
        input_sample = np.expand_dims(plt.imread(input_image_path), axis=0)[:, :, :, :3]
        output_sample = np.expand_dims(plt.imread(output_image_path), axis=0)[:, :, :, :3]

        wfc_input = WaveFunctionCollapse(grid_size, input_sample, pattern_size)
        wfc_output = WaveFunctionCollapse(grid_size, output_sample, pattern_size)

        input_image = wfc_input.get_image()
        output_image = wfc_output.get_image()

        examples.append({
            "input": input_image.tolist(),
            "output": output_image.tolist()
        })

    for _ in range(num_tests):
        input_pattern = generate_random_pattern(pattern_size, list(range(10)))

        # Save pattern as image
        input_image_path = f"test_input_{_}.png"
        save_pattern_as_image(input_pattern, input_image_path)

        # Load sample
        input_sample = np.expand_dims(plt.imread(input_image_path), axis=0)[:, :, :, :3]

        wfc_input = WaveFunctionCollapse(grid_size, input_sample, pattern_size)
        input_image = wfc_input.get_image()

        tests.append({
            "input": input_image.tolist(),
            "output": [[0]*grid_size[1] for _ in range(grid_size[0])]
        })

    return {"train": examples, "test": tests}

# Save the input/output pairs to a JSON file
def save_pairs_to_json(pairs, filename):
    with open(filename, 'w') as f:
        json.dump(pairs, f, indent=4)

# Main function
if __name__ == "__main__":
    num_examples = randint(2, 4)
    num_tests = randint(1, 2)
    grid_size = (1, randint(3, 30), randint(3, 30))
    pattern_size = (1, randint(2, 8), randint(2, 8))

    pairs = generate_input_output_pairs(grid_size, pattern_size, num_examples, num_tests)
    save_pairs_to_json(pairs, 'arc_agi.json')
